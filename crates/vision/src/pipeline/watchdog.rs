//! Watchdog responsible for detecting stalled pipeline stages and triggering
//! restarts.
//!
//! The watchdog tracks heartbeats emitted by the capture, processing, and
//! encoding stages. When any stage stops beating the pipeline gracefully shuts
//! down and the supervisor restarts it.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use tracing::error;

use crate::pipeline::telemetry;

/// Sleep interval between watchdog health checks.
pub(crate) const WATCHDOG_POLL_INTERVAL_MS: u64 = 500;
/// Time without a heartbeat before a component is considered stalled.
pub(crate) const WATCHDOG_STALE_THRESHOLD_MS: u64 = 1_500;
/// Grace period at startup allowing components to warm up before monitoring.
pub(crate) const WATCHDOG_STARTUP_GRACE_MS: u64 = 5_000;

#[derive(Copy, Clone, Debug)]
/// Logical components monitored by the watchdog.
pub(crate) enum HealthComponent {
    Capture,
    Processor,
    Encoder,
}

impl HealthComponent {
    /// Human readable label used in log messages and metrics.
    pub(crate) fn label(self) -> &'static str {
        match self {
            HealthComponent::Capture => "capture",
            HealthComponent::Processor => "processing",
            HealthComponent::Encoder => "encoding",
        }
    }
}

pub(crate) struct PipelineHealth {
    capture: AtomicU64,
    processor: AtomicU64,
    encoder: AtomicU64,
}

impl PipelineHealth {
    /// Initialise the health tracker with grace periods for each component.
    pub(crate) fn new() -> Self {
        let now = current_millis();
        let grace_deadline = now.saturating_add(WATCHDOG_STARTUP_GRACE_MS);
        Self {
            capture: AtomicU64::new(grace_deadline),
            processor: AtomicU64::new(grace_deadline),
            encoder: AtomicU64::new(grace_deadline),
        }
    }

    /// Register a heartbeat for the supplied component.
    pub(crate) fn beat(&self, component: HealthComponent) {
        let now = current_millis();
        match component {
            HealthComponent::Capture => self.capture.store(now, Ordering::Relaxed),
            HealthComponent::Processor => self.processor.store(now, Ordering::Relaxed),
            HealthComponent::Encoder => self.encoder.store(now, Ordering::Relaxed),
        }
    }

    /// Returns the first component that has not produced a heartbeat recently.
    pub(crate) fn stale_component(&self, now: u64) -> Option<HealthComponent> {
        if now.saturating_sub(self.capture.load(Ordering::Relaxed)) > WATCHDOG_STALE_THRESHOLD_MS {
            return Some(HealthComponent::Capture);
        }
        if now.saturating_sub(self.processor.load(Ordering::Relaxed)) > WATCHDOG_STALE_THRESHOLD_MS
        {
            return Some(HealthComponent::Processor);
        }
        if now.saturating_sub(self.encoder.load(Ordering::Relaxed)) > WATCHDOG_STALE_THRESHOLD_MS {
            return Some(HealthComponent::Encoder);
        }
        None
    }
}

/// Shared state exposing watchdog triggers to the pipeline supervisor.
pub(crate) struct WatchdogState {
    triggered: AtomicBool,
    reason: Mutex<Option<HealthComponent>>,
}

impl WatchdogState {
    /// Create an unarmed watchdog state.
    pub(crate) fn new() -> Self {
        Self {
            triggered: AtomicBool::new(false),
            reason: Mutex::new(None),
        }
    }

    /// Record a trigger reason and mark the watchdog as fired.
    pub(crate) fn arm(&self, component: HealthComponent) {
        if let Ok(mut guard) = self.reason.lock() {
            *guard = Some(component);
        }
        self.triggered.store(true, Ordering::SeqCst);
    }

    /// Returns whether the watchdog fired.
    pub(crate) fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::SeqCst)
    }

    /// Describe the component that caused the trigger, if known.
    pub(crate) fn reason(&self) -> Option<HealthComponent> {
        match self.reason.lock() {
            Ok(guard) => *guard,
            Err(_) => None,
        }
    }
}

/// Spawn the watchdog thread that polls component health and requests restarts.
pub(crate) fn spawn_watchdog(
    health: Arc<PipelineHealth>,
    running: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    state: Arc<WatchdogState>,
) -> std::thread::JoinHandle<()> {
    telemetry::spawn_thread("vision-watchdog", move || {
        while running.load(Ordering::Relaxed) && !shutdown.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(WATCHDOG_POLL_INTERVAL_MS));
            let now = current_millis();
            if let Some(component) = health.stale_component(now) {
                error!(
                    "Watchdog detected stalled {} stage; requesting pipeline restart",
                    component.label()
                );
                state.arm(component);
                running.store(false, Ordering::SeqCst);
                break;
            }
        }
    })
    .expect("failed to spawn watchdog thread")
}

fn current_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}
