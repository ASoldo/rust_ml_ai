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
pub(crate) const WATCHDOG_STARTUP_GRACE_MS: u64 = 30_000;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    started_at: u64,
    stale_threshold_ms: u64,
    capture: AtomicU64,
    processor: AtomicU64,
    encoder: AtomicU64,
}

impl PipelineHealth {
    /// Initialise the health tracker with grace periods for each component.
    pub(crate) fn new() -> Self {
        Self::new_at(current_millis())
    }

    /// A network source can pause briefly while TCP retransmits a frame. Use a
    /// bounded network deadline for all dependent stages while input is idle.
    pub(crate) fn with_stale_threshold_ms(stale_threshold_ms: u64) -> Self {
        let mut health = Self::new();
        health.stale_threshold_ms = stale_threshold_ms;
        health
    }

    fn new_at(now: u64) -> Self {
        Self {
            started_at: now,
            stale_threshold_ms: WATCHDOG_STALE_THRESHOLD_MS,
            capture: AtomicU64::new(0),
            processor: AtomicU64::new(0),
            encoder: AtomicU64::new(0),
        }
    }

    /// Register a heartbeat for the supplied component.
    pub(crate) fn beat(&self, component: HealthComponent) {
        self.beat_at(component, current_millis());
    }

    fn beat_at(&self, component: HealthComponent, now: u64) {
        match component {
            HealthComponent::Capture => self.capture.store(now, Ordering::Relaxed),
            HealthComponent::Processor => self.processor.store(now, Ordering::Relaxed),
            HealthComponent::Encoder => self.encoder.store(now, Ordering::Relaxed),
        }
    }

    /// Returns the first component that has not produced a heartbeat recently.
    pub(crate) fn stale_component(&self, now: u64) -> Option<HealthComponent> {
        [
            (HealthComponent::Capture, &self.capture),
            (HealthComponent::Processor, &self.processor),
            (HealthComponent::Encoder, &self.encoder),
        ]
        .into_iter()
        .find_map(|(component, heartbeat)| {
            let last = heartbeat.load(Ordering::Relaxed);
            // Probing a live stream can take several seconds before its first
            // frame. Once a stage starts, keep the normal short stall deadline.
            let stale = if last == 0 {
                now.saturating_sub(self.started_at) > WATCHDOG_STARTUP_GRACE_MS
            } else {
                now.saturating_sub(last) > self.stale_threshold_ms
            };
            stale.then_some(component)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_deadline_tolerates_retransmission_but_remains_bounded() {
        let mut health = PipelineHealth::new_at(1_000);
        health.stale_threshold_ms = 5_000;
        for stage in [
            HealthComponent::Capture,
            HealthComponent::Processor,
            HealthComponent::Encoder,
        ] {
            health.beat_at(stage, 10_000);
        }
        assert_eq!(health.stale_component(13_000), None);
        assert_eq!(health.stale_component(15_000), None);
        assert_eq!(
            health.stale_component(15_001),
            Some(HealthComponent::Capture)
        );
        health.beat_at(HealthComponent::Capture, 15_001);
        assert_eq!(
            health.stale_component(15_001),
            Some(HealthComponent::Processor)
        );
    }

    #[test]
    fn first_frame_can_arrive_after_a_long_rtsp_probe() {
        let health = PipelineHealth::new_at(1_000);
        assert_eq!(health.stale_component(16_000), None);
        for stage in [
            HealthComponent::Capture,
            HealthComponent::Processor,
            HealthComponent::Encoder,
        ] {
            health.beat_at(stage, 16_000);
        }
        assert_eq!(health.stale_component(17_500), None);
        assert_eq!(
            health.stale_component(17_501),
            Some(HealthComponent::Capture)
        );
    }

    #[test]
    fn startup_still_has_a_bounded_deadline() {
        let health = PipelineHealth::new_at(1_000);
        assert_eq!(health.stale_component(31_000), None);
        assert_eq!(
            health.stale_component(31_001),
            Some(HealthComponent::Capture)
        );
    }

    #[test]
    fn downstream_startup_does_not_hide_a_running_stage_stall() {
        let health = PipelineHealth::new_at(1_000);
        health.beat_at(HealthComponent::Capture, 10_000);
        assert_eq!(health.stale_component(11_500), None);
        assert_eq!(
            health.stale_component(11_501),
            Some(HealthComponent::Capture)
        );
        health.beat_at(HealthComponent::Capture, 31_001);
        assert_eq!(
            health.stale_component(31_001),
            Some(HealthComponent::Processor)
        );
        health.beat_at(HealthComponent::Processor, 31_001);
        assert_eq!(
            health.stale_component(31_001),
            Some(HealthComponent::Encoder)
        );
    }
}
