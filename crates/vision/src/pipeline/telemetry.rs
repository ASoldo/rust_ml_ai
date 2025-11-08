//! Telemetry helpers for tracing spans, Prometheus metrics, and optional console tooling.

use std::{
    collections::HashMap,
    io::{self, Write},
    panic,
    path::{Path, PathBuf},
    sync::OnceLock,
    thread,
    time::Duration,
};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::{
    filter::{EnvFilter, filter_fn},
    fmt,
    layer::SubscriberExt,
    prelude::*,
};

use crate::pipeline::config::TelemetryOptions;

static PROM_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
static PROM_UPKEEP_THREAD: OnceLock<thread::JoinHandle<()>> = OnceLock::new();

/// Guard returned when a telemetry subscriber has been installed for the current thread.
pub(crate) struct TelemetryGuard {
    default_guard: Option<tracing::subscriber::DefaultGuard>,
    _chrome_guard: Option<tracing_chrome::FlushGuard>,
    chrome_trace_path: Option<PathBuf>,
}

/// Ensure the global metrics recorder is installed and return the Prometheus handle.
pub(crate) fn init_metrics_recorder() -> &'static PrometheusHandle {
    PROM_HANDLE.get_or_init(|| {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();

        metrics::set_global_recorder(recorder).expect("metrics recorder already installed");

        let upkeep_handle = handle.clone();
        PROM_UPKEEP_THREAD.get_or_init(|| {
            spawn_thread("prometheus-upkeep", move || {
                loop {
                    thread::sleep(Duration::from_secs(5));
                    upkeep_handle.run_upkeep();
                }
            })
            .expect("failed to spawn prometheus upkeep thread")
        });

        handle
    })
}

/// Access the Prometheus handle when already initialised.
pub(crate) fn prometheus_handle() -> Option<&'static PrometheusHandle> {
    PROM_HANDLE.get()
}

/// Install tracing subscribers required for the runtime based on telemetry options.
pub(crate) fn enter_runtime(opts: &TelemetryOptions) -> TelemetryGuard {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let console_layer = if opts.enable_tokio_console {
        match panic::catch_unwind(|| {
            console_subscriber::ConsoleLayer::builder()
                .with_default_env()
                .spawn()
        }) {
            Ok(layer) => Some(layer),
            Err(_) => {
                tracing::warn!(
                    "tokio-console requested but current build lacks `tokio_unstable`; skipping console layer"
                );
                None
            }
        }
    } else {
        None
    };

    let (chrome_layer_opt, chrome_guard) = if let Some(path) = opts.chrome_trace_path.as_ref() {
        match build_chrome_layer(path) {
            Ok((layer, guard)) => (Some(layer), Some(guard)),
            Err(err) => {
                tracing::warn!(
                    "failed to initialise chrome trace writer at {}: {err}",
                    path.display()
                );
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    if chrome_layer_opt.is_some() && console_layer.is_some() {
        tracing::warn!(
            "Chrome trace and tokio-console enabled together; defaulting to chrome trace only"
        );
    }

    let console_layer = if chrome_layer_opt.is_some() {
        None
    } else {
        console_layer
    };

    let fmt_filter_console = env_filter.clone();
    let fmt_filter_chrome = env_filter.clone();
    let fmt_filter_default = env_filter;

    let span_only_filter = filter_fn(|metadata| metadata.is_span());

    let default_guard = match (console_layer, chrome_layer_opt) {
        (Some(console), None) => tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(console)
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_timer(fmt::time::uptime())
                        .with_filter(fmt_filter_console.clone()),
                )
                .with(tracing_error::ErrorLayer::default()),
        ),
        (Some(_console), Some(chrome)) => tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(chrome.with_filter(span_only_filter.clone()))
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_timer(fmt::time::uptime())
                        .with_filter(fmt_filter_chrome.clone()),
                )
                .with(tracing_error::ErrorLayer::default()),
        ),
        (None, Some(chrome)) => tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(chrome.with_filter(span_only_filter.clone()))
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_timer(fmt::time::uptime())
                        .with_filter(fmt_filter_chrome.clone()),
                )
                .with(tracing_error::ErrorLayer::default()),
        ),
        (None, None) => tracing::subscriber::set_default(
            tracing_subscriber::registry()
                .with(
                    fmt::layer()
                        .with_target(false)
                        .with_timer(fmt::time::uptime())
                        .with_filter(fmt_filter_default),
                )
                .with(tracing_error::ErrorLayer::default()),
        ),
    };

    TelemetryGuard {
        default_guard: Some(default_guard),
        _chrome_guard: chrome_guard,
        chrome_trace_path: opts.chrome_trace_path.clone(),
    }
}

/// Spawn a thread that inherits the current tracing dispatcher.
pub(crate) fn spawn_thread<F, T>(name: impl Into<String>, f: F) -> io::Result<thread::JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let dispatch = tracing::dispatcher::get_default(|current| current.clone());
    thread::Builder::new()
        .name(name.into())
        .spawn(move || tracing::dispatcher::with_default(&dispatch, f))
}

fn build_chrome_layer(
    path: &Path,
) -> Result<
    (
        tracing_chrome::ChromeLayer<tracing_subscriber::Registry>,
        tracing_chrome::FlushGuard,
    ),
    std::io::Error,
> {
    let file = std::fs::File::create(path)?;
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(file)
        .include_args(true)
        .build();
    Ok((layer, guard))
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        let chrome_path = self.chrome_trace_path.clone();
        if let Some(guard) = self._chrome_guard.take() {
            guard.flush();
            drop(guard);
        }

        if let Some(path) = chrome_path.as_ref() {
            if let Err(err) = fix_async_ids(path) {
                tracing::warn!("failed to normalise chrome trace {}: {err}", path.display());
            }
        }

        // Leaking the guard keeps the subscriber installed for the process lifetime,
        // which avoids `tried to drop a ref to Id(_)` panics when background threads
        // still hold spans during shutdown.
        if let Some(guard) = self.default_guard.take() {
            std::mem::forget(guard);
        }
    }
}

/// Post-process Chrome traces emitted in async mode so Perfetto renders them correctly.
///
/// The tracing `ChromeLayer` uses async slices when `TraceStyle::Async` is selected. That results
/// in two follow-up tasks before handing the trace to Perfetto:
///   1. Reassign async IDs so that nested spans in the same logical scope can be disambiguated.
///   2. Coerce same-thread `b/e` pairs back into synchronous `B/E` durations while leaving genuine
///      cross-thread spans encoded as async slices. This prevents Perfetto's `misplaced_end_event`
///      warnings while keeping true async flows intact.
fn fix_async_ids(path: &Path) -> io::Result<()> {
    if !path.is_file() {
        return Ok(());
    }

    let data = std::fs::read(path)?;
    let mut events: Vec<serde_json::Value> = match serde_json::from_slice(&data) {
        Ok(v) => v,
        Err(err) => {
            return Err(io::Error::new(io::ErrorKind::InvalidData, err));
        }
    };

    let mut next_id: u64 = 1;
    #[derive(Clone, Copy)]
    struct BeginInfo {
        new_id: u64,
        idx: usize,
        tid: u64,
    }
    let mut id_map: HashMap<u64, Vec<BeginInfo>> = HashMap::new();
    let mut orphan_end_events: Vec<usize> = Vec::new();

    fn to_sync(entry: &mut serde_json::Value, phase: &str) {
        if let Some(obj) = entry.as_object_mut() {
            obj.insert("ph".into(), phase.into());
            obj.remove("id");
        }
    }

    fn ensure_async(entry: &mut serde_json::Value, phase: &str) {
        if let Some(obj) = entry.as_object_mut() {
            obj.insert("ph".into(), phase.into());
        }
    }

    for idx in 0..events.len() {
        let Some(phase) = events[idx].get("ph").and_then(|v| v.as_str()) else {
            continue;
        };

        match phase {
            "b" => {
                let entry = &mut events[idx];
                let Some(old_id) = entry.get("id").and_then(|v| v.as_u64()) else {
                    continue;
                };
                let tid = entry.get("tid").and_then(|v| v.as_u64()).unwrap_or(0);
                let new_id = next_id;
                next_id = next_id.saturating_add(1);
                entry["id"] = new_id.into();
                id_map.entry(old_id).or_default().push(BeginInfo {
                    new_id,
                    idx,
                    tid,
                });
            }
            "e" => {
                let Some(old_id) = events[idx].get("id").and_then(|v| v.as_u64()) else {
                    continue;
                };
                let end_tid = events[idx].get("tid").and_then(|v| v.as_u64()).unwrap_or(0);
                let Some(stack) = id_map.get_mut(&old_id) else {
                    orphan_end_events.push(idx);
                    continue;
                };
                let Some(info) = stack.pop() else {
                    orphan_end_events.push(idx);
                    continue;
                };
                let (begin_slice, end_slice) = events.split_at_mut(idx);
                let begin_entry = &mut begin_slice[info.idx];
                let end_entry = &mut end_slice[0];
                end_entry["id"] = info.new_id.into();
                let start_tid = info.tid;
                if start_tid == end_tid {
                    to_sync(begin_entry, "B");
                    to_sync(end_entry, "E");
                } else {
                    ensure_async(begin_entry, "b");
                    ensure_async(end_entry, "e");
                }
            }
            _ => {}
        }
    }

    if !orphan_end_events.is_empty() {
        orphan_end_events.sort_unstable();
        for idx in orphan_end_events.into_iter().rev() {
            events.remove(idx);
        }
    }

    let mut meta_events: Vec<serde_json::Value> = Vec::new();
    let mut span_events: Vec<serde_json::Value> = Vec::new();
    for ev in events.into_iter() {
        match ev.get("ph").and_then(|v| v.as_str()) {
            Some("M") => meta_events.push(ev),
            _ => span_events.push(ev),
        }
    }
    span_events.sort_by(|a, b| {
        let ts_a = a.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ts_b = b.get("ts").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if ts_a == ts_b {
            let order = |ph: &str| match ph {
                "B" => 0,
                "E" => 1,
                _ => 2,
            };
            let pa = a.get("ph").and_then(|v| v.as_str()).map(order).unwrap_or(2);
            let pb = b.get("ph").and_then(|v| v.as_str()).map(order).unwrap_or(2);
            pa.cmp(&pb)
        } else {
            ts_a.partial_cmp(&ts_b).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    meta_events.append(&mut span_events);

    let tmp_path = path.with_extension("json.tmp");
    let mut file = std::fs::File::create(&tmp_path)?;
    serde_json::to_writer(&mut file, &meta_events)?;
    file.write_all(b"\n")?;
    std::fs::rename(tmp_path, path)?;
    Ok(())
}
