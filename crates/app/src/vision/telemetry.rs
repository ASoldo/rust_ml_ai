//! Telemetry helpers for tracing spans, Prometheus metrics, and optional console tooling.

use std::{io, panic, path::Path, sync::OnceLock, thread, time::Duration};

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use tracing_subscriber::{
    filter::{EnvFilter, filter_fn},
    fmt,
    layer::SubscriberExt,
    prelude::*,
};

use crate::vision::config::TelemetryOptions;

static PROM_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();
static PROM_UPKEEP_THREAD: OnceLock<thread::JoinHandle<()>> = OnceLock::new();

/// Guard returned when a telemetry subscriber has been installed for the current thread.
pub(crate) struct TelemetryGuard {
    _default_guard: tracing::subscriber::DefaultGuard,
    _chrome_guard: Option<tracing_chrome::FlushGuard>,
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
        _default_guard: default_guard,
        _chrome_guard: chrome_guard,
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
        .trace_style(tracing_chrome::TraceStyle::Threaded)
        .build();
    Ok((layer, guard))
}
