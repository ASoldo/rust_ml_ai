//! Binary entrypoint for the `vision` application. The binary wires CLI parsing,
//! GPU initialisation, and the high-level orchestrations exposed by the `vision`
//! module.

mod cli;
mod gpu;

#[cfg(feature = "with-tch")]
mod html;
#[cfg(feature = "with-tch")]
mod mnist;
#[cfg(feature = "with-tch")]
mod vision;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err:?}");
        std::process::exit(1);
    }
}

/// Bootstraps the application by wiring tracing, delegating CLI handling, and
/// falling back to the GPU demo if no subcommand consumes the invocation.
///
/// The CLI can trigger multiple flows (e.g. vision pipeline, MNIST demo). When no
/// specialised flow is requested we default to the GPU demo to keep developer UX
/// snappy during bring-up.
fn run() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    #[cfg(feature = "with-tch")]
    if cli::dispatch()? {
        return Ok(());
    }

    gpu::run();
    Ok(())
}
