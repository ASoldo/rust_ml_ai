mod cli;
mod gpu;

#[cfg(feature = "with-tch")]
mod atak;
#[cfg(feature = "with-tch")]
mod hud_html;
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

fn run() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let args: Vec<String> = std::env::args().collect();
    if cli::handle_commands(&args)? {
        return Ok(());
    }

    gpu::run();
    Ok(())
}
