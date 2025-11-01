use anyhow::Result;

#[cfg(feature = "with-tch")]
use crate::{mnist, vision};

#[cfg(feature = "with-tch")]
pub fn handle_commands(args: &[String]) -> Result<bool> {
    match args.get(1).map(|s| s.as_str()) {
        Some("mnist-train") => {
            mnist::run_mnist_training(args)?;
            Ok(true)
        }
        Some("mnist-predict") => {
            mnist::run_mnist_prediction(args)?;
            Ok(true)
        }
        Some("vision") => {
            vision::run_from_args(args)?;
            Ok(true)
        }
        Some("mnist-help") => {
            mnist::print_help();
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(not(feature = "with-tch"))]
pub fn handle_commands(_args: &[String]) -> Result<bool> {
    Ok(false)
}
