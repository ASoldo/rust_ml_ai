//! CLI dispatcher for the application binary.
//!
//! The CLI exists primarily to route invocations to feature-gated demos and the
//! full vision pipeline. When compiled without the `with-tch` feature the entry
//! points remain available but dormant.

use anyhow::Result;

#[cfg(feature = "with-tch")]
use crate::{mnist, vision};

#[cfg(feature = "with-tch")]
/// Routes the CLI arguments to the correct subsystem.
///
/// The function returns `Ok(true)` when a subcommand consumed the invocation,
/// allowing the caller to short-circuit fallback behaviour (e.g. running the GPU
/// demo). When no recognised subcommand is present the caller continues with
/// default startup.
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
/// No-op CLI handler used when the binary is compiled without deep-learning
/// support. We keep the signature identical so the call-site does not require
/// feature-specific branching.
pub fn handle_commands(_args: &[String]) -> Result<bool> {
    Ok(false)
}
