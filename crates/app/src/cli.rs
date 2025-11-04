//! CLI dispatcher for the application binary.
//!
//! The CLI exists primarily to route invocations to feature-gated demos and the
//! full vision pipeline. When compiled without the `with-tch` feature the entry
//! points remain available but dormant.

use anyhow::Result;

#[cfg(feature = "with-tch")]
use clap::{Parser, Subcommand};

#[cfg(feature = "with-tch")]
use crate::{
    mnist::{self, MnistPredictArgs, MnistTrainArgs},
    vision::{self, VisionCliArgs, VisionConfig},
};

/// Parse CLI arguments and run the requested subcommand.
///
/// Returns `Ok(true)` when a subcommand consumed the invocation so the caller
/// can short-circuit fallback behaviour (e.g. running the GPU demo). When no
/// subcommand is provided the caller continues with default startup.
#[cfg(feature = "with-tch")]
pub fn dispatch() -> Result<bool> {
    let cli = AppCli::parse();
    match cli.command {
        Some(Command::Vision(args)) => {
            let config = VisionConfig::try_from(args)?;
            vision::run(config)?;
            Ok(true)
        }
        Some(Command::MnistTrain(args)) => {
            mnist::run_mnist_training(args)?;
            Ok(true)
        }
        Some(Command::MnistPredict(args)) => {
            mnist::run_mnist_prediction(args)?;
            Ok(true)
        }
        Some(Command::MnistHelp) => {
            mnist::print_help();
            Ok(true)
        }
        None => Ok(false),
    }
}

/// No-op CLI handler used when the binary is compiled without deep-learning
/// support. We keep the signature identical so the call-site does not require
/// feature-specific branching.
#[cfg(not(feature = "with-tch"))]
pub fn dispatch() -> Result<bool> {
    Ok(false)
}

#[cfg(feature = "with-tch")]
#[derive(Debug, Parser)]
#[command(
    name = "vision",
    version,
    author,
    about = "Vision pipeline and ML demos"
)]
struct AppCli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[cfg(feature = "with-tch")]
#[derive(Debug, Subcommand)]
enum Command {
    /// Run the vision inference pipeline.
    Vision(VisionCliArgs),
    /// Train the MNIST digit classifier.
    MnistTrain(MnistTrainArgs),
    /// Run inference on a single MNIST image.
    MnistPredict(MnistPredictArgs),
    /// Print MNIST helper command usage.
    MnistHelp,
}
