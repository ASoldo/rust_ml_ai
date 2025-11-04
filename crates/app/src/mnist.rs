//! Command helpers for training and running an MNIST classifier.
//!
//! The MNIST flows serve as a lightweight example of the `ml-core` crate in
//! action. They are intentionally concise so the vision pipeline can lean on the
//! same infrastructure for model loading and device selection.

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::Args;
use ml_core::{TrainingConfig, predict_image_file, tch::Device, train_mnist};

/// Arguments accepted by the `mnist-train` subcommand.
#[derive(Debug, Args)]
pub struct MnistTrainArgs {
    /// Directory containing the MNIST dataset files.
    #[arg(value_name = "DATA_DIR")]
    pub data_dir: PathBuf,
    /// Path where the trained weights will be stored.
    #[arg(value_name = "MODEL_OUT")]
    pub model_out: PathBuf,
    /// Number of epochs to train.
    #[arg(value_name = "EPOCHS", default_value_t = 5)]
    pub epochs: i64,
    /// Mini-batch size used during training.
    #[arg(value_name = "BATCH_SIZE", default_value_t = 128)]
    pub batch_size: i64,
    /// Optimiser learning rate.
    #[arg(value_name = "LEARNING_RATE", default_value_t = 1e-3)]
    pub learning_rate: f64,
    /// Force CPU training even when CUDA is available.
    #[arg(long = "cpu", action = clap::ArgAction::SetTrue)]
    pub cpu: bool,
}

/// Arguments accepted by the `mnist-predict` subcommand.
#[derive(Debug, Args)]
pub struct MnistPredictArgs {
    /// Path to the trained `.ot` weights file.
    #[arg(value_name = "MODEL_PATH")]
    pub model_path: PathBuf,
    /// Path to the 28x28 grayscale image to classify.
    #[arg(value_name = "IMAGE_PATH")]
    pub image_path: PathBuf,
    /// Run inference on the CPU even if CUDA is available.
    #[arg(long = "cpu", action = clap::ArgAction::SetTrue)]
    pub cpu: bool,
}

/// Train the MNIST classifier with optional overrides for training hyperparameters.
///
/// Returns an error if mandatory arguments are missing or if training fails.
pub fn run_mnist_training(args: MnistTrainArgs) -> Result<()> {
    let mut config = TrainingConfig::new(&args.data_dir, &args.model_out);
    config.epochs = args.epochs;
    config.batch_size = args.batch_size;
    config.learning_rate = args.learning_rate;
    if args.cpu {
        config.device = Device::Cpu;
    }

    let report = train_mnist(&config).map_err(|err| {
        anyhow!(
            "Failed to train MNIST classifier: {err}\nHint: download the MNIST dataset (t10k/train \
.idx files) into the provided data directory."
        )
    })?;

    println!(
        "Training finished — epochs: {}, final loss: {:.4}, test accuracy: {:.2}%",
        report.epochs,
        report.final_loss,
        report.test_accuracy * 100.0
    );
    Ok(())
}

/// Run inference for a single image using a previously trained MNIST model.
///
/// The routine mirrors the command-line UX in official PyTorch tutorials, making
/// it easy to compare behaviour across toolchains.
pub fn run_mnist_prediction(args: MnistPredictArgs) -> Result<()> {
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };

    let prediction =
        predict_image_file(&args.model_path, &args.image_path, Some(device)).map_err(|err| {
            anyhow!(
                "Failed to run prediction: {err}\nHint: ensure the model path points to a `.ot` file \
produced by mnist-train and that the image is a 28x28 grayscale PNG or JPEG."
            )
        })?;

    println!("Predicted digit: {}", prediction.digit);
    println!("Class probabilities:");
    for (digit, prob) in prediction.probabilities.iter().enumerate() {
        println!("  {digit}: {prob:.3}");
    }
    Ok(())
}

/// Print command help for the MNIST utilities.
pub fn print_help() {
    println!("MNIST helper commands:");
    println!(
        "  vision mnist-train <DATA_DIR> <MODEL_OUT> [EPOCHS] [BATCH_SIZE] [LEARNING_RATE] [--cpu]"
    );
    println!("  vision mnist-predict <MODEL_PATH> <IMAGE_PATH> [--cpu]");
    println!("  vision vision [OPTIONS] <CAMERA_URI> <MODEL_PATH> <WIDTH> <HEIGHT> ...");
    println!("  vision mnist-help");
}
