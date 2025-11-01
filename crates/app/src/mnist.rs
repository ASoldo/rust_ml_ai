use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use ml_core::{TrainingConfig, predict_image_file, tch::Device, train_mnist};

const TRAIN_USAGE: &str = "Usage: cargo run -p vision --features with-tch -- \
mnist-train <data-dir> <model-out> [epochs] [batch-size] [learning-rate] [--cpu]";
const PREDICT_USAGE: &str = "Usage: cargo run -p vision --features with-tch -- mnist-predict <model-path> \
<image-path> [--cpu]";

pub fn run_mnist_training(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        bail!("{TRAIN_USAGE}");
    }

    let data_dir = PathBuf::from(&args[2]);
    let model_out = PathBuf::from(&args[3]);
    let epochs = args.get(4).and_then(|s| s.parse::<i64>().ok()).unwrap_or(5);
    let batch_size = args
        .get(5)
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(128);
    let learning_rate = args
        .get(6)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1e-3);
    let use_cpu = args.iter().any(|arg| arg == "--cpu");

    let mut config = TrainingConfig::new(&data_dir, &model_out);
    config.epochs = epochs;
    config.batch_size = batch_size;
    config.learning_rate = learning_rate;
    if use_cpu {
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

pub fn run_mnist_prediction(args: &[String]) -> Result<()> {
    if args.len() < 4 {
        bail!("{PREDICT_USAGE}");
    }

    let model_path = PathBuf::from(&args[2]);
    let image_path = PathBuf::from(&args[3]);
    let use_cpu = args.iter().any(|arg| arg == "--cpu");
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };

    let prediction = predict_image_file(&model_path, &image_path, Some(device)).map_err(|err| {
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

pub fn print_help() {
    println!("MNIST helper commands:");
    println!(
        "  {TRAIN_USAGE}\n      Train the digit classifier using files in <data-dir> and save weights to <model-out>."
    );
    println!("  {PREDICT_USAGE}\n      Load a trained model and classify a 28x28 grayscale image.");
    println!(
        "  vision <camera-uri> <model-path> <width> <height> [--cpu] [--nvdec] [--verbose] [--detector-width <px>] [--detector-height <px>] [--jpeg-quality <1-100>]\n      Stream frames, run the detector, opt into CPU fallback or NVDEC capture."
    );
    println!("  mnist-help\n      Show this message.");
}
