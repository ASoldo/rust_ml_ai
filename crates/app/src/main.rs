use gpu_kernels::add_vectors;
use ml_core::sample_inputs;

#[cfg(feature = "with-tch")]
use std::path::PathBuf;

#[cfg(feature = "with-tch")]
use ml_core::{TrainingConfig, predict_image_file, tch::Device, train_mnist};

const ELEMENT_COUNT: usize = 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if handle_digits_commands(&args) {
        return;
    }

    run_gpu_add_demo();
}

fn run_gpu_add_demo() {
    let (a, b) = sample_inputs(ELEMENT_COUNT);

    println!("Input A: {:?}", a);
    println!("Input B: {:?}", b);

    match add_vectors(&a, &b) {
        Ok(sum) => println!("Sum on GPU: {:?}", sum),
        Err(err) => {
            eprintln!("Failed to launch GPU kernel: {err}");
            eprintln!(
                "Hint: ensure an NVIDIA driver is installed and accessible (try `nvidia-smi`)."
            );
        }
    }
}

#[cfg(feature = "with-tch")]
fn handle_digits_commands(args: &[String]) -> bool {
    match args.get(1).map(|s| s.as_str()) {
        Some("mnist-train") => {
            run_mnist_training(args);
            true
        }
        Some("mnist-predict") => {
            run_mnist_prediction(args);
            true
        }
        Some("mnist-help") => {
            print_mnist_help();
            true
        }
        _ => false,
    }
}

#[cfg(not(feature = "with-tch"))]
fn handle_digits_commands(_args: &[String]) -> bool {
    false
}

#[cfg(feature = "with-tch")]
fn run_mnist_training(args: &[String]) {
    if args.len() < 4 {
        eprintln!(
            "Usage: cargo run -p cuda-app --features with-tch -- mnist-train <data-dir> <model-out> [epochs] [batch-size] [learning-rate] [--cpu]"
        );
        return;
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

    match train_mnist(&config) {
        Ok(report) => {
            println!(
                "Training finished — epochs: {}, final loss: {:.4}, test accuracy: {:.2}%",
                report.epochs,
                report.final_loss,
                report.test_accuracy * 100.0
            );
        }
        Err(err) => {
            eprintln!("Failed to train MNIST classifier: {err}");
            eprintln!(
                "Hint: download the MNIST dataset (t10k/train .idx files) into the provided data directory."
            );
        }
    }
}

#[cfg(feature = "with-tch")]
fn run_mnist_prediction(args: &[String]) {
    if args.len() < 4 {
        eprintln!(
            "Usage: cargo run -p cuda-app --features with-tch -- mnist-predict <model-path> <image-path> [--cpu]"
        );
        return;
    }

    let model_path = PathBuf::from(&args[2]);
    let image_path = PathBuf::from(&args[3]);
    let use_cpu = args.iter().any(|arg| arg == "--cpu");
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available()
    };

    match predict_image_file(&model_path, &image_path, Some(device)) {
        Ok(prediction) => {
            println!("Predicted digit: {}", prediction.digit);
            println!("Class probabilities:");
            for (digit, prob) in prediction.probabilities.iter().enumerate() {
                println!("  {digit}: {prob:.3}");
            }
        }
        Err(err) => {
            eprintln!("Failed to run prediction: {err}");
            eprintln!(
                "Hint: make sure the model path points to a `.ot` file produced by mnist-train \n        and that the image is a 28x28 grayscale PNG or JPEG."
            );
        }
    }
}

#[cfg(feature = "with-tch")]
fn print_mnist_help() {
    println!("MNIST helper commands:");
    println!(
        "  mnist-train <data-dir> <model-out> [epochs] [batch-size] [learning-rate] [--cpu]\n      Train the digit classifier using files in <data-dir> and save weights to <model-out>."
    );
    println!(
        "  mnist-predict <model-path> <image-path> [--cpu]\n      Load a trained model and classify a 28x28 grayscale image."
    );
    println!("  mnist-help\n      Show this message.");
}
