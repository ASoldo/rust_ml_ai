//! Lightweight GPU smoke-test exercised when no CLI subcommand is chosen.
//!
//! The routine launches a vector addition kernel via the `gpu-kernels` crate,
//! giving developers confidence that CUDA is reachable before running heavier
//! workloads.

use gpu_kernels::add_vectors;
use ml_core::sample_inputs;

const ELEMENT_COUNT: usize = 16;

/// Launches the vector addition kernel and prints the result.
///
/// The output doubles as a sanity check for the CUDA driver stack. Failures
/// prompt the user to verify their environment before attempting model
/// inference.
pub fn run() {
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
