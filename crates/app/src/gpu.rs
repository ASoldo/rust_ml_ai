use gpu_kernels::add_vectors;
use ml_core::sample_inputs;

const ELEMENT_COUNT: usize = 16;

/// Run the simple GPU vector addition check.
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
