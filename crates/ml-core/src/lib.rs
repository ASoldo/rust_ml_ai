/// High-level helpers for preparing host data before sending it to the GPU or
/// into a future `tch`/PyTorch pipeline.
pub fn sample_inputs(len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..len).map(|i| i as f32).collect();
    let b = (0..len).map(|i| (i as f32) * 10.0).collect();
    (a, b)
}

#[cfg(feature = "with-tch")]
/// Builds a CPU tensor from the first sample vector. Enable the `with-tch`
/// feature in `ml-core` to pull in the `tch` crate.
pub fn sample_tensor(len: usize) -> tch::Result<tch::Tensor> {
    let (a, _) = sample_inputs(len);
    Ok(tch::Tensor::from_slice(&a))
}

#[cfg(feature = "with-tch")]
pub use tch;

#[cfg(feature = "with-tch")]
mod digits;

#[cfg(feature = "with-tch")]
pub use digits::{
    DIGIT_CLASS_COUNT, DigitClassifier, IMAGE_EDGE_PIXELS, Prediction, TrainingConfig,
    TrainingReport,
};

#[cfg(feature = "with-tch")]
pub use digits::{predict_image_file, train_mnist};
