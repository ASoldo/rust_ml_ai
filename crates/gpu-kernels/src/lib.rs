use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use std::{error::Error, fmt, result::Result as StdResult};

/// Convenient result alias for GPU operations.
pub type Result<T> = StdResult<T, Box<dyn Error + Send + Sync>>;

/// Error raised when vector lengths differ.
#[derive(Debug)]
struct LengthMismatch {
    left: usize,
    right: usize,
}

impl fmt::Display for LengthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "input slices must have the same length (left={}, right={})",
            self.left, self.right
        )
    }
}

impl Error for LengthMismatch {}

/// Launches a CUDA kernel that adds the elements of `a` and `b` and returns the
/// result as a newly allocated `Vec<f32>` on the host.
///
/// The CUDA kernel is compiled at runtime with NVRTC, so there is no separate
/// `.cu` file to manage while prototyping.
pub fn add_vectors(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(Box::new(LengthMismatch {
            left: a.len(),
            right: b.len(),
        }));
    }

    let source = r#"
        extern "C" __global__ void add(const float* a, const float* b, float* out, size_t len) {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < len) {
                out[idx] = a[idx] + b[idx];
            }
        }
    "#;

    let ptx = compile_ptx(source)?;
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_module(ptx)?;
    let function = module.load_function("add")?;
    let stream = ctx.default_stream();

    let a_dev = stream.memcpy_stod(a)?;
    let b_dev = stream.memcpy_stod(b)?;
    let mut out_dev = stream.alloc_zeros::<f32>(a.len())?;

    let len = a.len();
    let launch = LaunchConfig::for_num_elems(len as u32);

    unsafe {
        stream
            .launch_builder(&function)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut out_dev)
            .arg(&len)
            .launch(launch)?;
    }

    stream.synchronize()?;
    let result = stream.memcpy_dtov(&out_dev)?;

    Ok(result)
}
