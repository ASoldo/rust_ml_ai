//! Dynamic loading helpers for Torch CUDA shared objects.

use std::sync::Once;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
use tracing::{info, warn};

/// Lazily load CUDA-dependent Torch shared libraries so `libtorch` kernels are
/// available to the detector runtime.
///
/// LibTorch defers resolution of certain symbols until the first CUDA
/// interaction. Loading them upfront avoids confusing runtime panics on systems
/// where LD paths omit the libtorch directories.
pub(crate) fn load_torch_cuda_runtime(verbose: bool) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let mut handles = Vec::new();
        for lib in [
            "libtorch_cuda.so",
            "libtorch_cuda_cu.so",
            "libtorch_cuda_cpp.so",
        ] {
            match unsafe { Library::open(Some(lib), RTLD_NOW | RTLD_GLOBAL) } {
                Ok(handle) => {
                    if verbose {
                        info!("Loaded {lib}");
                    }
                    handles.push(handle);
                }
                Err(err) => {
                    if verbose {
                        warn!("Warning: failed to load {lib}: {err}");
                    }
                }
            }
        }
        Box::leak(Box::new(handles));
    });
}
