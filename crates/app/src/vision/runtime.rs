use std::sync::Once;

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
use tracing::{info, warn};

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
