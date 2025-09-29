# CUDA C++ Dojo

This repository now exposes a small Rust workspace with three crates:

- `ml-core`: host-side helpers intended for future `tch`/PyTorch work
- `gpu-kernels`: CUDA utilities built on top of `cudarc`
- `cuda-app`: a thin binary crate that wires the other two pieces together

Out of the box the workspace demonstrates:

- compiling a CUDA kernel at runtime with NVRTC (`gpu-kernels`)
- launching the kernel via `cudarc` and copying results back to the host
- printing the host/GPU vectors from the `cuda-app` entry-point

## Prerequisites

- NVIDIA GPU with a compatible driver installed
- CUDA toolkit (for NVRTC to find its headers and libraries)
- Rust toolchain (`cargo` and `rustc`)

## Run the sample

```bash
cargo run -p cuda-app
```

If the CUDA driver is not available on the current machine you will see an error like `CUDA_ERROR_OPERATING_SYSTEM`. In that case, verify that the driver is loaded and the GPU is accessible (e.g. `nvidia-smi`).

When the kernel launches successfully the program prints the inputs and the element-wise
sum computed on the GPU:

```
Input A: [0.0, 1.0, 2.0, ...]
Input B: [0.0, 10.0, 20.0, ...]
Sum on GPU: [0.0, 11.0, 22.0, ...]
```

## Next steps

Tinker with the kernel in `crates/gpu-kernels/src/lib.rs` or adjust the wiring in
`crates/app/src/main.rs` to experiment with new data shapes. When you're ready to pull in
`tch`, enable the `with-tch` feature on `ml-core` and extend it with tensor utilities.
