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
`crates/app/src/main.rs` to experiment with new data shapes.

## Digit classifier (tch)

Enable the workspace feature to pull in `tch` and image handling utilities:

```bash
cargo run -p cuda-app --features with-tch -- mnist-help
```

> **Heads up:** `tch` needs a LibTorch installation. Either set the `LIBTORCH`
> environment variable to a local LibTorch folder or export
> `LIBTORCH_USE_PYTORCH=1` inside an environment where PyTorch is installed.

The helper subcommands expect the MNIST files that `tch` looks for:

- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

Place them inside a directory such as `./data/mnist`, then train a lightweight
classifier and save the weights:

```bash
cargo run -p cuda-app --features with-tch -- \
  mnist-train data/mnist models/mnist-linear.ot 5 128 0.001
```

You can pass `--cpu` at the end of the command to force CPU training or
prediction.

Once trained, run inference on a 28×28 grayscale PNG/JPEG (centered digits work
best):

```bash
cargo run -p cuda-app --features with-tch -- \
  mnist-predict models/mnist-linear.ot samples/five.png
```

The CLI prints the predicted digit along with per-class probabilities so you can
inspect the confidence distribution.
