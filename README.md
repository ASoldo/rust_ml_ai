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

## GPU demo

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

Create a directory such as `data/mnist` and place the four files directly inside
it—no extra subfolders. Create a `models` directory for the exported weights,
then train a lightweight classifier:

```bash
mkdir -p data/mnist models
# copy the MNIST idx files into data/mnist/
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

Example output using a hand-drawn `0`:

```
Predicted digit: 0
Class probabilities:
  0: 1.000
  1: 0.000
  2: 0.000
  3: 0.000
  4: 0.000
  5: 0.000
  6: 0.000
  7: 0.000
  8: 0.000
  9: 0.000
```

## Vision demo (TorchScript + video)

> This command path mixes TorchScript inference with our Rust/CUDA pipeline so
> we can experiment with real-time detection.

1. Create a Python environment that bundles PyTorch and Ultralytics (so the
   exporter gets a compatible libtorch):

   ```bash
   python3.11 -m venv ~/venvs/ultra
   source ~/venvs/ultra/bin/activate
   pip install --upgrade pip
   pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
     torch torchvision
   pip install ultralytics==8.3.0
   ```

2. Download a face-detection checkpoint and export it to TorchScript. The
   YOLOv12 nano face model is a convenient starting point:

   ```bash
   mkdir -p models
   wget https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov12n-face.pt \
     -O models/yolov12n-face.pt

   yolo export \
     model=models/yolov12n-face.pt \
     format=torchscript \
     imgsz=640 \
     device=cuda   # or `cpu` if a GPU isn't available during export
   ```

   The exporter writes `models/yolov12n-face.torchscript`. Note the image size
   (640×640 above); the Rust pipeline must feed frames at the same resolution.

3. Run the new CLI subcommand. Supply a camera URI (`0` opens the default
   webcam), the TorchScript path, and the width/height you exported with:

   ```bash
   cargo run -p cuda-app --features with-tch -- \
      vision-demo 0 models/yolov12n-face.torchscript 640 640 --cpu
   ```

   Drop `--cpu` to use CUDA inference once you're ready. The current pipeline
   logs raw YOLO-style detections (`[x, y, w, h]` centers in the 640×640 input
   space plus confidence):

   ```text
   frame #200: 11 detection(s)
     #0: class=0 conf=0.83 xywh=[335.0, 323.1, 175.3, 306.4]
     #1: class=0 conf=0.82 xywh=[334.6, 324.2, 173.9, 304.9]
     ...
   ```

   Multiple boxes per face are expected right now—non‑max suppression and
   additional post-processing will move into CUDA kernels in the upcoming steps.

4. While the command is running, open the quick preview endpoints:

   - `http://127.0.0.1:8080/frame.jpg` – last annotated frame as a JPEG snapshot.
   - `http://127.0.0.1:8080/stream.mjpg` – MJPEG stream showing detections in
     real time (the HTML index page embeds this stream by default).
   - `http://127.0.0.1:8080/detections` – latest detections as JSON, including
     track identifiers (simple sequential tracker for now).

   Each annotated frame includes the running frame counter and a smoothed FPS
   estimate in the lower-right corner so you can monitor throughput at a glance.
   Pass `--verbose` if you want the per-frame logs printed in the console.
