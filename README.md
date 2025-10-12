# CUDA C++ Dojo

This repository now exposes a small Rust workspace with three crates and a web HUD that can be served straight from the `cuda-app` binary:

- `ml-core`: host-side helpers intended for future `tch`/PyTorch work
- `gpu-kernels`: CUDA utilities built on top of `cudarc`
- `cuda-app`: a thin binary crate that wires the other two pieces together and serves the web interfaces

Out of the box the workspace demonstrates:

- compiling a CUDA kernel at runtime with NVRTC (`gpu-kernels`)
- launching the kernel via `cudarc` and copying results back to the host
- printing the host/GPU vectors from the `cuda-app` entry-point

## Container quickstart

Before building, make sure:

- Docker 24+ is installed and the NVIDIA Container Toolkit is configured (`sudo nvidia-ctk runtime configure --runtime=docker` then restart Docker).
- `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi` succeeds so the runtime can see your GPU.
- The camera you want to use is exposed (default `/dev/video0`; adjust compose if you have a different device path).
- TorchScript weights live under `models/` (e.g. `models/yolov12n-face.torchscript`); datasets and checkpoints are never committed, so populate them locally.

The Docker build bakes in CUDA 13.0.1, LibTorch 2.8.0+cu129, OpenCV, FFmpeg codec libs and V4L2 tools. Override `CUDA_BASE` or `LIBTORCH_URL` if you need a different toolkit/libtorch pairing.

### Build

```bash
docker compose build
# or
docker build -t cuda-cpp-dojo .
```

Need a different LibTorch build? Pass `--build-arg LIBTORCH_URL=...` to either command. The Dockerfile defaults to 2.8.0+cu129 (CUDA 13).
Set `CUDA_BASE` if you want a different CUDA image tag (e.g. `--build-arg CUDA_BASE=12.9.0-devel-ubuntu22.04`).

### One-off runs

```bash
docker run --rm -it --gpus all \
  --device /dev/video0:/dev/video0 \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  cuda-cpp-dojo:latest
```

This launches the default vector-add demo. The image ships with the `with-tch` feature already enabled so TorchScript paths work out of the box.

### Vision demo (web preview)

The web HUD now ships with two routes:

<img width="1893" height="940" alt="image" src="https://github.com/user-attachments/assets/e14e44f4-c6c4-4091-91bf-ef686c2eb4d6" />

- `http://localhost:8080/` &mdash; the **Recon HUD** that visualises one or more virtual camera rigs on top of an OpenStreetMap plane. Each rig reuses the shared MJPEG stream, snaps to a precise lat/lon anchor, and gets a clickable sphere so you can select which rig drives the azimuth dial and detection overlays.

<img width="1899" height="939" alt="image" src="https://github.com/user-attachments/assets/4a328f04-a21f-44b1-b151-eac09a5b21ca" />

- `http://localhost:8080/atak` &mdash; an **ATAK-style command view** intended to be self-hosted. It focuses on the map layer and gives you an overview of every registered camera system without the HUD widgets. Use this route when you want to coordinate multiple deployments from a single map-centric dashboard.

1. Build (or rebuild after code changes):
   ```bash
   docker compose build
   ```

2. Launch the detector with ports published:
   ```bash
   docker compose up vision-demo
   ```

3. For ad-hoc runs with custom flags use:
   ```bash
   docker compose run --rm --service-ports cuda-app \
     vision-demo /dev/video0 models/yolov12n-face.torchscript 640 640
   ```

Drop `--nvdec` for raw V4L2 capture or append `--cpu` to force CPU inference. Edit the `vision-demo` service in `docker-compose.yml` if you need different defaults (camera URI, model, resolution, flags).

Once the service is up, open `/` to inspect live detections with multi-camera control, or `/atak` to embed the map-only feed in wider command tooling.

### Compose services

| Service | Command executed in container | Purpose |
|---------|---------------------------------|---------|
| `cuda-app` | `cuda-app` | Vector add demo / sanity check |
| `vision-demo` | `cuda-app vision-demo /dev/video0 models/yolov12n-face.torchscript 640 640` | TorchScript + webcam HUD |

Run them with `docker compose up <service>` or `docker compose run --rm <service> [...]`.

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
   cargo run --release -p cuda-app --features with-tch -- \
      vision-demo 0 models/yolov12n-face.torchscript 640 640 --cpu
   ```

   Drop `--cpu` to use CUDA inference. If your camera outputs H.264 you can let
   FFmpeg+NVDEC handle decode with:

   ```bash
   cargo run --release -p cuda-app --features with-tch -- \
      vision-demo /dev/video0 models/yolov12n-face.torchscript 640 640 --nvdec
   ```

   (Requires an FFmpeg build with CUDA/NVDEC support.) The current pipeline
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

## GPU pipeline details

- Frames are captured as BGR8 via V4L and uploaded directly to CUDA without an
  intermediate RGBA conversion. The resize, normalisation, tensor build, and
  YOLO post-processing now live in CUDA kernels, so the annotated surface never
  leaves the device until the very end of the pipeline.
- Bounding boxes and labels are rasterised on the GPU against that BGR surface.
- The final JPEG bitstream is produced with `nvJPEG` on the same CUDA stream, so
  the CPU simply pushes the encoded bytes to the HTTP endpoints. The legacy CPU
  path still exists and can be forced with `--cpu` for machines without CUDA.
- Passing `--nvdec` spawns an FFmpeg helper process configured with
  `h264_cuvid`, letting NVDEC handle hardware H.264 decode before frames enter
  the CUDA preprocessing path. Without the flag the capture thread uses MJPEG
  via V4L as before.

### nvJPEG runtime requirement

`nvjpeg-sys` links against NVIDIA's `libnvjpeg`. Install the CUDA toolkit (or at
least the nvJPEG runtime) and surface the CUDA/LibTorch binaries before running
the demo. On Arch Linux the following environment works end-to-end:

```bash
export CUDA_HOME=/opt/cuda
export LIBTORCH=/opt/libtorch
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export LIBTORCH_BYPASS_VERSION_CHECK=1
# Optional: Nsight tools in PATH if you profile
export PATH=$PATH:/opt/nsight-systems/2025.3.2/target-linux-x64:/opt/nsight-compute/2025.3.1
```

For `--nvdec` you also need an FFmpeg build with CUDA/NVDEC enabled (Arch's
`ffmpeg` package ships with it).

- NVDEC capture is only meaningful when the device actually streams H.264. Use
  `v4l2-ctl --device=/dev/video0 --list-formats-ext` to confirm the available
  formats. If the camera only advertises `MJPG`/`YUYV`, the `--nvdec` flag will
  fail—stick with the default MJPEG path in that case.

- NVDEC/NVENC headers come from NVIDIA’s Video Codec SDK. Download the SDK (and
  accept the license) from https://developer.nvidia.com/nvidia-video-codec-sdk/download.

### Remaining CPU hotspots

The capture thread (V4L I/O), TorchScript host invocations, HTTP streaming, and
JPEG transport still execute on the CPU. To offload those pieces entirely you
will need GPU capture and streaming primitives (NVDEC/NVENC or a GPU-aware
transport stack); the current code path is ready to integrate with them once the
dependencies are available.
