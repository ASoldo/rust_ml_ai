# Vision GPU Pipeline

The repository contains a Rust workspace for GPU-first perception. The `vision` binary is the product that will ship to field units; everything else exists to validate drivers, kernels, or training workflows before we deploy. Simple samples (for example the vector add) are diagnostics only—they help developers verify that CUDA, toolchains, and shared libraries resolve correctly on a new machine.

## Workspace Overview
- `vision` — production pipeline: capture → detection → annotation → web delivery, tuned for edge devices and mid-to-top tier GPUs with 20 GB+ VRAM.
- `gpu-kernels` — CUDA kernels built with NVRTC/CUDARC for preprocessing, overlay, and nvJPEG stages.
- `ml-core` — TorchScript loader plus training helpers (MNIST sample, detector bootstrap utilities).
- `video-ingest` — capture backends (V4L2 MJPEG fallback and FFmpeg+NVDEC H.264 hardware decode).
- `viz` — auxiliary visualisation utilities (not required for deployment, used during exploration).

## Vision System at a Glance
- Capture: `video-ingest` streams camera frames into bounded queues, using NVDEC when `--nvdec` is set.
- Inference: `ml-core::detector` loads TorchScript weights to GPU (or CPU when `--cpu` is chosen), running batched detection with custom input resolution.
- Tracking & Annotation: detections are stabilised with a simple tracker, then overlaid via CUDA kernels (GPU path) or a minimal CPU fallback.
- Encoding: nvJPEG writes the annotated surfaces to JPEG without leaving device memory; CPU fallback uses `image` as a last resort.
- Serving: Actix Web exposes `/`, `/atak`, `/frame.jpg`, `/stream.mjpg`, `/detections`, and `/stream_detections` so HUD clients and TAK systems subscribe in real time.

### Processing Loop
1. A capture thread reads from the configured device or URI and normalises resolution.
2. Frames are scheduled into a bounded processing queue; overload drops oldest work to maintain latency.
3. The detection worker loads TorchScript once, pushes frames through CUDA preprocessing, and performs inference.
4. Detections are scaled back to source resolution, labeled, and each frame is annotated on GPU (preferred) or CPU.
5. Encoded JPEG payloads are published to shared state consumed by HTTP routes and SSE streams.
6. Ctrl+C or fatal errors trip an atomic flag; workers drain queues, join threads, and report shutdown.

## GPU Acceleration Highlights
- NVRTC compiles kernels at runtime so we can tailor preprocessing to the model (resize, normalise, NMS).
- CUDA streams and nvJPEG keep the annotated surface on-device, minimising PCIe copies.
- FFmpeg + `h264_cuvid` unlocks NVDEC decode, reducing CPU usage when cameras stream H.264.
- On compact edge devices we avoid desktop-class dependencies; library loading is gated behind feature flags.
- Logging is concise: device availability, detector load, HTTP endpoint exposure, and controlled shutdown.

## Configuration and Flags
- `vision <camera-uri> <model-path> <width> <height>` — base invocation.
- `--cpu` forces CPU inference and CPU overlay for machines without CUDA.
- `--nvdec` switches capture to FFmpeg/NVDEC (requires H.264 input and CUDA-enabled FFmpeg).
- `--verbose` prints detection counts, dropped frame diagnostics, and bounding box dumps.
- `--detector-width/--detector-height` run inference at a lower resolution than the preview stream.
- `--jpeg-quality <1-100>` tunes nvJPEG output (higher is better fidelity, lower is faster/lighter).

## Running the Vision Service

### Docker Compose (recommended for deployment)
1. Build the image (customise `CUDA_BASE` or `LIBTORCH_URL` as needed):
   ```bash
   docker compose build
   ```
2. Bring up the field pipeline with published ports:
   ```bash
   docker compose up vision
   ```
3. Launch ad-hoc runs with custom flags:
   ```bash
   docker compose run --rm --service-ports vision-cli \
     vision /dev/video0 models/yolov12n-face.torchscript 640 640 --nvdec
   ```
   The compose file shares `data/` and `models/` from the host and maps `/dev/video0`. Adjust the service command or environment to match your hardware.

### Direct Cargo on bare metal
- Ensure CUDA drivers, FFmpeg (with NVDEC), and TorchScript weights are installed locally.
- Build once to cache dependencies:
  ```bash
  cargo build -p vision --features with-tch --release
  ```
- Run the pipeline:
  ```bash
  cargo run --release -p vision --features with-tch -- \
    vision /dev/video0 models/yolov12n-face.torchscript 640 640 --verbose
  ```

## Web Interfaces
- `/` — Recon HUD (3D scene, camera rig widgets, live metrics).
- `/atak` — ATAK-style map for command operators.
- `/frame.jpg` — latest annotated JPEG (good for integrating with legacy dashboards).
- `/stream.mjpg` — MJPEG stream at ~30 Hz for HUD clients.
- `/detections` — JSON snapshot of detections, timestamps, FPS.
- `/stream_detections` — Server-Sent Events stream with periodic detection updates.

## CLI Summary
- `vision ...` — starts the production pipeline.
- `mnist-train` / `mnist-predict` — retained for training exercises and TorchScript export validation.
- `mnist-help` — usage overview for the MNIST utilities.
- No other subcommands are required for production.

## Developer Diagnostics
- `cargo run -p vision` (no arguments) executes the GPU vector add sample to validate CUDA setup.
- `just vision` runs the release build with the `with-tch` feature and default device/model arguments.
- `just vision-nvdec` toggles NVDEC for H.264 inputs.
- Use `just check`, `just fmt`, and `just lint` to keep the workspace clean.

## Environment Requirements
- NVIDIA GPU with compatible drivers (`nvidia-smi` must succeed).
- CUDA toolkit or runtime providing `libnvjpeg` and headers for NVRTC (installed by Docker image or manually).
- FFmpeg with CUDA/NVDEC (`ffmpeg -decoders | grep cuvid` should list `h264_cuvid` if available).
- TorchScript weights placed under `models/` (for example `yolov12n-face.torchscript`).
- For MNIST exercises: dataset files under `data/mnist/`.
- When running in Docker, install NVIDIA Container Toolkit and expose the target camera device.

## Extensibility Notes
- Add new HTML surfaces under `crates/app/src/html/` and export them via `html/mod.rs`.
- Additional detectors can piggyback on the existing TorchScript loader; ensure input resolution matches exported shapes.
- For multi-camera deployments, spawn multiple capture threads and publish additional MJPEG endpoints—the worker design already uses bounded queues and atomic shutdown flags.

## Troubleshooting
- No CUDA devices: set `--cpu`, confirm drivers, or run the vector add sanity check.
- NVDEC errors: confirm the camera really outputs H.264 and that FFmpeg was built with CUDA.
- CUDA kernel errors: rebuild with `--verbose` to capture stack traces, verify `libtorch_cuda*.so` preload works (`vision` automatically attempts to load them when CUDA mode is selected).
- High latency: raise the worker queue depth or lower inference resolution via `--detector-width/--detector-height`.

## Licensing
- The workspace is distributed under MIT (see `LICENSE`). Honor third-party licenses for CUDA, FFmpeg, OpenCV, LibTorch, and any model checkpoints.
