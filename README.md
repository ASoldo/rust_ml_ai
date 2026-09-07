# Vision GPU Pipeline

The repository contains a Rust workspace for GPU-first perception. The `vision` binary is the product that will ship to field units; everything else exists to validate drivers, kernels, or training workflows before we deploy. Simple samples (for example the vector add) are diagnostics only—they help developers verify that CUDA, toolchains, and shared libraries resolve correctly on a new machine.

See [camera release notes](docs/vision-camera-release.md) for the face/pose/silhouette configuration, recovery behavior, model contracts and measured validation.

## Workspace Overview
- `vision` — production pipeline: capture → detection → annotation → web delivery, with bounded queues and configurable model workers. The current 640×480 face/pose/silhouette path has been tested on an 8 GB RTX 3070.
- `gpu-kernels` — CUDA kernels built with NVRTC/CUDARC for preprocessing, overlay, and nvJPEG stages.
- `ml-core` — TorchScript loader plus training helpers (MNIST sample, detector bootstrap utilities).
- `video-ingest` — capture backends (V4L2 MJPEG fallback and FFmpeg+NVDEC H.264 hardware decode).
- `viz` — auxiliary visualisation utilities (not required for deployment, used during exploration).

## Vision System at a Glance
- Capture: `video-ingest` streams camera frames into bounded queues, using NVDEC when `--nvdec` is set.
- Inference: `ml-core::detector` loads TorchScript weights to GPU (or CPU when `--cpu` is chosen), running batched detection with custom input resolution.
- Tracking & Annotation: bounded image-space association stabilises track IDs and display geometry. CPU and GPU JPEG paths share a sparse BGR overlay renderer; GPU inference and NVJPEG encoding remain on CUDA.
- Encoding: each encode job owns its annotated capture-sized BGR pixels, then uploads them for nvJPEG encoding; CPU fallback uses `image`.
- Serving: Actix Web exposes `/`, `/atak`, `/operator`, `/frame.jpg`, `/stream.mjpg`, `/detections`, and `/stream_detections` so HUD clients and TAK systems subscribe in real time.

### Processing Loop
1. A capture thread reads from the configured device or URI and normalises resolution.
2. Frames are scheduled into a bounded processing queue; overload drops frames rather than growing queues without bound.
3. The detection worker loads TorchScript once, pushes frames through CUDA preprocessing, and performs inference.
4. Detections are scaled back to source resolution, labeled, and drawn by the shared CPU renderer before GPU or CPU JPEG encoding.
5. Encoded JPEG payloads are published to shared state consumed by HTTP routes and SSE streams.
6. Ctrl+C or fatal errors trip an atomic flag; workers drain queues, join threads, and report shutdown.

## GPU Acceleration Highlights
- NVRTC compiles kernels at runtime so we can tailor preprocessing to the model (resize, normalise, NMS).
- CUDA streams accelerate preprocessing and inference; nvJPEG encodes each owned annotation surface after its upload.
- FFmpeg + `h264_cuvid` unlocks NVDEC decode, reducing CPU usage when cameras stream H.264.
- On compact edge devices we avoid desktop-class dependencies; library loading is gated behind feature flags.
- Logging is concise: device availability, detector load, HTTP endpoint exposure, and controlled shutdown.

## Resilience & Recovery
- A watchdog samples heartbeats from capture, processing, and encoding stages; stalled components trigger an automatic pipeline restart.
- The supervisor loop restarts failed runs (with back-off) while still honouring operator Ctrl+C to shut down cleanly.
- A ring buffer stores the last 64 annotated frames so clients can recover gaps via `GET /frame.jpg?frame=<seq>`.
- Streaming endpoints tag payloads with monotonically increasing sequence IDs; SSE adds `id`/`retry` hints so frontline apps can reconnect and resynchronise after telemetry drops.

## Configuration and Flags
- `--source <uri>` — preferred way to specify the capture source (e.g. `/dev/video0`, `rtsp://user:pass@ip:554/stream`, `udp://127.0.0.1:5000`). Positional form `<camera-uri>` is still accepted for backwards compatibility.
- `--model <path>` — TorchScript weights. Positional form `<model-path>` remains valid.
- `--pose-model <path>` — optional YOLO26 COCO pose TorchScript export. Adds `PERSON` boxes, 17 named body/facial keypoints and a skeleton alongside the existing `FACE` detections. Export with `python tools/export_pose.py` in an environment with `ultralytics==8.4.142` and Torch 2.9.0. The helper verifies the NMS-free `[1,300,57]` contract and writes a checksum manifest.
- `--seg-model <path>` — optional YOLO26n-seg person silhouettes beneath the skeleton; requires `--pose-model`. Export with `tools/export_segmentation.py`.
- `--width <px>` / `--height <px>` — frame resolution to feed through the pipeline (positional form also works).
- `vision <camera-uri> <model-path> <width> <height>` — legacy positional invocation (still supported).
- `--cpu` forces CPU inference and CPU overlay for machines without CUDA.
- `--nvdec` switches capture to FFmpeg/NVDEC (requires H.264 input and CUDA-enabled FFmpeg).
- `--verbose` prints detection counts, dropped frame diagnostics, and bounding box dumps.
- `--detector-width/--detector-height` run inference at a lower resolution than the preview stream.
- `--jpeg-quality <1-100>` tunes nvJPEG output (higher is better fidelity, lower is faster/lighter).
- `--processors <n>` spins up that many concurrent detector workers (default `1`). Each worker maintains its own TorchScript module and CUDA state.
- `--batch-size <n>` lets a worker run up to `n` frames through the detector in a single call (default `1`). Higher values trade latency for throughput and only make sense on GPUs with ample compute.

> **Tip (edge devices):** start with `--processors 1 --batch-size 1`. Measure incoming frame rate, processing latency, and queue depth before increasing concurrency: a 15 FPS input or downstream frame-rate cap can limit playback even when the GPU has ample capacity.

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
    vision --source /dev/video0 --model models/yolov12n-face.torchscript \
    --width 640 --height 640 --verbose
  ```

- Test an RTSP feed (software decode):
  ```bash
  cargo run --release -p vision --features with-tch -- \
    vision --source rtsp://user:pass@camera/stream --model models/yolov12n-face.torchscript \
    --width 1280 --height 720 --verbose
  ```
- Consume a UDP/RTP feed (e.g. produced by `gst-launch-1.0`):
  ```bash
  cargo run --release -p vision --features with-tch -- \
    vision --source udp://127.0.0.1:5000?sprop=Z/QAFpGWgKA9sBagIMDIAAADAAgAAAMA9HixdQ==,aO8xkhk= \
    --model models/yolov12n-face.torchscript --width 640 --height 480 --verbose
  ```
  When streaming H.264 over RTP you must supply the `sprop-parameter-sets` (copy the value printed by your sender; GStreamer shows it in the pipeline caps). Append `?sprop=<base64 SPS>,<base64 PPS>` to the UDP URI and optionally `&payload=<pt>` if you use a payload type other than 96.
- Prefer NVDEC when targeting H.264 streams on capable GPUs:
  ```bash
  cargo run --release -p vision --features with-tch -- \
    vision --source rtsp://user:pass@camera/stream --model models/yolov12n-face.torchscript \
    --width 1280 --height 720 --nvdec
  ```

## Web Interfaces
- `/operator` — self-contained camera operator console: live video, selectable body/face tracks, confidence and standard pose details, source freshness, processing latency, matched-frame hold, JPEG snapshots and a video focus view. No external fonts, map tiles or JavaScript services are required. Existing 3D/map views remain available.
- `/` — Recon HUD (3D scene, camera rig widgets, live metrics).
- `/atak` — ATAK-style map for command operators.
- `/frame.jpg` — latest annotated JPEG (good for integrating with legacy dashboards). Append `?frame=<sequence>` to request a specific buffered frame when links hiccup. `&strict=true` returns 404 for an expired frame; the default still falls back to latest with a Warning header. JPEG responses carry `X-Sequence` and disable caching so a held image can be matched to its metadata.
- `/stream.mjpg` — MJPEG stream at ~30 Hz for HUD clients.
- `/detections` — JSON snapshot of detections, timestamps, FPS.
- `/stream_detections` — Server-Sent Events stream with periodic detection updates, sequence IDs, and reconnection hints.

## CLI Summary
- `vision ...` — starts the production pipeline.
- `mnist-train` / `mnist-predict` — retained for training exercises and TorchScript export validation.
- `mnist-help` — usage overview for the MNIST utilities.
- No other subcommands are required for production.

## Developer Diagnostics
- `cargo run -p vision` (no arguments) executes the GPU vector add sample to validate CUDA setup.
- `just vision` runs the release build with the `with-tch` feature and default device/model arguments.
- `just vision-nvdec` toggles NVDEC for H.264 inputs.
- `just vision-rtsp` runs the pipeline against an RTSP URI (override `source=…` as needed; append `flags='--nvdec'` to force GPU decode).
- `just gst-rtsp-server` spawns a lightweight RTSP server on port 8554 backed by `/dev/video0` for local testing.
- `just gst-udp-stream` starts a local GStreamer UDP sender (useful if you want to feed another restreamer).
- `just vision-batch` showcases a heavier configuration (`--processors 4 --batch-size 2`) for benchmarking on larger GPUs.
- Use `just check`, `just fmt`, and `just lint` to keep the workspace clean.

## Environment Requirements
- NVIDIA GPU with compatible drivers (`nvidia-smi` must succeed).
- CUDA toolkit or runtime providing `libnvjpeg` and headers for NVRTC (installed by Docker image or manually).
- FFmpeg with CUDA/NVDEC (`ffmpeg -decoders | grep cuvid` should list `h264_cuvid` if available).
- TorchScript weights placed under `models/` (for example `yolov12n-face.torchscript`).
- For MNIST exercises: dataset files under `data/mnist/`.
- When running in Docker, install NVIDIA Container Toolkit and expose the target camera device.

## Extensibility Notes
- Face-plus-pose mode reuses the same 640×640 input tensor for both GPU models. The face export must have five channel-first outputs; the pose export must have 57 values per detection (xyxy box, score, person class, 17 x/y/confidence triples). Other model layouts fail explicitly instead of silently mislabelling scores as classes. Keep `--detector-width 640 --detector-height 640 --batch-size 1` for the supplied fixed-size exports.
- `/detections` and SSE include a `keypoints` array on each person, with named points, confidence and capture-image coordinates. The rendered JPEG/RTSP feed shows only joints with confidence ≥0.5 inside the image; links require both endpoints to be visible. Faces have independent boxes and are not suppressed by person boxes. Pose points locate features; the pose model does not create independent face detections.
- Optional person silhouettes use `--seg-model models/yolo26n-seg.torchscript` together with `--pose-model`. Export using `tools/export_segmentation.py` in the same isolated Ultralytics environment. The fixed-size end-to-end contract is a tuple of `[1,300,38]` detection rows and `[1,32,160,160]` mask prototypes, validated on CPU and CUDA by the exporter. Runtime inference and mask assembly remain Rust/TorchScript.
- Segmentation reuses the existing input tensor, selects COCO person class zero, and reconstructs at most 16 current-frame masks at confidence ≥0.25. Masks attach one-to-one to person poses at box IoU ≥0.5; unmatched poses retain their normal overlay. Logits are bilinearly resized to capture geometry before thresholding and box cropping. A faint green fill and thin antialiased contour appear underneath the unchanged skeleton. No mask is held across missing detections. `silhouette_score` reports the independent mask confidence; transient mask rasters are discarded after drawing rather than stored in JSON or frame history.
- `vision_model_inference_seconds{model="person_segmentation"}` measures the additional model, mask assembly and association cost. Compare total processing and distinct capture FPS before/after enabling it; encoded output FPS can include repeated frames.
- `tools/compare_trackers.py INPUT.jsonl OUTPUT.json` replays timestamped native `/detections` samples through standard ByteTrack and OC-SORT configurations in the export environment. It reports observations, track counts and candidate update cost, with limitations. It does not change the native tracker or provide identity accuracy without labelled ground truth.
- `vision_model_inference_seconds{model="face"}` and `{model="person_pose"}` distinguish the model costs; the existing total processing/inference metrics include both.
- `track_id` is now a short-lived camera-session track, associated using same-class box overlap and motion. Tracks expire after a one-second gap, never draw missing detections, and reset on pipeline restart or geometry changes. Three matched observations change `track_state` from `NEW` to `TRACK`. These are local image-space tracks, not identity recognition; crossings and heavy occlusion can still change IDs. One processing worker gives the most consistent temporal order. Late frames from parallel workers are marked `UNTRACKED` and do not rewind stored tracks.
- Raw `bbox`, `keypoints` and confidence stay unchanged. `display_bbox` and `display_keypoints` expose the smoothed geometry used in the overlay. `person_track_id` links a face only when exactly one current person box plausibly contains it. Same-class duplicates above 0.8 IoU are suppressed; face boxes are retained inside person boxes.
- The operator overlay uses green corner brackets, antialiased COCO 17-point/19-link skeletons with ring joint markers, compact translucent labels and small corner readouts for UTC capture time, source FPS and detection counts. `POSE n/17` counts currently confident visible joints. `NEW` is amber and `TRACK` is green, with text carrying the same state information. Those colors do not encode military affiliation. Labels prefer positions outside face/joint regions; short leaders anchor displaced tags and distant/crowded tags are suppressed. The display reports model detections and keypoints without custom activity labels.
- Pose links and joint markers are green, with a dark outline for contrast. Skeleton strokes are approximately two pixels wide, with slope-corrected antialiasing so diagonal links stay visible. The skeleton topology remains the standard COCO/Ultralytics connections; no inferred neck or spine points are added.
- This is a tactical-style camera HUD, not a MIL-STD-2525 or APP-6 military symbol implementation or certification. The design uses limited colors and redundant text for readability. [Ultralytics skeleton plotting](https://docs.ultralytics.com/reference/utils/plotting/), [tracking concepts](https://docs.ultralytics.com/modes/track/), [MIL-STD-2525 scope](https://quicksearch.dla.mil/qsDocDetails.aspx?ident_number=114934), [display color guidance](https://hfcc.dot.gov/publications/docs/GeneralGuidance/zz_FAA_GeneralGuidanceDoc_Chapter_03_Section_07.html).
- Add new HTML surfaces under `crates/vision/src/html/` and export them via `html/mod.rs`.
- The operator console embeds `html/operator.html` directly. It decodes the existing MJPEG stream in the browser, uses multipart `Content-Length` and `X-Sequence` headers to skip repeated frames, and measures video and detection freshness independently. Live selection follows the most recent sampled detection geometry; Hold fetches an exact JPEG/metadata pair for inspection. Filters affect the track list; the green pose and detection labels are baked into the camera stream. Held-frame review is explicitly labelled while the source health continues updating. Stale live detections are cleared after 1.5 seconds, source interruption after five seconds; temporary IDs are reset on a detected pipeline restart. The 60-second rate graph and recent metric deltas are local to the open browser, with gaps shown for missing data. This original console uses public [Anduril entity presentation concepts](https://developer.anduril.com/guides/entities/overview) as a reference for source and track organization; it is not a Lattice integration.
- Additional detectors can piggyback on the existing TorchScript loader; ensure input resolution matches exported shapes.
- For multi-camera deployments, spawn multiple capture threads and publish additional MJPEG endpoints—the worker design already uses bounded queues and atomic shutdown flags.

## Troubleshooting
- No CUDA devices: set `--cpu`, confirm drivers, or run the vector add sanity check.
- NVDEC errors: confirm the camera really outputs H.264 and that FFmpeg was built with CUDA.
- RTSP startup allows up to 30 seconds for the first frame from each stage. After a stage starts, the watchdog retains its 1.5-second stall deadline. Capture waits poll shutdown, and stopping or restarting the pipeline kills and reaps its FFmpeg child.
- If a private Torch/OpenCV runtime conflicts with system FFmpeg, set `VISION_FFMPEG_BIN=/usr/bin/ffmpeg` and `VISION_FFMPEG_CLEAN_LIBRARY_PATH=1`. Only the FFmpeg child drops `LD_LIBRARY_PATH` and `LD_PRELOAD`; the vision process keeps its native runtime settings.
- HTTP/HTTPS multipart JPEG sources use the owned FFmpeg reader as well, for example `--source http://127.0.0.1:8556/raw`. A local authenticated relay can keep camera credentials out of process arguments and logs. This input option does not change the output stream address.
- HTTP capture uses a five-second running-stage deadline so short TCP retransmissions do not repeatedly disconnect downstream viewers. A sustained stall still triggers recovery. GPU annotations use the original capture geometry and queue owned pixel snapshots so a later frame cannot overwrite an earlier frame's image.
- CUDA kernel errors: rebuild with `--verbose` to capture stack traces, verify `libtorch_cuda*.so` preload works (`vision` automatically attempts to load them when CUDA mode is selected).
- High latency: lower inference resolution via `--detector-width/--detector-height`, reduce `--batch-size`, or keep `--processors` at 1 on underpowered GPUs.
- The HUD and `vision_pipeline_fps` report received frames over a two-second moving window, including network gaps. Short bursts no longer inflate the rate by averaging reciprocal frame intervals. To locate a bottleneck, compare `/metrics` capture/processing/encoding counter deltas over the same interval, stage latencies, queue depths, and the downstream encoder's configured frame rate. The MJPEG preview repeats the latest image at about 30 Hz; that refresh rate does not mean 30 distinct camera frames arrived.

## Licensing
- The workspace is distributed under MIT (see `LICENSE`). Honor third-party licenses for CUDA, FFmpeg, OpenCV, LibTorch, and any model checkpoints.
