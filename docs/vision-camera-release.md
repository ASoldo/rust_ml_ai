# Camera pipeline: face, pose and silhouette release

The pipeline now preserves the camera's capture geometry through queued GPU encoding, recovers from silent network capture, and draws current face/body observations directly into the video consumed by ATAK. The browser console is an additional inspection surface; it does not control the overlays baked into video.

## Behavior

- HTTP MJPEG, RTSP and UDP capture use owned FFmpeg readers. Dropping a reader terminates and reaps its child even during a blocked read. Network timeouts and supervisor retries bound recovery; RTSP probing retains initial packets/keyframes. The capture loop polls so shutdown and watchdog requests can interrupt a quiet source.
- The watchdog allows 30 seconds for startup, then uses the existing 1.5-second stall deadline, or five seconds for HTTP MJPEG. Preview shutdown closes unbounded MJPEG/SSE clients immediately so reconnection is not delayed by a graceful HTTP timeout.
- GPU encode jobs own capture-sized annotation pixels. Later preprocessing cannot replace an earlier queued frame or change its aspect ratio. Frame rate is measured over arrivals rather than averaging reciprocal intervals, which exaggerated burst delivery.
- Separate face and person-pose models share one preprocessed tensor. Faces remain independent when contained inside a body. Pose includes the standard 17 COCO points and 19 links; confidence and image bounds gate both joints and links. No custom neck, spine or action labels are inferred.
- A bounded, one-second motion tracker assigns temporary per-camera IDs. Raw measurements remain in JSON; separate display geometry smooths boxes/keypoints. Face-to-body association requires one plausible current containing body. IDs are not identities; crossings and occlusion can change them.
- The video renderer uses compact translucent labels, corner readouts, green corner brackets and approximately two-pixel antialiased green skeleton strokes with ring joints. Amber NEW and green TRACK states also have text labels; colors do not represent affiliation.
- Optional segmentation reconstructs up to 16 current person masks at confidence ≥0.25. Masks match person poses one-to-one at IoU ≥0.5. Bilinear resizing precedes threshold/crop in capture coordinates. Faint green fill and a thin contour sit underneath the unchanged skeleton; missing masks are never held from earlier frames. Raster data is discarded after drawing; metadata exposes independent `silhouette_score`.
- `/operator` provides the camera, track list/inspection, 17 keypoint confidences, frame hold, snapshots and source health. Multipart lengths/sequence headers identify repeated frames. `GET /frame.jpg?frame=N&strict=true` returns the exact retained frame or 404, with no silent substitution; the legacy non-strict fallback remains. Stale browser input is marked, and stale live tracks are cleared.

## Build and run

Use a release executable for service startup and reconnects. With the matching CUDA/Torch/OpenCV environment configured:

```sh
cargo +1.91.0 build --locked --release -p vision -j 2
./target/release/vision vision \
  --source http://127.0.0.1:8556/raw \
  --model models/yolov12n-face.torchscript \
  --pose-model models/yolo26n-pose.torchscript \
  --seg-model models/yolo26n-seg.torchscript \
  --width 640 --height 480 \
  --detector-width 640 --detector-height 640 \
  --processors 1 --batch-size 1 --jpeg-quality 80
```

The source above is an example local relay. Configure the actual source and access credentials outside the repository. The preview is served on port 8080; an external publisher can convert `/stream.mjpg` to H.264/RTSP for ATAK. Configure a publisher to retry after both clean EOF and failures when upstream restarts. An encoded 30 FPS output can repeat a 15 FPS source.

The validated host uses Rust 1.91.0, Torch 2.9.0 with CUDA 12.8, OpenCV 4.12 and an isolated native library directory. Its build uses `cargo rustc --locked --release -p vision --bin vision -j 2 -- -L native=<native-library-directory>` with the preconfigured environment. Avoid a RAM-backed build target. The existing Docker image was not rebuilt or validated in this release; its older dependency defaults are not the tested host environment.

Service commands should name the absolute release executable and model paths. After rebuilding, restart the vision service only if needed to activate the new executable, verify that the running process resolves to that artifact, then confirm advancing fresh frames and decode the actual return endpoint. Preserve the previous executable/configuration for rollback. Removing `--seg-model` disables silhouettes on the next restart. Local saved service definitions, source configuration and camera evidence are deployment state, not repository assets.

## Models and output contracts

| Model | Exported output | Runtime use |
|---|---|---|
| Existing face model | `[B,5,N]` | FACE boxes |
| YOLO26n-pose | `[1,300,57]` | PERSON xyxy, score, class 0, 17 x/y/confidence points |
| YOLO26n-seg | `([1,300,38], [1,32,160,160])` | xyxy, score, COCO class, 32 coefficients; mask prototypes |

The optional exports require batch one and 640-square detector input. Wrong layouts fail explicitly. Pose and segmentation are distinct model tasks. `tools/export_pose.py` and `tools/export_segmentation.py` reproduce the TorchScript exports in an isolated `ultralytics==8.4.142`, Torch 2.9.0 environment. Both validate CPU/CUDA output contracts and write source/checksum manifests beside the artifacts. Live inference remains Rust/TorchScript. See [model provenance](../models/README.md).

Per-model metrics distinguish `face`, `person_pose` and `person_segmentation`; total processing, drawing, encoding, queue and capture-rate metrics remain available at `/metrics`.

## Validation and limits

On an RTX 3070 with a 640×480 camera source, the accepted skeleton baseline and silhouette-enabled stream were sampled for 12 and 15 seconds respectively:

| Measurement | Face + pose | Face + pose + silhouette |
|---|---:|---:|
| Distinct processed FPS | 15.00 | 14.99 |
| Mean processing | 10.04 ms | 17.64 ms |
| Mean drawing | 0.134 ms | 3.467 ms |
| Mean JPEG encoding | 0.412 ms | 0.489 ms |
| Additional segmentation/model-mask stage | — | 4.190 ms |
| Capture drops during sample | 0 | 0 |

The short windows contain camera movement and are not identical-input benchmarks or a maximum-throughput claim. A later host snapshot showed about 1.2 GiB total GPU allocation, 27% GPU utilization and 52°C; the vision service used about 1.1 GiB host memory. The current camera remained near 15 distinct FPS. A release process/artifact match, fresh silhouette metadata, and a decoded H.264 return frame were verified.

Validation includes 18 normal vision tests, three model decoder/association tests, two FFmpeg lifecycle tests, and two explicit GPU regressions. The GPU tests check queued frame ownership/capture geometry and combined face/pose/silhouette output; use a known person/face image via `VISION_TEST_IMAGE`. `VISION_TEST_OUTPUT` optionally saves the encoded fixture. Both GPU tests are ignored by default and must be requested explicitly. Operator console checks covered hold/resume, inspection/filtering, stale-data handling/recovery, desktop/mobile layout and JavaScript errors. Full workspace/container builds were not part of this release validation.

`tools/compare_trackers.py INPUT.jsonl OUTPUT.json` replays native detections through standard ByteTrack and OC-SORT configurations in the export environment. On 449 sampled frames across 30 seconds, both candidates reduced the number of person IDs, while neither retained as many face observations as the existing tracker. Candidate update costs for both classes were approximately 0.41 and 0.54 ms/frame in Python. The existing tracker stays: this scene has no identity ground truth, candidate confirmation can suppress detections, lost-track durations differ, and the native 0.25 score floor excludes lower-score ByteTrack recovery. This is not an appearance, crowded-scene or camera-motion benchmark.
