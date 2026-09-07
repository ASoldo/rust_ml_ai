# Model artifacts

The repository already carries the face detector and MNIST example. This release adds official Ultralytics YOLO26n pose and instance-segmentation weights and their validated TorchScript exports. Separate JSON manifests record the download source, weights/export SHA-256, export versions, input/output layouts and tested devices.

| Added model | Official weights | Export helper |
|---|---|---|
| YOLO26n-pose | [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt) | `tools/export_pose.py` |
| YOLO26n-seg | [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt) | `tools/export_segmentation.py` |

The exporters were validated with Ultralytics 8.4.142 and Torch 2.9.0. They use fixed 640-square, batch-one, end-to-end exports without added NMS. Python is needed only to export or evaluate; the production pipeline loads TorchScript in Rust.

Third-party model assets retain their upstream licensing terms. The [included upstream license text](LICENSE.ultralytics) accompanies these assets. See the [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and [licensing information](https://www.ultralytics.com/license); the workspace's MIT metadata does not relicense these model assets. Camera recordings, source credentials and machine-specific runtime files are not model artifacts and are not included here.
