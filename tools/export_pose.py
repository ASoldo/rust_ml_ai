#!/usr/bin/env python3
"""Export the optional YOLO26 pose model for the Rust detector.

Validated with ultralytics==8.4.142 and torch==2.9.0 (CPU export).
Run in an isolated export environment; inference remains entirely in Rust.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="models/yolo26n-pose.pt")
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    if model.task != "pose" or model.names != {0: "person"}:
        raise ValueError("Expected a single-class COCO person pose model")
    output = Path(model.export(
        format="torchscript", imgsz=640, batch=1, device="cpu", nms=False, optimize=False,
    ))
    module = torch.jit.load(str(output)).eval()
    with torch.inference_mode():
        tensor = module(torch.zeros(1, 3, 640, 640))
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, 300, 57):
        raise ValueError(f"Wrong pose export contract: {getattr(tensor, 'shape', type(tensor))}")
    validated_devices = ["cpu"]
    if torch.cuda.is_available():
        # Loading onto CUDA also remaps traced constant anchor tensors.
        module = torch.jit.load(str(output), map_location="cuda:0").eval()
        with torch.inference_mode():
            tensor = module(torch.zeros(1, 3, 640, 640, device="cuda:0"))
        if tuple(tensor.shape) != (1, 300, 57):
            raise ValueError("CUDA output differs from the CPU contract")
        validated_devices.append("cuda:0")
    manifest = {
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "source": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "export_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "input": [1, 3, 640, 640], "output": [1, 300, 57],
        "layout": "xyxy, confidence, class_id=0 (person), 17 COCO x/y/confidence keypoints",
        "nms": False,
        "validated_devices": validated_devices,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
