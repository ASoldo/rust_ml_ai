#!/usr/bin/env python3
"""Replay sampled native /detections JSONL through standard motion-only trackers.

This measures output continuity and candidate runtime, not identity accuracy.
Use the isolated model export environment; no Python enters the live pipeline.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import ultralytics
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.oc_sort import OCSORT


def describe(outputs):
    return {
        label: {
            "assigned_observations": sum(len(f.get(label, [])) for f in outputs),
            "frames_with_track": sum(bool(f.get(label)) for f in outputs),
            "distinct_track_ids": len({i for f in outputs for i in f.get(label, [])}),
        }
        for label in ("PERSON", "FACE")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frames = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if len(frames) < 2:
        raise ValueError("Need at least two distinct frames")
    config_dir = Path(ultralytics.__file__).parent / "cfg/trackers"
    native = [{c: [d["track_id"] for d in f["detections"] if d["class"] == c] for c in ("PERSON", "FACE")} for f in frames]
    result = {
        "ultralytics": ultralytics.__version__,
        "frames": len(frames),
        "seconds": (frames[-1]["timestamp_ms"] - frames[0]["timestamp_ms"]) / 1000,
        "unsampled_source_frames": frames[-1]["frame_number"] - frames[0]["frame_number"] + 1 - len(frames),
        "native_observed": describe(native),
        "candidates": {},
        "limitations": [
            "No identity ground truth; fewer IDs can also mean incorrect merges.",
            "Native observations include tracks already active before recording.",
            "Input has the native 0.25 confidence floor and duplicate suppression; ByteTrack cannot recover discarded lower scores.",
            "Candidate defaults retain lost tracks for 30 frames, versus one second in the native tracker.",
            "Two independent class trackers avoid face/person cross-association.",
            "Candidate Python update time excludes model inference and does not predict a Rust port's speed.",
            "No appearance or camera motion compensation is compared; this is a short motion-only replay.",
        ],
    }
    for name, cls in (("ByteTrack", BYTETracker), ("OC-SORT", OCSORT)):
        config = yaml.safe_load((config_dir / ("bytetrack.yaml" if name == "ByteTrack" else "ocsort.yaml")).read_text())
        trackers = {c: cls(SimpleNamespace(**config)) for c in ("PERSON", "FACE")}
        outputs, durations = [], []
        last_frame = frames[0]["frame_number"] - 1
        for frame in frames:
            assignment = {}
            for label, tracker in trackers.items():
                # Retain elapsed frame count when the HTTP sampler missed a frame.
                tracker.frame_id += max(0, frame["frame_number"] - last_frame - 1)
                rows = np.array([d["bbox"] + [d["score"], 0] for d in frame["detections"] if d["class"] == label], dtype=np.float32).reshape(-1, 6)
                boxes = Boxes(rows, (480, 640))
                started = time.perf_counter()
                tracks = tracker.update(boxes)
                durations.append(time.perf_counter() - started)
                assignment[label] = [int(row[4]) for row in tracks]
            outputs.append(assignment)
            last_frame = frame["frame_number"]
        result["candidates"][name] = {
            "config": config,
            "outputs": describe(outputs),
            "mean_both_classes_ms": sum(durations) / len(frames) * 1000,
            "p95_class_update_ms": float(np.percentile(durations, 95) * 1000),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
