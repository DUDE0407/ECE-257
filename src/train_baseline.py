#!/usr/bin/env python3
"""Train the initial ECE 257 YOLO segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU training run")
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(args.project / "config" / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project=str(args.project / "runs"),
        name="baseline_v1",
        pretrained=True,
        patience=15,
        seed=257,
        deterministic=True,
        plots=True,
        cache=False,
        fliplr=0.5,
        degrees=5.0,
        translate=0.12,
        scale=0.35,
        perspective=0.0005,
        mosaic=0.8,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
