#!/usr/bin/env python3
"""Run the current ECE 257 segmentation model on an image, folder, or video."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Image, video, or folder to process")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, help="Optional checkpoint path; defaults to the current v3 model")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--name", default="latest")
    args = parser.parse_args()

    project = args.project.resolve()
    model_path = args.model.resolve() if args.model else project / "models" / "best_colored_turn_v3.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not args.source.exists():
        raise FileNotFoundError(f"Input not found: {args.source}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0.0 and 1.0")

    model = YOLO(str(model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    output_root = project / "predictions"
    model.predict(
        source=str(args.source.resolve()),
        conf=args.conf,
        imgsz=640,
        device=device,
        save=True,
        project=str(output_root),
        name=args.name,
        exist_ok=True,
    )
    print(f"Finished. Results are in: {output_root / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
