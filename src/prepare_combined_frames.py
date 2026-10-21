#!/usr/bin/env python3
"""Extract 80 diverse floor+obstacle samples from the new track media."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


VIDEO_COUNTS = {
    "4c85fb31bdc8fe6ab05c7c4d6d7a1c1f.mp4": 15,
    "583dcb966c249c056e35fd09c0972cdd.mp4": 11,
    "a639b752be5493dbb80623980443cfed.mp4": 21,
    "c3164b30f6af092612f861abede2f267.mp4": 25,
}


def save_jpeg(path: Path, frame) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output folder is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    records = []
    sequence = 1
    for video_name, count in VIDEO_COUNTS.items():
        video = args.source / video_name
        capture = cv2.VideoCapture(str(video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total <= 0 or fps <= 0:
            raise RuntimeError(f"Could not inspect {video}")
        indices = np.linspace(round(total * 0.05), round(total * 0.95), count, dtype=int)
        for sample_in_video, frame_index in enumerate(indices, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {video}")
            name = f"combined_{sequence:03d}.jpg"
            save_jpeg(output / name, frame)
            records.append(
                {
                    "image": name,
                    "source": video_name,
                    "source_type": "video",
                    "frame_index": int(frame_index),
                    "time_seconds": round(float(frame_index) / fps, 3),
                    "required_labels": ["floor", "obstacle"],
                }
            )
            sequence += 1
        capture.release()

    stills = sorted(args.source.glob("*.jpg"))
    if len(stills) != 8:
        raise RuntimeError(f"Expected 8 still photos, found {len(stills)}")
    for still in stills:
        name = f"combined_{sequence:03d}.jpg"
        with Image.open(still) as opened:
            ImageOps.exif_transpose(opened).convert("RGB").save(output / name, quality=94)
        records.append(
            {
                "image": name,
                "source": still.name,
                "source_type": "still",
                "frame_index": None,
                "time_seconds": None,
                "required_labels": ["floor", "obstacle"],
            }
        )
        sequence += 1

    if len(records) != 80:
        raise RuntimeError(f"Expected 80 samples, created {len(records)}")
    (output.parent / "combined_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Created {len(records)} samples in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
