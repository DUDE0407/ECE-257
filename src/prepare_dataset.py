#!/usr/bin/env python3
"""Build a 200-image LabelMe + YOLO segmentation dataset from ECE 257 media."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import SAM


CLASSES = {"floor": 0, "obstacle": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def save_cv_jpeg(path: Path, frame: np.ndarray, quality: int = 92) -> None:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    encoded.tofile(path)


def allocate_counts(weights: list[int], total: int) -> list[int]:
    raw = [total * value / sum(weights) for value in weights]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    return counts


def largest_polygon(mask: np.ndarray) -> list[list[float]]:
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("SAM returned no usable contour")
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(2.0, 0.0025 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(simplified) < 3:
        simplified = contour.reshape(-1, 2)[:: max(1, len(contour) // 40)]
    return [[float(x), float(y)] for x, y in simplified]


def labelme_record(image_name: str, width: int, height: int, shapes: list[dict]) -> dict:
    return {
        "version": "6.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def shape(label: str, points: list[list[float]], auto: str) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": auto,
        "shape_type": "polygon",
        "flags": {"needs_review": True},
        "mask": None,
    }


def write_yolo(path: Path, shapes: list[dict], width: int, height: int) -> None:
    lines = []
    for item in shapes:
        class_id = CLASSES[item["label"]]
        coords = []
        for x, y in item["points"]:
            coords.extend((min(1.0, max(0.0, x / width)), min(1.0, max(0.0, y / height))))
        if len(coords) >= 6:
            lines.append(str(class_id) + " " + " ".join(f"{value:.6f}" for value in coords))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def overlay_preview(image: Image.Image, shapes: list[dict], target: Path) -> None:
    canvas = ImageOps.exif_transpose(image).convert("RGBA")
    fill_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(fill_layer)
    colors = {"floor": (30, 210, 80, 90), "obstacle": (245, 45, 35, 130)}
    for item in shapes:
        pts = [(round(x), round(y)) for x, y in item["points"]]
        draw.polygon(pts, fill=colors[item["label"]], outline=colors[item["label"]][0:3] + (255,), width=4)
        if pts:
            draw.text(pts[0], item["label"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    Image.alpha_composite(canvas, fill_layer).convert("RGB").save(target, quality=90)


def extract_frames(videos: list[Path], counts: list[int], output: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for video_index, (video, count) in enumerate(zip(videos, counts)):
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Avoid the first/last 3%, which often contain phone handling motion.
        positions = np.linspace(frame_count * 0.03, frame_count * 0.97, count, dtype=int)
        for sample_index, frame_index in enumerate(positions):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {video}")
            name = f"video{video_index + 1:02d}_{sample_index + 1:03d}.jpg"
            target = output / name
            save_cv_jpeg(target, frame)
            records.append((target, "floor"))
        capture.release()
    return records


def copy_stills(source: Path, output: Path) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    floor_files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    obstacle_files = sorted(p for p in (source / "obstacle").iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for index, item in enumerate(floor_files, 1):
        target = output / f"still_floor_{index:03d}.jpg"
        ImageOps.exif_transpose(Image.open(item)).convert("RGB").save(target, quality=94)
        records.append((target, "floor", item.name))
    for index, item in enumerate(obstacle_files, 1):
        target = output / f"still_obstacle_{index:03d}.jpg"
        ImageOps.exif_transpose(Image.open(item)).convert("RGB").save(target, quality=94)
        records.append((target, "obstacle", item.name))
    return records


OBSTACLE_BOXES = [
    (0.30, 0.20, 0.80, 0.86),
    (0.29, 0.17, 0.66, 0.72),
    (0.29, 0.05, 0.65, 0.74),
    (0.35, 0.15, 0.60, 0.41),
    (0.10, 0.13, 0.71, 0.71),
    (0.22, 0.26, 0.61, 0.68),
    (0.35, 0.20, 0.57, 0.44),
    (0.24, 0.15, 0.57, 0.48),
]


def sam_polygon(model: SAM, image: Image.Image, kind: str, obstacle_index: int | None = None) -> list[list[float]]:
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    if kind == "floor":
        points = [[[0.50 * w, 0.93 * h], [0.50 * w, 0.79 * h], [0.50 * w, 0.64 * h],
                   [0.04 * w, 0.58 * h], [0.96 * w, 0.58 * h], [0.50 * w, 0.10 * h]]]
        labels = [[1, 1, 1, 0, 0, 0]]
        result = model.predict(rgb, points=points, labels=labels, device=0, verbose=False)[0]
        # Tight turns can put a wall at the usual middle-floor prompt. Retry with
        # only a near-field floor point so these difficult corner frames remain
        # in the dataset instead of being silently discarded.
        if result.masks is None or not len(result.masks.data):
            turn_points = [[[0.50 * w, 0.93 * h], [0.50 * w, 0.10 * h]]]
            result = model.predict(rgb, points=turn_points, labels=[[1, 0]], device=0, verbose=False)[0]
    else:
        assert obstacle_index is not None
        rel = OBSTACLE_BOXES[obstacle_index]
        box = [rel[0] * w, rel[1] * h, rel[2] * w, rel[3] * h]
        result = model.predict(rgb, bboxes=box, device=0, verbose=False)[0]
    if result.masks is None or not len(result.masks.data):
        raise RuntimeError(f"SAM produced no {kind} mask")
    mask = result.masks.data[0].detach().cpu().numpy() > 0.5
    return largest_polygon(mask)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--total", type=int, default=200)
    args = parser.parse_args()

    dataset = args.project / "dataset"
    labelme_dir = args.project / "labelme_json"
    preview_dir = args.project / "annotation_previews"
    staging = args.project / "staging_images"
    for path in (labelme_dir, preview_dir, staging):
        path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    still_records = copy_stills(args.source, staging)
    video_total = args.total - len(still_records)
    videos = sorted(args.source.glob("*.mp4"))
    frame_counts = []
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    allocations = allocate_counts(frame_counts, video_total)
    video_records = extract_frames(videos, allocations, staging)
    records: list[tuple[Path, str, str]] = [(p, kind, p.name) for p, kind in video_records] + still_records
    if len(records) != args.total:
        raise RuntimeError(f"Expected {args.total} samples, got {len(records)}")

    model = SAM(str(args.project / "sam2_t.pt"))
    annotations: dict[str, tuple[list[dict], int, int]] = {}
    obstacle_counter = 0
    failures = []
    for number, (image_path, kind, original_name) in enumerate(records, 1):
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        try:
            polygon = sam_polygon(model, image, kind, obstacle_counter if kind == "obstacle" else None)
            shapes = [shape(kind, polygon, "Auto-generated with SAM2; review before final training")]
            if kind == "obstacle":
                obstacle_counter += 1
        except Exception as exc:
            failures.append({"image": image_path.name, "error": str(exc)})
            continue
        record = labelme_record(image_path.name, image.width, image.height, shapes)
        (labelme_dir / f"{image_path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        annotations[image_path.name] = (shapes, image.width, image.height)
        if number % 20 == 1 or kind == "obstacle":
            overlay_preview(image, shapes, preview_dir / image_path.name)
        print(f"[{number:03d}/{len(records)}] {image_path.name}: {kind}", flush=True)

    if failures:
        (args.project / "annotation_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} annotations failed; see annotation_failures.json")
    failure_file = args.project / "annotation_failures.json"
    if failure_file.exists():
        failure_file.unlink()

    # Stratify so the small obstacle class appears in both splits.
    floor_names = sorted(name for name, (items, _, _) in annotations.items() if items[0]["label"] == "floor")
    obstacle_names = sorted(name for name, (items, _, _) in annotations.items() if items[0]["label"] == "obstacle")
    rng = random.Random(257)
    rng.shuffle(floor_names)
    rng.shuffle(obstacle_names)
    floor_val_count = max(1, round(len(floor_names) * 0.20))
    obstacle_val_count = max(1, round(len(obstacle_names) * 0.20))
    val_names = set(floor_names[:floor_val_count] + obstacle_names[:obstacle_val_count])
    names = floor_names + obstacle_names
    manifest = {"train": [], "val": [], "video_allocations": dict(zip((p.name for p in videos), allocations))}
    for name in names:
        split = "val" if name in val_names else "train"
        source_image = staging / name
        target_image = dataset / "images" / split / name
        shutil.move(str(source_image), str(target_image))
        shapes, width, height = annotations[name]
        label_path = dataset / "labels" / split / f"{Path(name).stem}.txt"
        write_yolo(label_path, shapes, width, height)
        manifest[split].append(name)

    yaml_text = (
        f"path: {dataset.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: floor\n  1: obstacle\n"
    )
    (dataset / "dataset.yaml").write_text(yaml_text, encoding="utf-8")
    (args.project / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(staging)
    print(f"DONE: {len(names)} samples; train={len(manifest['train'])}, val={len(manifest['val'])}")
    print(f"Video allocations: {allocations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
