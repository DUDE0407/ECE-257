#!/usr/bin/env python3
"""Colab pipeline for the ECE 257 combined floor+obstacle segmentation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM, YOLO


ROOT = Path("/content/ece257_v2")
CLASSES = {"floor": 0, "obstacle": 1}


def reset_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def mask_to_polygon(mask: np.ndarray, width: int, height: int) -> list[list[float]]:
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50:
        return []
    epsilon = max(2.0, 0.0025 * cv2.arcLength(contour, True))
    contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in contour]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1.0)


def deduplicate_boxes(boxes: list[np.ndarray], threshold: float = 0.50) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for box in sorted(boxes, key=lambda item: float((item[2] - item[0]) * (item[3] - item[1])), reverse=True):
        if all(iou(box, existing) < threshold for existing in kept):
            kept.append(box)
    return kept


def shape(label: str, points: list[list[float]], method: str) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": method,
        "shape_type": "polygon",
        "flags": {"needs_review": True},
        "mask": None,
    }


def write_yolo(path: Path, shapes: list[dict], width: int, height: int) -> None:
    lines = []
    for item in shapes:
        coords = []
        for x, y in item["points"]:
            coords.extend((min(1.0, max(0.0, x / width)), min(1.0, max(0.0, y / height))))
        if len(coords) >= 6:
            lines.append(str(CLASSES[item["label"]]) + " " + " ".join(f"{value:.6f}" for value in coords))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_preview(image: Image.Image, shapes: list[dict], output: Path) -> None:
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    colors = {"floor": (30, 80, 245, 85), "obstacle": (245, 40, 35, 150)}
    for item in shapes:
        pts = [(round(x), round(y)) for x, y in item["points"]]
        color = colors[item["label"]]
        draw.polygon(pts, fill=color, outline=color[:3] + (255,), width=4)
        if pts:
            draw.text(pts[0], item["label"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    Image.alpha_composite(canvas, layer).convert("RGB").save(output, quality=90)


def floor_mask_from_baseline(result, width: int, height: int) -> np.ndarray | None:
    if result.masks is None or result.boxes is None:
        return None
    combined = np.zeros((height, width), dtype=np.uint8)
    for index, class_id in enumerate(result.boxes.cls.detach().cpu().tolist()):
        if int(class_id) != 0:
            continue
        mask = result.masks.data[index].detach().cpu().numpy()
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        combined |= (mask > 0.5).astype(np.uint8)
    return combined if combined.any() else None


def baseline_obstacle_boxes(result) -> list[np.ndarray]:
    if result.boxes is None:
        return []
    boxes = []
    for index, class_id in enumerate(result.boxes.cls.detach().cpu().tolist()):
        if int(class_id) == 1:
            boxes.append(result.boxes.xyxy[index].detach().cpu().numpy().astype(float))
    return boxes


def annotate(images_zip: Path, baseline_model: Path) -> None:
    raw = ROOT / "raw_images"
    labelme = ROOT / "labelme_json"
    labels = ROOT / "new_yolo_labels"
    previews = ROOT / "annotation_previews"
    for path in (raw, labelme, labels, previews):
        reset_folder(path)
    with zipfile.ZipFile(images_zip) as archive:
        archive.extractall(raw)
    nested = list(raw.glob("**/combined_001.jpg"))
    if nested and nested[0].parent != raw:
        source_parent = nested[0].parent
        for item in source_parent.glob("*.jpg"):
            shutil.move(str(item), raw / item.name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = YOLO(str(baseline_model))
    sam = SAM("sam2_t.pt")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(dino_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    prompt = "white foam block. white cardboard box. rectangular foam obstacle."

    summary = []
    for number, image_path in enumerate(sorted(raw.glob("combined_*.jpg")), 1):
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]

        base_result = baseline.predict(rgb, conf=0.08, imgsz=640, device=0 if device == "cuda" else "cpu", verbose=False)[0]
        floor_mask = floor_mask_from_baseline(base_result, width, height)
        floor_method = "baseline_v1"
        if floor_mask is None or float(floor_mask.mean()) < 0.12:
            points = [[
                [0.50 * width, 0.95 * height],
                [0.50 * width, 0.78 * height],
                [0.50 * width, 0.60 * height],
                [0.50 * width, 0.42 * height],
                [0.08 * width, 0.35 * height],
                [0.92 * width, 0.35 * height],
                [0.50 * width, 0.08 * height],
            ]]
            sam_result = sam.predict(
                rgb,
                points=points,
                labels=[[1, 1, 1, 1, 0, 0, 0]],
                device=0 if device == "cuda" else "cpu",
                verbose=False,
            )[0]
            if sam_result.masks is None:
                raise RuntimeError(f"No floor mask for {image_path.name}")
            floor_candidates = sam_result.masks.data.detach().cpu().numpy() > 0.5
            floor_mask = max(floor_candidates, key=lambda item: int(item.sum()))
            floor_method = "sam2_floor_fallback"

        proximity_size = max(21, int(round(0.045 * height))) | 1
        floor_near = cv2.dilate(
            floor_mask.astype(np.uint8),
            np.ones((proximity_size, proximity_size), np.uint8),
        ) > 0

        candidate_boxes = baseline_obstacle_boxes(base_result)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = detector(**inputs)
        detected = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=0.28,
            text_threshold=0.22,
            target_sizes=[(height, width)],
        )[0]
        for box in detected["boxes"].detach().cpu().numpy():
            x1, y1, x2, y2 = box.astype(float)
            area_fraction = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)
            bottom_x = int(np.clip(round((x1 + x2) / 2), 0, width - 1))
            bottom_y = int(np.clip(round(y2), 0, height - 1))
            near_floor = bool(floor_near[bottom_y, bottom_x])
            if 0.00020 <= area_fraction <= 0.12 and y2 >= 0.20 * height and near_floor:
                candidate_boxes.append(box.astype(float))
        candidate_boxes = deduplicate_boxes(candidate_boxes)

        shapes = []
        floor_polygon = mask_to_polygon(floor_mask, width, height)
        if floor_polygon:
            shapes.append(shape("floor", floor_polygon, floor_method))

        accepted_boxes = []
        for box in candidate_boxes:
            sam_result = sam.predict(rgb, bboxes=box.tolist(), device=0 if device == "cuda" else "cpu", verbose=False)[0]
            if sam_result.masks is None:
                continue
            mask = sam_result.masks.data[0].detach().cpu().numpy() > 0.5
            area_fraction = float(mask.mean())
            if not (0.00012 <= area_fraction <= 0.12):
                continue
            ys, xs = np.where(mask)
            if not len(xs):
                continue
            mask_width = int(xs.max() - xs.min() + 1)
            mask_height = int(ys.max() - ys.min() + 1)
            # Hallway light reflections are usually very flat horizontal streaks,
            # unlike the foam/cardboard blocks used as physical obstacles.
            if mask_width / max(mask_height, 1) > 3.0:
                continue
            if ys.min() <= 0.02 * height:
                continue
            bottom_start = int(max(ys.min(), ys.max() - 0.25 * (ys.max() - ys.min() + 1)))
            bottom_band = mask.copy()
            bottom_band[:bottom_start] = False
            support = float(np.logical_and(bottom_band, floor_near).sum()) / max(float(bottom_band.sum()), 1.0)
            if support < 0.10:
                continue
            polygon = mask_to_polygon(mask, width, height)
            if polygon:
                shapes.append(shape("obstacle", polygon, "grounding_dino_plus_sam2"))
                accepted_boxes.append([round(float(value), 1) for value in box])

        record = {
            "version": "6.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        (labelme / f"{image_path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        write_yolo(labels / f"{image_path.stem}.txt", shapes, width, height)
        make_preview(image, shapes, previews / image_path.name)
        summary.append(
            {
                "image": image_path.name,
                "floor_count": sum(item["label"] == "floor" for item in shapes),
                "obstacle_count": sum(item["label"] == "obstacle" for item in shapes),
                "obstacle_boxes": accepted_boxes,
            }
        )
        print(f"[{number:02d}/80] {image_path.name}: floor={summary[-1]['floor_count']} obstacles={summary[-1]['obstacle_count']}")

    (ROOT / "annotation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    create_contact_sheet(previews, ROOT / "annotation_contact_sheet.jpg")
    missing = [item["image"] for item in summary if item["floor_count"] != 1 or item["obstacle_count"] < 1]
    print(f"Annotation finished. Images requiring manual review: {len(missing)}")
    if missing:
        print("Missing/uncertain:", ", ".join(missing))


def create_contact_sheet(preview_dir: Path, output: Path) -> None:
    files = sorted(preview_dir.glob("*.jpg"))
    columns, thumb_w, thumb_h = 8, 180, 330
    rows = (len(files) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 8, thumb_h - 28))
        x = (index % columns) * thumb_w + (thumb_w - image.width) // 2
        y = (index // columns) * thumb_h
        sheet.paste(image, (x, y))
        draw.text(((index % columns) * thumb_w + 4, y + thumb_h - 24), path.stem, fill="black")
    sheet.save(output, quality=88)


def copy_dataset_tree(source: Path, target: Path) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src = source / kind / split
            dst = target / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            for item in src.glob("*"):
                shutil.copy2(item, dst / item.name)


def train(baseline_zip: Path, baseline_model: Path, epochs: int) -> None:
    baseline_root = ROOT / "baseline_dataset"
    merged = ROOT / "merged_dataset"
    reset_folder(baseline_root)
    reset_folder(merged)
    with zipfile.ZipFile(baseline_zip) as archive:
        archive.extractall(baseline_root)
    dataset_candidates = [baseline_root] + [path for path in baseline_root.iterdir() if path.is_dir()]
    source = next((path for path in dataset_candidates if (path / "images" / "train").is_dir()), None)
    if source is None:
        raise RuntimeError("Could not locate the baseline dataset inside the zip")
    copy_dataset_tree(source, merged)

    new_val = set(range(16, 27)) | set(range(73, 78))
    for index in range(1, 81):
        name = f"combined_{index:03d}"
        split = "val" if index in new_val else "train"
        shutil.copy2(ROOT / "raw_images" / f"{name}.jpg", merged / "images" / split / f"{name}.jpg")
        shutil.copy2(ROOT / "new_yolo_labels" / f"{name}.txt", merged / "labels" / split / f"{name}.txt")

    yaml_path = merged / "dataset.yaml"
    yaml_path.write_text(
        f"path: {merged}\ntrain: images/train\nval: images/val\nnames:\n  0: floor\n  1: obstacle\n",
        encoding="utf-8",
    )
    model = YOLO(str(baseline_model))
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project=str(ROOT / "runs"),
        name="combined_v2",
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


def package() -> None:
    output = Path("/content/ece257_combined_v2_results")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    for folder in ("labelme_json", "new_yolo_labels", "annotation_previews", "merged_dataset", "runs"):
        source = ROOT / folder
        if source.exists():
            shutil.copytree(source, output / folder)
    for name in ("annotation_summary.json", "annotation_contact_sheet.jpg"):
        source = ROOT / name
        if source.exists():
            shutil.copy2(source, output / name)
    archive = shutil.make_archive(str(output), "zip", output)
    print(f"Deliverable: {archive}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("annotate", "train", "package"))
    parser.add_argument("--images-zip", type=Path)
    parser.add_argument("--baseline-zip", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "annotate":
        annotate(args.images_zip, args.model)
    elif args.stage == "train":
        train(args.baseline_zip, args.model, args.epochs)
    else:
        package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
