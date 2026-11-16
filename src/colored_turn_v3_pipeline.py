#!/usr/bin/env python3
"""ECE 257 incremental YOLO segmentation training for colored obstacles and turns."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import SAM, YOLO


SOURCE = Path(r"D:\文件\ECE 257\images\obstacle\colored")
YOLO_ROOT = Path(r"D:\文件\ECE 257\yolo_segmentation")
PROJECT = YOLO_ROOT / "colored_turn_v3"
BASE_DATASET = YOLO_ROOT / "dataset"
COMBINED_ROOT = YOLO_ROOT / "combined_obstacle_v2"
COMBINED_RESULTS = COMBINED_ROOT / "ece257_combined_v2_results"
BASE_MODEL = COMBINED_RESULTS / "best_combined_v2.pt"
SAM_MODEL = YOLO_ROOT / "sam2_t.pt"
CLASSES = {"floor": 0, "obstacle": 1}
VIDEO_COUNTS = {
    "ae9ada52dc89fd12d45ce7792ec4b773.mp4": 25,
    "f6bfd2b36bd48dad9f88be8a528f73ae.mp4": 25,
    "f82ec4702befdbe623ca2311eed40c71.mp4": 41,
}


def require_new_project() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Project already exists; refusing to overwrite: {PROJECT}")
    for name in (
        "raw_images",
        "labelme_json",
        "yolo_labels",
        "annotation_previews",
        "merged_dataset/images/train",
        "merged_dataset/images/val",
        "merged_dataset/labels/train",
        "merged_dataset/labels/val",
    ):
        (PROJECT / name).mkdir(parents=True, exist_ok=True)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(path)


def sample_media() -> list[dict]:
    raw = PROJECT / "raw_images"
    records: list[dict] = []
    number = 1
    for source in sorted(SOURCE.glob("*.jpg")):
        target = raw / f"colored_turn_{number:03d}.jpg"
        shutil.copy2(source, target)
        records.append(
            {
                "image": target.name,
                "source": source.name,
                "source_type": "still",
                "frame_index": None,
                "time_seconds": None,
            }
        )
        number += 1

    for filename, count in VIDEO_COUNTS.items():
        source = SOURCE / filename
        capture = cv2.VideoCapture(str(source))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        positions = np.linspace(frame_count * 0.03, frame_count * 0.97, count, dtype=int)
        if len(set(int(value) for value in positions)) != count:
            raise RuntimeError(f"Sampling produced duplicate frame indexes: {source}")
        for frame_index in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from {source}")
            target = raw / f"colored_turn_{number:03d}.jpg"
            write_jpeg(target, frame)
            records.append(
                {
                    "image": target.name,
                    "source": source.name,
                    "source_type": "video",
                    "frame_index": int(frame_index),
                    "time_seconds": round(float(frame_index) / fps, 3) if fps else None,
                }
            )
            number += 1
        capture.release()

    if len(records) != 100:
        raise RuntimeError(f"Expected exactly 100 new samples, got {len(records)}")
    return records


def floor_mask_from_result(result, width: int, height: int) -> np.ndarray | None:
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
    return combined.astype(bool) if combined.any() else None


def color_components(image_bgr: np.ndarray, floor_mask: np.ndarray) -> list[tuple[np.ndarray, list[float]]]:
    height, width = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (((hue <= 25) | (hue >= 172)) & (saturation >= 180) & (value >= 100)).astype(np.uint8)
    colored[: int(0.07 * height)] = 0
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)
    components: list[tuple[np.ndarray, list[float]]] = []
    min_area = max(100, int(0.00020 * width * height))
    for component_id in range(1, count):
        x, y, w, h, area = stats[component_id]
        if area < min_area or h < 5 or w < 5:
            continue
        component = labels == component_id
        pad_x = max(7, int(0.018 * width))
        pad_y = max(7, int(0.012 * height))
        box = [
            float(max(0, x - pad_x)),
            float(max(0, y - pad_y)),
            float(min(width - 1, x + w + pad_x)),
            float(min(height - 1, y + h + pad_y)),
        ]
        components.append((component, box))
    return components


def sam_obstacle_mask(sam: SAM, rgb: np.ndarray, component: np.ndarray, box: list[float], device: int | str) -> np.ndarray:
    result = sam.predict(rgb, bboxes=box, device=device, verbose=False)[0]
    if result.masks is None or not len(result.masks.data):
        return component
    height, width = component.shape
    best_mask = component
    best_score = -1.0
    component_area = max(float(component.sum()), 1.0)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    for raw_mask in result.masks.data.detach().cpu().numpy():
        mask = raw_mask > 0.5
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        coverage = float(np.logical_and(mask, component).sum()) / component_area
        area_ratio = float(mask.sum()) / box_area
        score = coverage - 0.03 * max(0.0, area_ratio - 2.0)
        if coverage >= 0.60 and 0.00006 <= float(mask.mean()) <= 0.15 and score > best_score:
            best_mask = mask
            best_score = score
    return best_mask


def mask_to_polygon(mask: np.ndarray, width: int, height: int) -> list[list[float]]:
    if mask.shape != (height, width):
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    binary = (mask > 0).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 45:
        return []
    epsilon = max(1.5, 0.0022 * cv2.arcLength(contour, True))
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in points]


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
        coordinates = []
        for x, y in item["points"]:
            coordinates.extend((np.clip(x / width, 0.0, 1.0), np.clip(y / height, 0.0, 1.0)))
        if len(coordinates) >= 6:
            lines.append(str(CLASSES[item["label"]]) + " " + " ".join(f"{float(value):.6f}" for value in coordinates))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_preview(image_rgb: np.ndarray, shapes: list[dict], output: Path) -> None:
    image = Image.fromarray(image_rgb).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    colors = {"floor": (35, 90, 245, 82), "obstacle": (245, 35, 25, 155)}
    for item in shapes:
        points = [(round(x), round(y)) for x, y in item["points"]]
        color = colors[item["label"]]
        draw.polygon(points, fill=color, outline=color[:3] + (255,), width=4)
        if points:
            draw.text(points[0], item["label"], fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    Image.alpha_composite(image, layer).convert("RGB").save(output, quality=88)


def annotate(records: list[dict]) -> list[dict]:
    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(BASE_MODEL))
    sam = SAM(str(SAM_MODEL))
    summary = []
    for index, record in enumerate(records, 1):
        image_path = PROJECT / "raw_images" / record["image"]
        image_bgr = read_image(image_path)
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        result = model.predict(rgb, conf=0.08, imgsz=640, device=device, verbose=False)[0]
        floor_mask = floor_mask_from_result(result, width, height)
        method = "best_combined_v2_floor"
        if floor_mask is None or not 0.08 <= float(floor_mask.mean()) <= 0.88:
            points = [[[0.50 * width, 0.95 * height], [0.50 * width, 0.80 * height], [0.50 * width, 0.66 * height], [0.04 * width, 0.55 * height], [0.96 * width, 0.55 * height], [0.50 * width, 0.08 * height]]]
            sam_result = sam.predict(rgb, points=points, labels=[[1, 1, 1, 0, 0, 0]], device=device, verbose=False)[0]
            if sam_result.masks is None:
                raise RuntimeError(f"No floor mask for {image_path.name}")
            candidates = sam_result.masks.data.detach().cpu().numpy() > 0.5
            floor_mask = max(candidates, key=lambda value: int(value.sum()))
            if floor_mask.shape != (height, width):
                floor_mask = cv2.resize(floor_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            method = "sam2_floor_fallback"

        shapes = []
        floor_polygon = mask_to_polygon(floor_mask, width, height)
        if not floor_polygon:
            raise RuntimeError(f"No usable floor polygon for {image_path.name}")
        shapes.append(shape("floor", floor_polygon, method))

        obstacle_boxes = []
        for component, box in color_components(image_bgr, floor_mask):
            obstacle_mask = sam_obstacle_mask(sam, rgb, component, box, device)
            polygon = mask_to_polygon(obstacle_mask, width, height)
            if polygon:
                shapes.append(shape("obstacle", polygon, "HSV color proposal + SAM2"))
                obstacle_boxes.append([round(value, 1) for value in box])

        labelme = {
            "version": "6.0.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        stem = image_path.stem
        (PROJECT / "labelme_json" / f"{stem}.json").write_text(json.dumps(labelme, ensure_ascii=False, indent=2), encoding="utf-8")
        write_yolo(PROJECT / "yolo_labels" / f"{stem}.txt", shapes, width, height)
        make_preview(rgb, shapes, PROJECT / "annotation_previews" / f"{stem}.jpg")
        updated = dict(record)
        updated.update(
            {
                "floor_count": 1,
                "obstacle_count": sum(item["label"] == "obstacle" for item in shapes),
                "obstacle_boxes": obstacle_boxes,
            }
        )
        summary.append(updated)
        print(f"[{index:03d}/100] {image_path.name}: floor=1 obstacles={updated['obstacle_count']}", flush=True)
    return summary


def copy_dataset_tree(source: Path, target: Path) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            for item in (source / kind / split).glob("*"):
                shutil.copy2(item, target / kind / split / item.name)


def build_merged_dataset(records: list[dict]) -> None:
    merged = PROJECT / "merged_dataset"
    copy_dataset_tree(BASE_DATASET, merged)

    old_val = set(range(16, 27)) | set(range(73, 78))
    for index in range(1, 81):
        stem = f"combined_{index:03d}"
        split = "val" if index in old_val else "train"
        shutil.copy2(COMBINED_ROOT / "raw_images" / f"{stem}.jpg", merged / "images" / split / f"{stem}.jpg")
        shutil.copy2(COMBINED_RESULTS / "new_yolo_labels" / f"{stem}.txt", merged / "labels" / split / f"{stem}.txt")

    new_val = {3, 8}
    offsets = ((10, 25), (35, 25), (60, 41))
    for start, count in offsets:
        new_val.update(start + offset for offset in range(2, count, 5))
    if len(new_val) != 20:
        raise RuntimeError(f"Expected 20 new validation samples, got {len(new_val)}")

    for index, record in enumerate(records, 1):
        stem = Path(record["image"]).stem
        split = "val" if index in new_val else "train"
        record["split"] = split
        shutil.copy2(PROJECT / "raw_images" / record["image"], merged / "images" / split / record["image"])
        shutil.copy2(PROJECT / "yolo_labels" / f"{stem}.txt", merged / "labels" / split / f"{stem}.txt")

    yaml_path = merged / "dataset.yaml"
    yaml_path.write_text(
        f"path: {merged}\ntrain: images/train\nval: images/val\nnames:\n  0: floor\n  1: obstacle\n",
        encoding="utf-8",
    )
    counts = {
        f"{kind}_{split}": len(list((merged / kind / split).glob("*")))
        for kind in ("images", "labels")
        for split in ("train", "val")
    }
    if counts != {"images_train": 304, "images_val": 76, "labels_train": 304, "labels_val": 76}:
        raise RuntimeError(f"Unexpected merged dataset counts: {counts}")


def make_contact_sheets() -> None:
    previews = sorted((PROJECT / "annotation_previews").glob("*.jpg"))
    for sheet_index in range(4):
        selected = previews[sheet_index * 25 : (sheet_index + 1) * 25]
        cell_width, cell_height = 250, 350
        sheet = np.full((5 * cell_height, 5 * cell_width, 3), 245, dtype=np.uint8)
        for index, path in enumerate(selected):
            image = read_image(path)
            scale = min((cell_width - 8) / image.shape[1], (cell_height - 28) / image.shape[0])
            resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            x = (index % 5) * cell_width + (cell_width - resized.shape[1]) // 2
            y = (index // 5) * cell_height + 23
            sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            cv2.putText(sheet, path.stem[-3:], ((index % 5) * cell_width + 5, (index // 5) * cell_height + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
        write_jpeg(PROJECT / f"annotation_contact_sheet_{sheet_index + 1}.jpg", sheet, quality=90)


def prepare() -> None:
    require_new_project()
    records = sample_media()
    records = annotate(records)
    build_merged_dataset(records)
    make_contact_sheets()
    (PROJECT / "new_batch_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PREPARE_DONE: 100 new samples; merged train=304 val=76", flush=True)


def reannotate() -> None:
    manifest_path = PROJECT / "new_batch_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("The prepared project manifest was not found")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = annotate(records)
    build_merged_dataset(records)
    make_contact_sheets()
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REANNOTATE_DONE: 100 labels refreshed", flush=True)


def train(epochs: int) -> None:
    yaml_path = PROJECT / "merged_dataset" / "dataset.yaml"
    if not yaml_path.exists():
        raise RuntimeError("Run the prepare stage first")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a long CPU training run")
    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=str(PROJECT / "runs"),
        name="colored_turn_v3",
        pretrained=True,
        patience=12,
        seed=257,
        deterministic=True,
        plots=True,
        cache=False,
        fliplr=0.5,
        degrees=5.0,
        translate=0.12,
        scale=0.35,
        perspective=0.0005,
        mosaic=0.65,
        amp=True,
    )


def evaluate_and_package() -> None:
    run = PROJECT / "runs" / "colored_turn_v3"
    best = run / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError("Training best.pt was not found")
    results = PROJECT / "ece257_colored_turn_v3_results"
    results.mkdir(parents=True, exist_ok=False)
    shutil.copy2(best, results / "best_colored_turn_v3.pt")
    shutil.copy2(PROJECT / "new_batch_manifest.json", results / "new_batch_manifest.json")
    shutil.copy2(Path(__file__), results / Path(__file__).name)
    for sheet in PROJECT.glob("annotation_contact_sheet_*.jpg"):
        shutil.copy2(sheet, results / sheet.name)
    shutil.copytree(PROJECT / "labelme_json", results / "labelme_json")
    shutil.copytree(PROJECT / "yolo_labels", results / "new_yolo_labels")

    training_results = results / "training_results"
    training_results.mkdir()
    for item in run.iterdir():
        if item.name != "weights" and item.is_file():
            shutil.copy2(item, training_results / item.name)

    model = YOLO(str(best))
    metrics = model.val(
        data=str(PROJECT / "merged_dataset" / "dataset.yaml"),
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project=str(PROJECT / "validation"),
        name="colored_turn_v3_val",
        plots=True,
    )
    metric_summary = {
        "mask_precision": float(metrics.seg.mp),
        "mask_recall": float(metrics.seg.mr),
        "mask_mAP50": float(metrics.seg.map50),
        "mask_mAP50_95": float(metrics.seg.map),
        "mask_mAP50_95_per_class": {
            "floor": float(metrics.seg.maps[0]),
            "obstacle": float(metrics.seg.maps[1]),
        },
    }
    (results / "validation_metrics.json").write_text(json.dumps(metric_summary, indent=2), encoding="utf-8")
    validation_dir = PROJECT / "validation" / "colored_turn_v3_val"
    if validation_dir.exists():
        shutil.copytree(validation_dir, results / "validation_results")

    demo = results / "demo_predictions"
    demo.mkdir()
    demo_sources = [
        PROJECT / "raw_images" / "colored_turn_001.jpg",
        PROJECT / "raw_images" / "colored_turn_003.jpg",
        PROJECT / "raw_images" / "colored_turn_034.jpg",
        PROJECT / "raw_images" / "colored_turn_073.jpg",
        PROJECT / "raw_images" / "colored_turn_095.jpg",
    ]
    for source in demo_sources:
        prediction = model.predict(str(source), imgsz=640, conf=0.25, device=0, verbose=False)[0]
        write_jpeg(demo / source.name, prediction.plot(), quality=92)

    digest = hashlib.sha256((results / "best_colored_turn_v3.pt").read_bytes()).hexdigest().upper()
    readme = f"""# ECE 257 colored obstacle + turn segmentation v3

- Dataset: 380 images total (304 training, 76 validation).
- New batch: 100 images (9 still photos and 91 video frames).
- Classes: `floor` and `obstacle`.
- Starting checkpoint: `best_combined_v2.pt`.
- Saved model: `best_colored_turn_v3.pt`.
- SHA256: `{digest}`.

Validation metrics are stored in `validation_metrics.json`. The 100 editable LabelMe polygon annotations are in `labelme_json`, and their YOLO segmentation labels are in `new_yolo_labels`.

These metrics describe this validation split only. Final vehicle performance still needs testing and fine-tuning with the onboard camera position, field of view, lighting, and motion.
"""
    (results / "README.md").write_text(readme, encoding="utf-8")
    shutil.make_archive(str(PROJECT / "ece257_colored_turn_v3_essential"), "zip", root_dir=results)
    print(json.dumps(metric_summary, indent=2), flush=True)
    print(f"MODEL_SHA256={digest}", flush=True)
    print(f"PACKAGE_DONE={results}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "reannotate", "train", "evaluate"))
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "reannotate":
        reannotate()
    elif args.stage == "train":
        train(args.epochs)
    else:
        evaluate_and_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
