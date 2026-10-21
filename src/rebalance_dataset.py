#!/usr/bin/env python3
"""Repair an existing split so both classes appear in validation."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def move_pair(dataset: Path, name: str, source: str, destination: str) -> None:
    shutil.move(dataset / "images" / source / name, dataset / "images" / destination / name)
    label = f"{Path(name).stem}.txt"
    shutil.move(dataset / "labels" / source / label, dataset / "labels" / destination / label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    dataset = args.project / "dataset"
    manifest_path = args.project / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    obstacle_to_val = sorted(name for name in manifest["train"] if name.startswith("still_obstacle_"))[:2]
    floor_to_train = sorted(name for name in manifest["val"] if not name.startswith("still_obstacle_"))[:2]
    if len(obstacle_to_val) != 2 or len(floor_to_train) != 2:
        raise RuntimeError("Could not find the required samples to rebalance")

    for name in obstacle_to_val:
        move_pair(dataset, name, "train", "val")
        manifest["train"].remove(name)
        manifest["val"].append(name)
    for name in floor_to_train:
        move_pair(dataset, name, "val", "train")
        manifest["val"].remove(name)
        manifest["train"].append(name)
    manifest["train"].sort()
    manifest["val"].sort()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("moved obstacle->val", obstacle_to_val)
    print("moved floor->train", floor_to_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
