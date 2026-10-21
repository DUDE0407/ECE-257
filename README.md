# Intelligent Real-Time Error Detection and Correction System for Autonomous Vehicles

ECE 257 Team A3 project repository.

The complete project is an autonomous-vehicle error detection and correction system. Its planned modules include visual perception, route and stopping decisions, vehicle control and error correction, and an offline video-based virtual prototype. The offline virtual prototype replaces the original digital-twin concept and is one subsystem of the project, not the name of the complete project.

## Current progress

The visual perception module currently provides a working baseline using YOLO11 segmentation. It segments the traversable floor and obstacles in the same image and has demonstrated recognition of multiple obstacles at different distances.

- Dataset: 280 images
- Training split: 224 images
- Validation split: 56 images
- Overall mask mAP50: 0.980
- Overall mask mAP50-95: 0.912
- Floor mask mAP50-95: 0.969
- Obstacle mask mAP50-95: 0.854
- Local inference time: approximately 21.6 ms per image on an RTX 5070 GPU
- Best checkpoint: `models/best_combined_v2.pt`

These metrics describe the current validation split. They do not represent complete-vehicle performance or onboard Raspberry Pi performance.

Route planning, obstacle avoidance, stopping decisions, vehicle control, and complete-system integration are not yet implemented. A moving-obstacle platform is currently being planned and fabricated for a later dynamic-obstacle scenario.

## Repository structure

```text
config/       Dataset configuration
examples/     Five representative predictions from the current model
models/       Current trained checkpoint
notebooks/    Training and testing notebooks
results/      Training curves, metrics, and confusion matrices
src/          Data preparation, training, and inference code
```

The raw image and video dataset is not included in this repository.

## Installation

Create a Python environment and install the dependencies:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Run inference

```powershell
& ".\.venv\Scripts\python.exe" src\predict.py --source "path\to\image_or_video"
```

Predictions are saved under `predictions/latest` by default. Class `0` is the traversable floor or track region, and class `1` is an obstacle.

## Training

Place the private dataset under `dataset/images/train`, `dataset/images/val`, `dataset/labels/train`, and `dataset/labels/val`. The current combined workflow is in `src/combined_v2_pipeline.py` and `notebooks/ECE257_combined_v2_training.ipynb`.

## Next steps

- Add new images and videos after changing the obstacle colors.
- Expand coverage across turns, distance, position, and lighting conditions.
- Generate a safe driving region from the floor and obstacle masks.
- Develop route planning and stopping decisions.
- Integrate and test the perception and control modules on the physical vehicle.
