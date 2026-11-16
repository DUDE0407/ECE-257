# Intelligent Real-Time Error Detection and Correction System for Autonomous Vehicles

ECE 257 Team A3 project repository.

The project is being developed as two connected functional sections:

1. **Camera perception and route decision support** — segment the traversable floor, detect obstacles, estimate free driving space, generate a route centerline, measure route deviation, and participate in stopping decisions.
2. **Vehicle control and safety** — control steering and wheel speed, use ToF distance measurements, interpret route-deviation signals, and make the final stop or safety-override decision.

An offline video demonstration may be added later as a test and presentation format, but it is not currently treated as a separate core subsystem.

## Current progress

The camera-perception section has a working YOLO11 segmentation prototype. The current model recognizes the traversable floor and obstacles in the same frame and has been incrementally trained with additional colored obstacles and turn-focused footage.

- Dataset: 380 images
- Training split: 304 images
- Validation split: 76 images
- New increment: 9 still images and 91 sampled video frames
- Overall mask precision: 0.947
- Overall mask recall: 1.000
- Overall mask mAP50: 0.986
- Overall mask mAP50-95: 0.913
- Floor mask mAP50-95: 0.984
- Obstacle mask mAP50-95: 0.841
- Best checkpoint: `models/best_colored_turn_v3.pt`

The validation split is more difficult than the earlier v2 split because it includes the new hallway, turn, and colored-obstacle scenes. For a fair comparison, both checkpoints were evaluated on the same 76 validation images:

| Mask metric | Previous v2 model | Current v3 model |
|---|---:|---:|
| Overall mAP50-95 | 0.739 | 0.913 |
| Floor mAP50-95 | 0.940 | 0.984 |
| Obstacle mAP50-95 | 0.539 | 0.841 |

The detailed comparison is saved in `results/same_validation_comparison.json`.

These values describe the saved validation set, not guaranteed performance on every physical track or from the final onboard camera. The model will need another short fine-tuning and calibration round after the vehicle camera height, field of view, and mounting angle are fixed.

Route-centerline generation, route-deviation measurement, ToF fusion, vehicle control, and full-system integration are not implemented yet.

## Repository structure

```text
annotations/  LabelMe JSON polygons and matching YOLO segmentation labels
config/       Dataset configuration
examples/     Five representative predictions from the current model
models/       Current best trained checkpoint
notebooks/    Interactive photo-testing notebook
results/      Key metrics, comparisons, plots, and confusion matrix
src/          Data preparation, training, and inference code
```

The raw image/video dataset and merged image dataset are intentionally not included. Some source footage contains school interior scenes, and the full media dataset is maintained separately on local storage.

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

The default confidence threshold is 0.50 to reduce low-confidence false obstacles from reflections. It can be changed with `--conf`. Predictions are saved under `predictions/latest` by default. Class `0` is the traversable floor or track region, and class `1` is an obstacle.

## Training and annotations

The v3 incremental preparation and training workflow is saved in `src/colored_turn_v3_pipeline.py`. Its local source paths are project-specific and must be adjusted before running on another computer.

The 100 new editable LabelMe annotations are under `annotations/labelme_json`, with matching YOLO segmentation labels under `annotations/yolo_labels`. Images referenced by the annotations are not included in this repository.

## Next development step

The recommended first route-generation implementation is:

1. Subtract an expanded obstacle mask from the floor mask to create a safe drivable region.
2. Find the left and right boundaries of the connected drivable region at multiple look-ahead rows.
3. Smooth the boundary midpoints into an expected route centerline.
4. Calculate lateral deviation and heading deviation relative to the vehicle center.
5. Request a visual stop when no connected corridor remains wide enough for the vehicle and its safety margin; ToF sensing will confirm distance and retain final stopping authority.
