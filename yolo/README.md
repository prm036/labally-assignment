# YOLO detector: `chemical_structure` + `reaction_scheme`

Trains a YOLO model to find chemical structures and reaction schemes on handwritten
chemistry pages, using the boxes annotated earlier.

## Files
- `annotations_normalized.py` — `ANN = {filename: [(label, x0, y0, x1, y1), ...]}`, normalized 0–1 corners.
- `prepare_dataset.py` — converts those to YOLO format and writes `data.yaml`.
- `train.py` — fine-tunes a pretrained YOLO model.
- `predict.py` — runs the trained model on new pages.

## Setup
```bash
pip install ultralytics            # pulls in torch; use a CUDA build for GPU
```

## 1) Build the dataset
Point `--images` at the folder holding the original page images.
```bash
python prepare_dataset.py --images /path/to/images --out dataset --val-frac 0.2
```
Produces:
```
dataset/
  images/{train,val}/...
  labels/{train,val}/...     # "cls cx cy w h", cls 0=chemical_structure 1=reaction_scheme
  data.yaml
```

## 2) Train
```bash
python train.py --data dataset/data.yaml --epochs 200 --imgsz 1024 --batch 8
# CPU-only: add --device cpu (slow); pick --model yolo11s.pt or yolo11m.pt for more capacity
```
Best weights land in `runs/chem/yolo11n_chem/weights/best.pt`. Inspect the predicted
val images and `results.png` in that folder.

## 3) Predict on a new document
```bash
python predict.py --weights runs/chem/yolo11n_chem/weights/best.pt \
                  --source /path/to/new_page.png --conf 0.25
```
Writes annotated images plus `predictions/detections.json` (pixel boxes + confidences).

## Reality check on dataset size
This set is ~43 images / 382 boxes — small for detection. The scripts lean on
pretrained weights + strong augmentation, but expect modest, noisy metrics. To improve:
- **Label more pages** (a few hundred is where this gets reliable). Tools: Label Studio,
  Roboflow, CVAT — all export YOLO format.
- **Use 5-fold CV** for a trustworthy mAP (a single 9-image val split swings a lot).
- The classes overlap by design (structures sit inside schemes); that's fine for
  detection, but keep your `--conf`/`--iou` thresholds in mind when consuming results.
- If you later want structures *grouped by* the scheme they belong to, that's a
  post-processing step (containment / IoU between the two classes), not a model change.
```
