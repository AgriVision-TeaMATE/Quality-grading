# Tea Grade Composition Prediction

Particle-level classification pipeline for early-stage tea grade composition prediction.
Photographs a spread tea sample after withering/rolling, segments every individual
particle, classifies each one independently, and derives the sample's grade
composition by counting predictions per grade (`count(grade) / total particles`) -
a **classify-then-count** design, not a whole-image regression.

Two classification arms are trained and compared on identical particles:

- **Arm A (handcrafted features):** geometric (mm), LAB colour, GLCM/LBP texture -> Random Forest / SVM / XGBoost
- **Arm B (CNN transfer learning):** ResNet-18 / MobileNetV3-Small trained directly on particle crops

Current status: 3 of 6 grades captured and trained (OPA, OP, BOP1 - 844 particles,
27 trays). Best result: ResNet-18 at 94.5% test accuracy / 0.943 macro-F1.

---

## Project structure

```
configs/config.yaml          # all paths, grades, calibration, model hyperparameters
dataset/
  ready/<grade>/*.jpg        # raw tray photos, one grade per folder (pure-grade captures)
  segmented/<grade>/*.png    # per-particle crops, output of segment_particles.py
  manifests/                 # full_manifest.csv, feature_matrix.csv (train/val/test splits)
  mixed_samples/             # mixed-composition sample photos + ground_truth.csv
results/
  models/                    # trained model files (.joblib, .pt)
  metrics/                   # JSON results per model + mixture evaluations
  figures/                   # segmentation checks, confusion matrices, charts
  logs/
src/
  segmentation/               segment_particles.py, visualize_segmentation.py
  traditional_cv/             feature_extraction.py, build_feature_matrix(_chunked).py, train_traditional_models.py
  deep_learning/               dataset.py, model.py, train_cnn.py
  evaluation/                  estimate_mixture.py, batch_evaluate_mixtures.py, generate_report.py
  utils/                       common.py (config/calibration helpers), build_manifests.py
  api/                         main.py, model_cache.py (FastAPI prediction service)
check_lighting.py            # pre-capture lighting QC script
```

---

## 0. Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (used by `check_lighting.py`):

```
SRC_FOR_COLOR_HISTOGRAM_ANALYSIS=path/to/your/capture/folder
```

Open `configs/config.yaml` and check:

- `classes.grades` - only grades with captured data should be **uncommented**. Adding a
  new grade folder does nothing until it's also added here.
- `traditional_cv.px_per_mm_by_grade` - which capture zoom (scale level) each grade uses.
  Currently: OPA/OP/BOP1/Fiber = level 1 (130% zoom, 20 px/mm), BOP/Dust = level 2
  (200% zoom, 31 px/mm).

---

## 1. Capture and lighting QC

Before every capture session, shoot a calibration frame (empty background / plain
tray under your lighting) and check it:

```bash
python check_lighting.py                      # uses folder from .env
python check_lighting.py path/to/folder        # or an explicit override
```

Checks colour cast, brightness, clipping %, and 8x8-grid illumination uniformity;
saves histogram/heatmap plots to `<folder>/lighting_reports/`. Only proceed with the
real capture session once this passes.

Locked capture settings used so far: ISO 25, shutter 1/300s, HDR mode, Cloudy white
balance, +1.7 EV, two 10W diffused sources at ~45 degrees, matte white background,
particles spread non-touching.

Save tray photos to `dataset/ready/<grade>/*.jpg` - one grade per folder, particles
from a single known grade per tray (pure-grade capture, no manual per-particle
annotation needed).

---

## 2. Segmentation

Segment every tray image into individual particle crops:

```bash
python src/segmentation/segment_particles.py \
  --input_dir dataset/ready/opa \
  --output_dir dataset/segmented/opa
# repeat per grade folder (op, bop1, ...)
```

QA-check segmentation on a sample image before running the full batch - always look
at this before trusting the output, bad segmentation silently corrupts everything
downstream:

```bash
python src/segmentation/visualize_segmentation.py \
  --image dataset/ready/opa/001.png \
  --output_dir results/figures
```

Saves an annotated image (green boxes + particle count) and a binary mask to
`results/figures/`.

---

## 3. Build manifests and feature matrix

```bash
python src/utils/build_manifests.py
```

Scans `dataset/segmented/<grade>/` for every grade listed in `config.yaml`, builds
`dataset/manifests/full_manifest.csv` with a stratified 70/15/15 train/val/test split.

Extract handcrafted features (Arm A) for every particle in the manifest:

```bash
python src/traditional_cv/build_feature_matrix.py
```

If your environment has a short per-command timeout, use the chunked version instead
(appends to the same output file across several calls):

```bash
python src/traditional_cv/build_feature_matrix_chunked.py --start 0 --end 220
python src/traditional_cv/build_feature_matrix_chunked.py --start 220 --end 440
python src/traditional_cv/build_feature_matrix_chunked.py --start 440 --end 660
python src/traditional_cv/build_feature_matrix_chunked.py --start 660 --end 844
```

Each particle's geometric features (area/perimeter/length/width) are converted to mm
using the `px_per_mm_by_grade` calibration for that particle's grade - not one global
value - so mixing capture zoom levels across grades doesn't corrupt size features.

Output: `dataset/manifests/feature_matrix.csv`.

---

## 4. Train models

**Arm A - traditional CV:**

```bash
python src/traditional_cv/train_traditional_models.py               # trains RF + SVM + XGBoost
python src/traditional_cv/train_traditional_models.py --model random_forest   # just one
```

Saves `results/models/traditional_<name>.joblib` (model + scaler + label encoder
bundled together) and `results/metrics/traditional_cv_results.json`.

**Arm B - CNN transfer learning:**

```bash
python src/deep_learning/train_cnn.py                     # uses config.yaml's backbone (default resnet18)
python src/deep_learning/train_cnn.py --backbone resnet18
python src/deep_learning/train_cnn.py --backbone mobilenet_v3_small
```

Applies online augmentation (horizontal/vertical flip, up to 180° rotation, mild
colour jitter) on the train split only. Saves
`results/models/cnn_<backbone>_best.pt` and `results/metrics/cnn_<backbone>_results.json`.

Current best results (3-class: OPA/OP/BOP1, 844 particles):

| model | test accuracy | macro-F1 |
|---|---|---|
| Random Forest | 93.7% | 0.926 |
| XGBoost | 92.9% | 0.916 |
| SVM | 91.3% | 0.893 |
| **ResNet-18** | **94.5%** | **0.943** |
| MobileNetV3-Small | 87.4% | 0.865 |

---

## 5. Generate the results report

```bash
python src/evaluation/generate_report.py
```

Aggregates all trained models' metrics into summary tables/figures under
`results/figures/` and `results/metrics/`.

---

## 6. Evaluate on a mixed sample

Photograph a mixed-composition sample (particles from multiple grades combined) and
save it to `dataset/mixed_samples/`. Since a mixed photo has one fixed zoom, tell the
script which scale level it was shot at - it can't be inferred per-particle before
classification the way it can for pure-grade trays:

```bash
python src/evaluation/estimate_mixture.py \
  --image dataset/mixed_samples/mix1.jpg \
  --method traditional --model_name random_forest \
  --scale_level 1 \
  --ground_truth '{"opa":0.4,"op":0.3,"bop1":0.3}'

# or the CNN arm:
python src/evaluation/estimate_mixture.py \
  --image dataset/mixed_samples/mix1.jpg \
  --method cnn --backbone resnet18 \
  --scale_level 1
```

Prints predicted composition, and per-grade error/MAE/RMSE if `--ground_truth` is
given. Saves a JSON result to `results/metrics/`.

For multiple mixed samples at once, build `dataset/mixed_samples/ground_truth.csv`
(columns: `image_filename, <grade columns 0-1>, scale_level`) and run:

```bash
python src/evaluation/batch_evaluate_mixtures.py
```

Reports mean MAE/RMSE per method across all samples - the number that actually
compares the two arms on composition accuracy, not just per-particle accuracy.

---

## 7. Serve predictions over an API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Interactive docs at `http://127.0.0.1:8000/docs`. Models are loaded once and cached
in memory (see `src/api/model_cache.py`), not reloaded per request.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@dataset/ready/opa/001.png" \
  -F "method=traditional" \
  -F "model_name=random_forest" \
  -F "scale_level=1"
```

Returns predicted grade composition, per-particle label + bounding box (for
traceability), and an annotated, segmented image as a base64 PNG
(`segmented_image_base64`) ready to drop into an `<img src>`.

---

## Adding a new grade later

1. Capture pure-grade trays following the lighting QC + acquisition protocol above.
2. Uncomment the grade in `configs/config.yaml` -> `classes.grades`, and confirm its
   entry in `px_per_mm_by_grade` matches the zoom it was actually shot at.
3. `segment_particles.py` on the new grade's tray folder.
4. Re-run `build_manifests.py` (rebuilds the full stratified split including the new
   grade) and `build_feature_matrix.py` (or the chunked version).
5. Re-run `train_traditional_models.py` and `train_cnn.py` - both retrain from
   scratch on the updated manifest/feature matrix.
6. Re-run `generate_report.py` to refresh aggregated metrics/figures.

Remaining grades to capture: **BOP1** additional trays, **Dust**, **Fiber** (currently
only 1 image), and **BOP** (distinct from BOP1 - not yet captured).
