"""
Tea Grade Prediction API
===========================
Serves the trained particle classifiers (traditional-CV and CNN) over HTTP so
a mobile/web client can upload a sample tray photo and get back:
    - predicted grade composition (classify-then-count)
    - per-particle predictions (label + bounding box), for traceability
    - an annotated, segmented image with each particle boxed and labelled,
      returned inline as a base64 PNG so the client can render it directly

Run locally:
    pip install fastapi uvicorn python-multipart
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive API docs (Swagger UI),
or POST an image to /predict directly.
"""
import os
import sys
import base64
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# --- make the project's src/ modules importable, same convention the CLI scripts use ---
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for sub in ("utils", "segmentation", "traditional_cv", "deep_learning"):
    sys.path.append(os.path.join(SRC_DIR, sub))

from common import load_config, get_px_per_mm  # noqa: E402
from segment_particles import segment_image  # noqa: E402
from feature_extraction import extract_all_features  # noqa: E402
import model_cache  # noqa: E402

app = FastAPI(
    title="Tea Grade Prediction API",
    description="Particle-level tea grade classification and composition estimation.",
    version="1.0.0",
)

# Allow a mobile app / web frontend on a different origin to call this during development.
# Tighten this to your actual client origin(s) before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CFG = load_config()
GRADES = CFG["classes"]["grades"]

# Green box/label style, consistent with the report/presentation figures.
BOX_COLOR = (76, 140, 76)     # BGR - matches the project's established green palette
TEXT_COLOR = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@app.on_event("startup")
def startup():
    # Warm the cache so the FIRST real request isn't slow. Safe to skip/fail silently
    # if a given model hasn't been trained yet - /predict will raise a clear 404 instead.
    model_cache.warm_up(CFG)


class ParticlePrediction(BaseModel):
    label: str
    bbox: List[int]          # [x, y, w, h] in pixels, original image coordinates
    area_mm2: Optional[float] = None
    area_px: Optional[float] = None


class PredictResponse(BaseModel):
    method: str
    weighting: str
    px_per_mm_used: float
    num_particles_detected: int
    num_particles_classified: int
    predicted_proportions: dict
    particles: List[ParticlePrediction]
    segmented_image_base64: Optional[str] = None  # data URI, ready for <img src=...>


def _resolve_px_per_mm(scale_level: Optional[int], px_per_mm: Optional[float]) -> float:
    """Mirrors estimate_mixture.py's CLI logic, but returns a value rather than
    mutating shared config - safe under concurrent requests."""
    cfg_cv = CFG["traditional_cv"]
    if px_per_mm is not None:
        return px_per_mm
    if scale_level is not None:
        return 20.0 if scale_level == 1 else 31.0
    return cfg_cv.get("px_per_mm_default", cfg_cv.get("px_per_mm", 20.0))


def _classify_traditional(crops, model_name: str, px_per_mm: float):
    bundle = model_cache.get_traditional_bundle(CFG, model_name)
    model, scaler, label_encoder, feature_cols = (
        bundle["model"], bundle["scaler"], bundle["label_encoder"], bundle["feature_cols"]
    )
    cfg_cv = dict(CFG["traditional_cv"])
    cfg_cv["px_per_mm"] = px_per_mm

    tmp_dir = "/tmp/api_crops"
    os.makedirs(tmp_dir, exist_ok=True)

    predictions, areas = [], []
    for i, crop in enumerate(crops):
        if crop.size == 0:
            predictions.append(None)
            areas.append(0.0)
            continue
        tmp_path = os.path.join(tmp_dir, f"crop_{i}.png")
        cv2.imwrite(tmp_path, crop)

        feats = extract_all_features(tmp_path, cfg_cv)
        if feats is None:
            predictions.append(None)
            areas.append(0.0)
            continue

        x = np.array([[feats[c] for c in feature_cols]])
        x_scaled = scaler.transform(x)
        pred_idx = model.predict(x_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        predictions.append(pred_label)
        areas.append(feats.get("area_mm2", feats.get("area_px", 0.0)))

    return predictions, areas


def _classify_cnn(crops, backbone: str):
    import torch
    from PIL import Image

    model, classes, transform, device = model_cache.get_cnn_model(CFG, backbone)

    predictions, areas = [], []
    with torch.no_grad():
        for crop in crops:
            if crop.size == 0:
                predictions.append(None)
                areas.append(0.0)
                continue
            area_px = crop.shape[0] * crop.shape[1]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            output = model(tensor)
            pred_idx = output.argmax(dim=1).item()
            predictions.append(classes[pred_idx])
            areas.append(float(area_px))

    return predictions, areas


def _compute_proportions(predictions, areas, weighting: str):
    proportions = {g: 0.0 for g in GRADES}
    valid = [(p, a) for p, a in zip(predictions, areas) if p is not None]
    if not valid:
        return proportions

    if weighting == "area":
        total = sum(a for _, a in valid)
        if total == 0:
            return proportions
        for pred, area in valid:
            if pred in proportions:
                proportions[pred] += area / total
    else:
        total = len(valid)
        for pred, _ in valid:
            if pred in proportions:
                proportions[pred] += 1.0 / total

    return proportions


def _draw_annotated(image, contours, predictions):
    vis = image.copy()
    for c, label in zip(contours, predictions):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x + w, y + h), BOX_COLOR, 2)
        text = label if label else "?"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), BOX_COLOR, -1)
        cv2.putText(vis, text, (x + 2, y - 4), FONT, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(vis, f"Detected: {len(contours)} particles", (10, 30),
                FONT, 1.0, (255, 0, 0), 2)
    return vis


def _encode_image_base64(image) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode annotated image")
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """What's actually trained and available on disk right now."""
    from common import resolve_path
    models_dir = resolve_path(CFG["paths"]["results_models"])
    if not os.path.isdir(models_dir):
        return {"traditional": [], "cnn": []}
    files = os.listdir(models_dir)
    traditional = [f.replace("traditional_", "").replace(".joblib", "")
                   for f in files if f.startswith("traditional_") and f.endswith(".joblib")]
    cnn = [f.replace("cnn_", "").replace("_best.pt", "")
           for f in files if f.startswith("cnn_") and f.endswith("_best.pt")]
    return {"traditional": traditional, "cnn": cnn, "grades": GRADES}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Sample tray image (mixed or pure-grade)."),
    method: str = Form("cnn", description="'traditional' or 'cnn'"),
    model_name: str = Form("random_forest", description="Used when method='traditional': random_forest | svm | xgboost"),
    backbone: str = Form("resnet18", description="Used when method='cnn': resnet18 | mobilenet_v3_small"),
    scale_level: Optional[int] = Form(None, description="1 (130% zoom, ~20 px/mm) or 2 (200% zoom, ~31 px/mm)"),
    px_per_mm: Optional[float] = Form(None, description="Overrides scale_level if given directly."),
    weighting: Optional[str] = Form(None, description="'count' or 'area'; defaults to config value."),
    include_image: bool = Form(True, description="Include the annotated segmented image as base64 in the response."),
):
    if method not in ("traditional", "cnn"):
        raise HTTPException(status_code=400, detail="method must be 'traditional' or 'cnn'")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    resolved_px_per_mm = _resolve_px_per_mm(scale_level, px_per_mm)

    crops, contours, _ = segment_image(image, CFG["segmentation"])
    if not crops:
        raise HTTPException(status_code=422, detail="No particles detected - check image/segmentation settings.")

    try:
        if method == "traditional":
            predictions, areas = _classify_traditional(crops, model_name, resolved_px_per_mm)
            method_label = f"traditional_{model_name}"
        else:
            predictions, areas = _classify_cnn(crops, backbone)
            method_label = f"cnn_{backbone}"
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    weighting = weighting or CFG["mixture_estimation"]["weighting"]
    proportions = _compute_proportions(predictions, areas, weighting)

    particles = []
    for c, label, area in zip(contours, predictions, areas):
        x, y, w, h = cv2.boundingRect(c)
        entry = {"label": label or "unclassified", "bbox": [int(x), int(y), int(w), int(h)]}
        if method == "traditional":
            entry["area_mm2"] = round(float(area), 3)
        else:
            entry["area_px"] = float(area)
        particles.append(entry)

    segmented_image_b64 = None
    if include_image:
        annotated = _draw_annotated(image, contours, predictions)
        segmented_image_b64 = _encode_image_base64(annotated)

    return PredictResponse(
        method=method_label,
        weighting=weighting,
        px_per_mm_used=resolved_px_per_mm,
        num_particles_detected=len(crops),
        num_particles_classified=sum(1 for p in predictions if p is not None),
        predicted_proportions={g: round(v, 4) for g, v in proportions.items()},
        particles=particles,
        segmented_image_base64=segmented_image_b64,
    )
