"""
Handcrafted Feature Extraction (Approach A: Traditional CV)
=============================================================
Extracts geometric, shape, texture, and color features from a single
tea particle image. Used to build the feature matrix for Random Forest /
SVM / XGBoost classifiers.

Features extracted:
    Geometric : area, perimeter, aspect_ratio, solidity, extent, circularity
    Shape     : 7 Hu moments (scale/rotation/translation invariant shape descriptors)
    Texture   : GLCM (contrast, homogeneity, energy, correlation) + LBP histogram
    Color     : mean/std per channel in HSV (or RGB)
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def get_main_contour(binary_mask):
    """Return the largest contour in a binary mask (assumed to be the particle)."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def segment_single_particle(image, threshold_method="otsu"):
    """
    Produce a clean binary mask for a single-particle crop (already isolated
    from the background by the segmentation stage, but may still have a
    contrasting background within the crop). Ensures the particle (assumed
    minority pixel class) ends up as foreground (255), matching the
    convention used in segment_particles.py.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold_method == "otsu":
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 21, 2)

    white_fraction = np.count_nonzero(mask == 255) / mask.size
    if white_fraction > 0.5:
        mask = cv2.bitwise_not(mask)

    return mask


def extract_geometric_features(contour, px_per_mm=None):
    """Area, perimeter, aspect ratio, solidity, extent, circularity.

    IMPORTANT: pass the contour from the ORIGINAL (un-resized) crop, not one
    measured after resizing to a fixed image_size — resizing to a fixed size
    changes the effective px/mm scale per particle (crops aren't all the same
    original size), which destroys absolute size as a usable feature. If
    px_per_mm is given, area/perimeter/length/width are converted to mm units
    (mm^2, mm) so they're comparable across particles and capture sessions;
    aspect_ratio/solidity/extent/circularity are unitless ratios either way.
    """
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0.0
    extent = float(area_px) / (w * h) if (w * h) > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area_px) / hull_area if hull_area > 0 else 0.0

    circularity = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else 0.0

    feats = {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,
        "circularity": circularity,
    }

    if px_per_mm:
        feats["area_mm2"] = area_px / (px_per_mm ** 2)
        feats["perimeter_mm"] = perimeter_px / px_per_mm
        feats["length_mm"] = max(w, h) / px_per_mm
        feats["width_mm"] = min(w, h) / px_per_mm
    else:
        # No calibration available - fall back to raw pixel units (not
        # comparable across sessions/scales, but keeps the pipeline running).
        feats["area_px"] = area_px
        feats["perimeter_px"] = perimeter_px
        feats["length_px"] = max(w, h)
        feats["width_px"] = min(w, h)

    return feats


def extract_hu_moments(contour):
    """7 Hu moments — invariant shape descriptors, good for irregular particle shapes."""
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale transform (standard practice — raw Hu moments span huge ranges)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return {f"hu_{i}": hu_log[i] for i in range(7)}


def extract_texture_features(gray_crop, distances, angles, lbp_points, lbp_radius):
    """GLCM texture properties + LBP histogram."""
    # GLCM expects uint8
    img = gray_crop.astype(np.uint8)
    glcm = graycomatrix(img, distances=distances, angles=angles,
                         levels=256, symmetric=True, normed=True)

    features = {}
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop)
        features[f"glcm_{prop}_mean"] = float(np.mean(vals))
        features[f"glcm_{prop}_std"] = float(np.std(vals))

    lbp = local_binary_pattern(img, P=lbp_points, R=lbp_radius, method="uniform")
    n_bins = lbp_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for i, v in enumerate(hist):
        features[f"lbp_bin_{i}"] = float(v)

    return features


def extract_color_features(image, mask, color_space="lab"):
    """Mean and std of each color channel, computed only over particle pixels (using mask)."""
    if color_space == "lab":
        # LAB per the research plan: L (lightness) separates dark/light grades,
        # a/b (green-red, blue-yellow) capture the brown/olive tone differences
        # between grades independent of lighting brightness.
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        channel_names = ["L", "A", "B"]
    elif color_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channel_names = ["H", "S", "V"]
    else:
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        channel_names = ["R", "G", "B"]

    features = {}
    binary_mask = mask > 0
    for i, name in enumerate(channel_names):
        channel = converted[:, :, i]
        pixels = channel[binary_mask] if binary_mask.sum() > 0 else channel.flatten()
        features[f"color_{name}_mean"] = float(np.mean(pixels))
        features[f"color_{name}_std"] = float(np.std(pixels))
    return features


def extract_all_features(image_path, cfg_cv):
    """
    Full feature extraction pipeline for one particle image.
    Returns a flat dict of feature_name -> value, or None if the image
    couldn't be processed (e.g. empty mask).
    """
    original = cv2.imread(image_path)
    if original is None:
        return None

    # Geometric features come from the ORIGINAL crop, before resizing, so
    # absolute size (converted to mm via px_per_mm) survives. Resizing crops
    # of different original dimensions to one fixed size would otherwise
    # apply a different effective scale to each particle.
    orig_mask = segment_single_particle(original)
    orig_contour = get_main_contour(orig_mask)
    if orig_contour is None or cv2.contourArea(orig_contour) < 1:
        return None

    features = {}
    features.update(extract_geometric_features(orig_contour, px_per_mm=cfg_cv.get("px_per_mm")))
    features.update(extract_hu_moments(orig_contour))

    # Texture/color are computed on the resized crop for consistent feature
    # dimensionality regardless of original crop size.
    target_size = tuple(cfg_cv["image_size"])
    image = cv2.resize(original, target_size, interpolation=cv2.INTER_AREA)
    mask = segment_single_particle(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features.update(extract_texture_features(
        gray,
        distances=cfg_cv["glcm_distances"],
        angles=cfg_cv["glcm_angles"],
        lbp_points=cfg_cv["lbp_points"],
        lbp_radius=cfg_cv["lbp_radius"],
    ))
    features.update(extract_color_features(image, mask, cfg_cv.get("color_space", "lab")))

    return features
