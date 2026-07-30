"""
Particle Segmentation
======================
Takes full tray/sample images (particles scattered on a contrasting background)
and outputs cropped images of each individual particle.

Usage:
    python segment_particles.py --input_dir data/raw/BOP --output_dir data/segmented/BOP

If your dataset is ALREADY pre-cropped to individual particles, you can skip
this script entirely and point downstream scripts directly at data/raw/<grade>/.

Assumptions:
    - Particles are reasonably separated (not heavily overlapping) on a
      background that contrasts with tea particles (e.g. white/black tray).
    - Lighting is roughly uniform across the image (controlled capture setup
      per the research plan's "Image Acquisition" step).

If particles in your images are heavily touching/overlapping, contour-based
segmentation will under-segment (merge particles). In that case consider
watershed segmentation (see segment_particles_watershed() below) instead.
"""
import os
import sys
import argparse
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, ensure_dir


def preprocess_for_segmentation(image, blur_kernel=5):
    """Convert to grayscale and blur to reduce noise before thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return blurred


def threshold_image(blurred, method="otsu"):
    """
    Binarize the image so PARTICLES are foreground (255) and background is 0.
    cv2.findContours treats white pixels as foreground, so polarity matters:
    if we get this backwards, findContours returns one giant contour around
    the background (with particles as unreachable "holes") instead of one
    contour per particle.

    We assume particles occupy a minority of the image area (typical for
    tray photos with particles scattered on a background). Otsu/adaptive
    thresholding doesn't know which side is "the particles", so we check
    which binary class is smaller and flip if needed.
    """
    if method == "otsu":
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 2
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    # If white (255) pixels are the majority, particles are almost certainly
    # the minority/dark class -> invert so particles become foreground.
    white_fraction = np.count_nonzero(thresh == 255) / thresh.size
    if white_fraction > 0.5:
        thresh = cv2.bitwise_not(thresh)

    return thresh


def clean_mask(thresh, morph_kernel=3):
    """Morphological open/close to remove speckle noise and fill small holes."""
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def find_particle_contours(mask, min_area=50, max_area=50000):
    """Find contours and filter by area to remove noise specks and merged blobs."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            valid.append(c)
    return valid


def crop_particle(image, contour, padding=4):
    """Crop the bounding box of a particle from the original image, with padding."""
    x, y, w, h = cv2.boundingRect(contour)
    H, W = image.shape[:2]
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, W)
    y1 = min(y + h + padding, H)
    return image[y0:y1, x0:x1]


def segment_image(image, cfg_seg):
    """Full segmentation pipeline for a single image. Returns list of (crop, contour)."""
    blurred = preprocess_for_segmentation(image, cfg_seg.get("blur_kernel", 5))
    thresh = threshold_image(blurred, cfg_seg.get("threshold_method", "otsu"))
    mask = clean_mask(thresh, cfg_seg.get("morph_kernel", 3))
    contours = find_particle_contours(
        mask,
        min_area=cfg_seg.get("min_particle_area", 50),
        max_area=cfg_seg.get("max_particle_area", 50000),
    )
    crops = [crop_particle(image, c, cfg_seg.get("padding", 4)) for c in contours]
    return crops, contours, mask


def segment_particles_watershed(image, cfg_seg):
    """
    Alternative segmentation for TOUCHING/OVERLAPPING particles.
    Uses distance transform + watershed to split merged blobs.
    Swap this in for segment_image() if simple contour detection under-segments.
    """
    blurred = preprocess_for_segmentation(image, cfg_seg.get("blur_kernel", 5))
    thresh = threshold_image(blurred, cfg_seg.get("threshold_method", "otsu"))
    mask = clean_mask(thresh, cfg_seg.get("morph_kernel", 3))

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0

    color_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)

    crops = []
    for label in range(2, num_labels + 2):
        particle_mask = np.uint8(markers == label) * 255
        area = cv2.countNonZero(particle_mask)
        if not (cfg_seg.get("min_particle_area", 50) <= area <= cfg_seg.get("max_particle_area", 50000)):
            continue
        contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            crops.append(crop_particle(image, contours[0], cfg_seg.get("padding", 4)))
    return crops


def process_directory(input_dir, output_dir, cfg_seg, use_watershed=False):
    """Run segmentation over every image in input_dir, saving particle crops to output_dir."""
    ensure_dir(output_dir)
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]

    if not image_files:
        print(f"  [WARN] No images found in {input_dir}")
        return 0

    total_particles = 0
    for fname in tqdm(image_files, desc=f"Segmenting {os.path.basename(input_dir)}"):
        path = os.path.join(input_dir, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  [WARN] Could not read {path}, skipping.")
            continue

        if use_watershed:
            crops = segment_particles_watershed(image, cfg_seg)
        else:
            crops, _, _ = segment_image(image, cfg_seg)

        base_name = os.path.splitext(fname)[0]
        for i, crop in enumerate(crops):
            if crop.size == 0:
                continue
            out_path = os.path.join(output_dir, f"{base_name}_particle{i:04d}.png")
            cv2.imwrite(out_path, crop)
        total_particles += len(crops)

    return total_particles


def main():
    parser = argparse.ArgumentParser(description="Segment tea particles from tray images.")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory of raw tray/sample images for ONE grade.")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to save cropped particle images.")
    parser.add_argument("--watershed", action="store_true",
                         help="Use watershed segmentation for touching/overlapping particles.")
    parser.add_argument("--config", type=str, default=None,
                         help="Path to config.yaml (default: configs/config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_seg = cfg["segmentation"]

    n = process_directory(args.input_dir, args.output_dir, cfg_seg, use_watershed=args.watershed)
    print(f"\nDone. Extracted {n} particles from {args.input_dir} -> {args.output_dir}")


if __name__ == "__main__":
    main()
