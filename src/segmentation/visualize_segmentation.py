"""
Segmentation QA Visualizer
===========================
Draws detected contours/bounding boxes on the original image so you can
visually verify segmentation quality BEFORE running it across the whole
dataset. Always check this on a handful of images first — bad segmentation
silently corrupts every downstream step.

Usage:
    python visualize_segmentation.py --image path/to/tray_image.jpg --output_dir results/figures
"""
import os
import sys
import argparse
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import load_config, ensure_dir
from segment_particles import segment_image


def draw_detections(image, contours):
    vis = image.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.drawContours(vis, [c], -1, (0, 0, 255), 1)
    cv2.putText(vis, f"Detected: {len(contours)} particles", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="Visualize particle segmentation for QA.")
    parser.add_argument("--image", type=str, required=True, help="Path to a single tray image.")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_seg = cfg["segmentation"]

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    crops, contours, mask = segment_image(image, cfg_seg)
    vis = draw_detections(image, contours)

    ensure_dir(args.output_dir)
    base = os.path.splitext(os.path.basename(args.image))[0]
    out_vis = os.path.join(args.output_dir, f"{base}_segmentation_check.png")
    out_mask = os.path.join(args.output_dir, f"{base}_mask.png")

    cv2.imwrite(out_vis, vis)
    cv2.imwrite(out_mask, mask)

    print(f"Detected {len(contours)} particles.")
    print(f"Saved annotated image to: {out_vis}")
    print(f"Saved binary mask to:     {out_mask}")
    print("\nReview these images. If particles are merged/missed, adjust")
    print("segmentation params in configs/config.yaml (min/max area, threshold method)")
    print("or try --watershed mode in segment_particles.py for touching particles.")


if __name__ == "__main__":
    main()
