"""Check lighting quality of tea-tray capture shots.

Reads the image folder from .env (SRC_FOR_COLOR_HISTOGRAM_ANALYSIS).
A command-line argument overrides it:

    python check_lighting.py               # uses path from .env
    python check_lighting.py other/folder  # explicit override

Particles are masked out automatically, so this works on shots WITH tea
particles, not just empty-background frames.

Reports per image:
  1. White balance      - mean background R, G, B (should be nearly equal)
  2. Brightness level   - mean background gray (target ~230-250, <1% clipped)
  3. Uniformity         - background brightness variation, 8x8 grid (<10%)
  4. Histogram plot     - saved to <folder>/lighting_reports/

Ends with a summary table and a shot-to-shot drift check.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def analyze(path: Path, report_dir: Path, idx: int):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  SKIP (unreadable): {path.name}")
        return None
    b, g, r = cv2.split(img.astype(np.float64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Mask out particles (dark objects) so stats reflect the BACKGROUND only.
    _, bg_mask = cv2.threshold(gray.astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask = cv2.erode(bg_mask, np.ones((15, 15), np.uint8))
    bg = bg_mask > 0
    if bg.mean() < 0.3:  # Otsu likely failed; fall back to all pixels
        bg = np.ones_like(gray, bool)

    mr, mg, mb = r[bg].mean(), g[bg].mean(), b[bg].mean()
    cast = max(mr, mg, mb) / max(min(mr, mg, mb), 1)
    m = gray[bg].mean()
    clipped = (gray[bg] >= 254).mean() * 100

    h, w = gray.shape
    patches = []
    for i in range(8):
        for j in range(8):
            tile = gray[i * h // 8:(i + 1) * h // 8, j * w // 8:(j + 1) * w // 8]
            tmask = bg[i * h // 8:(i + 1) * h // 8, j * w // 8:(j + 1) * w // 8]
            patches.append(tile[tmask].mean() if tmask.mean() > 0.5 else np.nan)
    patches = list(np.where(np.isnan(patches), np.nanmean(patches), patches))
    variation = (max(patches) - min(patches)) / max(patches) * 100

    ok_cast = cast < 1.05
    ok_bright = 210 <= m <= 252 and clipped < 1
    ok_uniform = variation < 10

    print(f"\n=== {path.name} ===")
    print(f"  Mean R/G/B (bg) : {mr:.1f} / {mg:.1f} / {mb:.1f}")
    print(f"  Cast ratio      : {cast:.3f}  ({'OK' if ok_cast else 'COLOR CAST - fix white balance'})")
    print(f"  Mean brightness : {m:.1f}, clipped {clipped:.2f}%  "
          f"({'OK' if ok_bright else 'adjust exposure (target ~230-250, <1% clipped)'})")
    print(f"  Uniformity      : {variation:.1f}% variation  "
          f"({'OK' if ok_uniform else 'UNEVEN - rebalance/raise sources'})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ch, col in zip((r, g, b), ("red", "green", "blue")):
        ax[0].hist(ch[bg].ravel(), bins=256, range=(0, 255), color=col, alpha=0.5)
    ax[0].set_title("RGB histogram, background only")
    im = ax[1].imshow(np.array(patches).reshape(8, 8), cmap="viridis")
    ax[1].set_title("Background brightness map (8x8)")
    fig.colorbar(im, ax=ax[1])
    fig.suptitle(path.name)
    out = report_dir / (path.stem + "_lighting_report.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)

    is_passed = ok_cast and ok_bright and ok_uniform
    if is_passed:
        grade_name = report_dir.parent.name
        passed_dir = report_dir.parent.parent.parent / "ready" / grade_name
        passed_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(passed_dir / (f"{(idx+1):03d}" + path.suffix)), img)

    return {"name": path.name, "cast": cast, "brightness": m,
            "variation": variation, "pass": is_passed}


def main():
    load_dotenv(Path(__file__).parent / ".env")
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        src = os.getenv("SRC_FOR_COLOR_HISTOGRAM_ANALYSIS")
        if not src:
            sys.exit("SRC_FOR_COLOR_HISTOGRAM_ANALYSIS not set in .env and no folder given")
        target = Path(__file__).parent / src if not Path(src).is_absolute() else Path(src)

    if target.is_dir():
        images = sorted(p for p in target.rglob("*") if p.suffix.lower() in EXTS
                        and "lighting_reports" not in p.parts)
        if not images:
            sys.exit(f"No images found in {target}")
        report_dir = target / "lighting_reports"
    elif target.is_file():
        images = [target]
        report_dir = target.parent / "lighting_reports"
    else:
        sys.exit(f"Not found: {target}")

    print(f"Analyzing {len(images)} image(s) in {target}")
    report_dir.mkdir(exist_ok=True)
    results = [r for idx, p in enumerate(images) if (r := analyze(p, report_dir, idx))]

    if len(results) > 1:
        print("\n" + "=" * 66)
        print(f"{'image':<30}{'cast':>7}{'bright':>8}{'var%':>7}  result")
        print("-" * 66)
        for r in results:
            print(f"{r['name']:<30}{r['cast']:>7.3f}{r['brightness']:>8.1f}"
                  f"{r['variation']:>7.1f}  {'PASS' if r['pass'] else 'FAIL'}")
        n_pass = sum(r["pass"] for r in results)
        print(f"\n{n_pass}/{len(results)} images passed. Reports in: {report_dir}")

        brights = [r["brightness"] for r in results]
        drift = (max(brights) - min(brights)) / max(brights) * 100
        print(f"Brightness drift between shots: {drift:.1f}% "
              f"({'OK' if drift < 5 else 'settings not locked - check AE/WB'})")


if __name__ == "__main__":
    main()
