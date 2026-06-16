"""Qualitative demo: verify OUT-OF-COCO-80 object mentions with an open-vocab
detector. Florence-2 ``<OD>`` can only emit COCO-80 labels, so objects like
``tripod``, ``plate``, or ``guitar`` cannot even be represented; an open-vocab
verifier can localize (or reject) them directly.

Run on a GPU, e.g.:
  python code/demo_openvocab_oov.py \
    --image data/coco_val/000000XXXXXX.jpg \
    --words "tripod,plate,guitar,wine bottle,candle" \
    --out paper/figures/oov_example1.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from detector_verifiers import OwlV2Detector


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--words", required=True, help="comma-separated candidate object words (any vocabulary)")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    queries = [w.strip() for w in args.words.split(",") if w.strip()]
    image = Image.open(args.image).convert("RGB")
    det = OwlV2Detector()
    best = det.detect_best(image, queries, threshold=args.threshold)  # one box per word

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image); ax.axis("off")
    for q, (score, (x1, y1, x2, y2)) in best.items():
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#2b8c8c", linewidth=2.4))
        ax.text(x1, max(0, y1 - 4), f"{q} {score:.2f}", color="white", fontsize=12,
                bbox=dict(facecolor="#2b8c8c", edgecolor="none", pad=1.5))
    supported = [q for q in queries if q in best]
    unsupported = [q for q in queries if q not in best]
    title = "supported: " + (", ".join(supported) or "—") + "\nunsupported: " + (", ".join(unsupported) or "—")
    ax.set_title(title, fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out, dpi=170, bbox_inches="tight")
    print(f"Saved -> {args.out}")
    print(f"supported (out-of-COCO ok): {supported}")
    print(f"unsupported (flagged): {unsupported}")


if __name__ == "__main__":
    main()
