"""Verify Florence-2 caption mentions with an OPEN-VOCABULARY detector
(OWLv2 or Grounding DINO) instead of Florence-2 ``<OD>``.

Uses the SAME caption CSV and the SAME strict-vocab mention set as the
Florence-2 self-verification result, so the only variable is the verifier.
Writes a grounding CSV in the exact format consumed by ``run_step4.py``:
  image_id, surface, canonical, status, x1, y1, x2, y2

Run on a GPU (RTX 4090-class), e.g.:
  python code/run_cross_detector.py --detector owlv2 --n 500
  python code/run_cross_detector.py --detector gdino --n 500

Then score with the existing metric code:
  python code/run_step4.py \
    --grounding-csv outputs/cross/owlv2_grounding_500.csv \
    --gt-json data/coco_annotations/instances_val2017.json \
    --out-json outputs/metrics/verif_owlv2_500_metrics.json \
    --out-table outputs/metrics/verif_owlv2_500_result_table.csv \
    --label "OWLv2 verifier"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from detector_verifiers import build_detector
from modules.extract import extract_mentions
from modules.verify import verify_with_detections

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_caption_csv(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((int(row["image_id"]), row["caption"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", choices=["owlv2", "gdino"], required=True)
    ap.add_argument(
        "--caption-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "captions" / "large_strict_caption_500.csv",
        help="Florence-2 caption CSV (same source as the self-verification result)",
    )
    ap.add_argument("--n", type=int, default=500, help="First N images from the caption CSV")
    ap.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "coco_val")
    ap.add_argument("--vocab", choices=["strict", "aggressive"], default="strict")
    ap.add_argument("--threshold", type=float, default=0.10, help="OWLv2 score threshold")
    ap.add_argument("--box-threshold", type=float, default=0.30, help="Grounding DINO box threshold")
    ap.add_argument("--text-threshold", type=float, default=0.25, help="Grounding DINO text threshold")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = _load_caption_csv(args.caption_csv)[: args.n]
    out_path = args.out or (REPO_ROOT / "outputs" / "cross" / f"{args.detector}_grounding_{args.n}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[verifier={args.detector}] {len(rows)} images from {args.caption_csv.name}")
    detector = build_detector(args.detector)

    grounding_rows: list[tuple] = []
    n_unsupp = 0
    for i, (img_id, caption) in enumerate(rows, 1):
        mentions = extract_mentions(caption, vocab=args.vocab)
        if not mentions:
            continue
        image = Image.open(args.image_dir / f"{img_id:012d}.jpg").convert("RGB")
        queries = sorted({m.canonical for m in mentions})
        if args.detector == "owlv2":
            detections = detector.detect(image, queries, threshold=args.threshold)
        else:
            detections = detector.detect(
                image, queries, box_threshold=args.box_threshold, text_threshold=args.text_threshold
            )
        for v in verify_with_detections(mentions, detections):
            bb = v.bbox if v.bbox else (None, None, None, None)
            grounding_rows.append(
                (
                    img_id,
                    v.mention.surface,
                    v.mention.canonical,
                    "supported" if v.supported else "unsupported",
                    bb[0], bb[1], bb[2], bb[3],
                )
            )
            if not v.supported:
                n_unsupp += 1
        if i % 25 == 0 or i == len(rows):
            print(f"  [{i}/{len(rows)}] ok")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "surface", "canonical", "status", "x1", "y1", "x2", "y2"])
        for r in grounding_rows:
            w.writerow(r)
    print(f"Saved -> {out_path}")
    print(f"mentions: {len(grounding_rows)}  unsupported: {n_unsupp}")


if __name__ == "__main__":
    main()
