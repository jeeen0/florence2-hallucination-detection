"""Step 2: Florence-2 caption generation + COCO-80 object-mention extraction.

Pipeline:
  COCO val2017 sample → Florence-2 <CAPTION> → COCO80+synonym extraction
  → CSVs (one row per image; one row per mention)

Usage:
  python code/run_step2.py --n 50
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from data_utils import collect_val2017_images
from florence2 import Florence2Runner
from modules.extract import extract_mentions, extract_unique_canonicals


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default="microsoft/Florence-2-base-ft")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", choices=["caption", "detailed"], default="caption")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "coco_val")
    ap.add_argument("--gt-json", type=Path,
                    default=REPO_ROOT / "data" / "coco_annotations" / "instances_val2017.json",
                    help="If present, sample image IDs from this file instead of CANDIDATE_IDS")
    ap.add_argument("--out-caption", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "base_caption_50.csv")
    ap.add_argument("--out-mentions", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "base_extracted_objects.csv")
    args = ap.parse_args()

    print(f"[step2] collecting up to {args.n} COCO val2017 images -> {args.data_dir}")
    images = collect_val2017_images(args.n, args.data_dir, seed=args.seed,
                                    gt_path=args.gt_json if args.gt_json.exists() else None)
    print(f"[step2] {len(images)} images available")
    if not images:
        raise SystemExit("No images downloaded; check network or CANDIDATE_IDS")

    prompt_token = "<CAPTION>" if args.prompt == "caption" else "<DETAILED_CAPTION>"

    print(f"[step2] loading {args.model}")
    runner = Florence2Runner(model_id=args.model)

    args.out_caption.parent.mkdir(parents=True, exist_ok=True)
    args.out_mentions.parent.mkdir(parents=True, exist_ok=True)

    caption_rows: list[tuple[int, str, str, str]] = []
    mention_rows: list[tuple[int, str, str, int, int]] = []

    for idx, (img_id, path) in enumerate(images, 1):
        image = Image.open(path).convert("RGB")
        caption = runner.run(prompt_token, image)[prompt_token].strip()
        canonicals = extract_unique_canonicals(caption)
        caption_rows.append((img_id, prompt_token, caption, ",".join(canonicals)))
        for m in extract_mentions(caption):
            mention_rows.append((img_id, m.surface, m.canonical, m.start, m.end))
        print(f"  [{idx:3d}/{len(images)}] id={img_id:012d}  -> {canonicals}")

    with args.out_caption.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "prompt", "caption", "canonical_objects"])
        for row in caption_rows:
            w.writerow(row)

    with args.out_mentions.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "surface", "canonical", "start", "end"])
        for row in mention_rows:
            w.writerow(row)

    print(f"\nSaved -> {args.out_caption.relative_to(REPO_ROOT)}")
    print(f"Saved -> {args.out_mentions.relative_to(REPO_ROOT)}")
    print(f"images={len(caption_rows)}  total_mentions={len(mention_rows)}")


if __name__ == "__main__":
    main()
