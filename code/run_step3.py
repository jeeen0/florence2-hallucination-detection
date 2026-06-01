"""Step 3: verify each caption mention against Florence-2 visual outputs.

Default strategy: one <OD> call per image, then match canonical labels.
Optional: --strategy grounding uses <CAPTION_TO_PHRASE_GROUNDING> per mention.

Inputs:
  outputs/captions/base_caption_50.csv
  outputs/captions/base_extracted_objects.csv

Outputs:
  outputs/grounding/base_grounding_results.csv
  outputs/visualizations/base/<image_id>_verified.jpg

Usage:
  python code/run_step3.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

from florence2 import Florence2Runner
from modules.extract import Mention
from modules.verify import (
    detect_objects,
    verify_with_detections,
    verify_with_phrase_grounding,
)
from modules.visualize import draw_verdicts


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_captions(path: Path) -> dict[int, str]:
    captions: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            captions[int(row["image_id"])] = row["caption"]
    return captions


def _load_mentions(path: Path) -> dict[int, list[Mention]]:
    grouped: dict[int, list[Mention]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[int(row["image_id"])].append(
                Mention(
                    surface=row["surface"],
                    canonical=row["canonical"],
                    start=int(row["start"]),
                    end=int(row["end"]),
                )
            )
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Florence-2-base-ft")
    ap.add_argument("--strategy", choices=["od", "grounding"], default="od")
    ap.add_argument("--vocab", choices=["strict", "aggressive"], default="strict",
                    help="Synonym mapping for OD label canonicalization")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "coco_val")
    ap.add_argument("--caption-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "base_caption_50.csv")
    ap.add_argument("--mentions-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "base_extracted_objects.csv")
    ap.add_argument("--out-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "grounding" / "base_grounding_results.csv")
    ap.add_argument("--out-viz-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "visualizations" / "base")
    args = ap.parse_args()

    captions = _load_captions(args.caption_csv)
    mentions_by_image = _load_mentions(args.mentions_csv)

    image_ids = sorted(captions.keys())
    print(f"[step3] verifying {len(image_ids)} images, strategy={args.strategy}")

    print(f"[step3] loading {args.model}")
    runner = Florence2Runner(model_id=args.model)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_viz_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple] = []

    for idx, img_id in enumerate(image_ids, 1):
        path = args.data_dir / f"{img_id:012d}.jpg"
        if not path.exists():
            print(f"  [{idx:3d}/{len(image_ids)}] id={img_id:012d}  MISSING image")
            continue
        image = Image.open(path).convert("RGB")
        mentions = mentions_by_image.get(img_id, [])

        if args.strategy == "od":
            detections = detect_objects(runner, image, vocab=args.vocab)
            verdicts = verify_with_detections(mentions, detections)
        else:
            verdicts = verify_with_phrase_grounding(runner, image, mentions)

        for v in verdicts:
            bbox = v.bbox if v.bbox else (None, None, None, None)
            rows.append((
                img_id,
                v.mention.surface,
                v.mention.canonical,
                "supported" if v.supported else "unsupported",
                bbox[0], bbox[1], bbox[2], bbox[3],
            ))

        viz = draw_verdicts(image, verdicts, caption=captions.get(img_id))
        viz.save(args.out_viz_dir / f"{img_id:012d}_verified.jpg", quality=92)

        sup = sum(1 for v in verdicts if v.supported)
        un = len(verdicts) - sup
        print(f"  [{idx:3d}/{len(image_ids)}] id={img_id:012d}  supported={sup}  unsupported={un}")

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "surface", "canonical", "status", "x1", "y1", "x2", "y2"])
        for row in rows:
            w.writerow(row)

    total_sup = sum(1 for r in rows if r[3] == "supported")
    total_un = sum(1 for r in rows if r[3] == "unsupported")
    print(f"\nSaved -> {args.out_csv}")
    print(f"Saved -> {args.out_viz_dir}/ ({len(image_ids)} images)")
    print(f"total_mentions={len(rows)}  supported={total_sup}  unsupported={total_un}")


if __name__ == "__main__":
    main()
