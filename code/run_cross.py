"""Phase 4: cross-model verification.

For a fixed image set, we generate captions with TWO captioners (Florence-2 and
BLIP-2), extract mentions for each, then verify both against the SAME
Florence-2 <OD> result. The point is to test whether using a different
captioner produces a different hallucination signal — i.e. whether the apparent
agreement in self-verification is just shared blind spots.

Pipeline:
1. Read existing Florence-2 caption CSV.
2. Load BLIP-2 to generate fresh captions for the same image IDs.
3. (Florence-2 OD outputs are reused from the existing strict 500 grounding
   results; raw OD is re-computed here per image.)
4. Build mentions for both caption sources with strict vocab.
5. Score each against Florence-2 OD: produce grounding CSVs and metrics.

Usage:
  python code/run_cross.py --n 200
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

from blip2_runner import Blip2Captioner
from florence2 import Florence2Runner
from modules.extract import extract_mentions, extract_unique_canonicals
from modules.verify import detect_objects, verify_with_detections


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_caption_csv(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((int(row["image_id"]), row["caption"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--florence-caption-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "large_strict_caption_500.csv")
    ap.add_argument("--n", type=int, default=200, help="First N images from the caption CSV")
    ap.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "coco_val")
    ap.add_argument("--blip2-model", default="Salesforce/blip-image-captioning-large")
    ap.add_argument("--florence-model", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--vocab", choices=["strict", "aggressive"], default="strict")
    ap.add_argument("--out-blip2-caption", type=Path,
                    default=REPO_ROOT / "outputs" / "cross" / "blip2_caption.csv")
    ap.add_argument("--out-self-grounding", type=Path,
                    default=REPO_ROOT / "outputs" / "cross" / "self_grounding.csv")
    ap.add_argument("--out-cross-grounding", type=Path,
                    default=REPO_ROOT / "outputs" / "cross" / "cross_grounding.csv")
    args = ap.parse_args()

    florence_rows = _load_caption_csv(args.florence_caption_csv)[: args.n]
    image_ids = [iid for iid, _ in florence_rows]
    print(f"[cross] running on {len(image_ids)} images from {args.florence_caption_csv}")

    # BLIP-2 captioning
    print(f"[cross] loading {args.blip2_model}")
    blip2 = Blip2Captioner(model_id=args.blip2_model)
    blip2_captions: dict[int, str] = {}
    for i, (img_id, _) in enumerate(florence_rows, 1):
        path = args.image_dir / f"{img_id:012d}.jpg"
        image = Image.open(path).convert("RGB")
        cap = blip2.caption(image)
        blip2_captions[img_id] = cap
        if i % 25 == 0 or i == len(florence_rows):
            print(f"  [{i}/{len(florence_rows)}] blip2 done")
    del blip2  # free VRAM
    import torch
    torch.cuda.empty_cache()

    args.out_blip2_caption.parent.mkdir(parents=True, exist_ok=True)
    with args.out_blip2_caption.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "caption"])
        for iid in image_ids:
            w.writerow([iid, blip2_captions[iid]])
    print(f"Saved -> {args.out_blip2_caption}")

    # Florence-2 OD
    print(f"[cross] loading {args.florence_model}")
    runner = Florence2Runner(model_id=args.florence_model)

    self_rows: list[tuple] = []
    cross_rows: list[tuple] = []

    for i, (img_id, florence_cap) in enumerate(florence_rows, 1):
        path = args.image_dir / f"{img_id:012d}.jpg"
        image = Image.open(path).convert("RGB")
        detections = detect_objects(runner, image, vocab=args.vocab)

        # self (Florence caption -> Florence OD)
        self_mentions = extract_mentions(florence_cap, vocab=args.vocab)
        self_verdicts = verify_with_detections(self_mentions, detections)
        for v in self_verdicts:
            bbox = v.bbox if v.bbox else (None, None, None, None)
            self_rows.append((
                img_id, v.mention.surface, v.mention.canonical,
                "supported" if v.supported else "unsupported",
                bbox[0], bbox[1], bbox[2], bbox[3],
            ))

        # cross (BLIP-2 caption -> Florence OD)
        cross_mentions = extract_mentions(blip2_captions[img_id], vocab=args.vocab)
        cross_verdicts = verify_with_detections(cross_mentions, detections)
        for v in cross_verdicts:
            bbox = v.bbox if v.bbox else (None, None, None, None)
            cross_rows.append((
                img_id, v.mention.surface, v.mention.canonical,
                "supported" if v.supported else "unsupported",
                bbox[0], bbox[1], bbox[2], bbox[3],
            ))

        if i % 25 == 0 or i == len(florence_rows):
            print(f"  [{i}/{len(florence_rows)}] od done")

    args.out_self_grounding.parent.mkdir(parents=True, exist_ok=True)
    args.out_cross_grounding.parent.mkdir(parents=True, exist_ok=True)

    for path, rows in [(args.out_self_grounding, self_rows), (args.out_cross_grounding, cross_rows)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "surface", "canonical", "status", "x1", "y1", "x2", "y2"])
            for row in rows:
                w.writerow(row)
        print(f"Saved -> {path}")

    print(f"\nself mentions: {len(self_rows)}  unsup={sum(1 for r in self_rows if r[3]=='unsupported')}")
    print(f"cross mentions: {len(cross_rows)}  unsup={sum(1 for r in cross_rows if r[3]=='unsupported')}")


if __name__ == "__main__":
    main()
