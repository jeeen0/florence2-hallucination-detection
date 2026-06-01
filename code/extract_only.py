"""Re-extract object mentions from an existing caption CSV (no model call).

Use when you want to re-run extraction with a different vocab without paying
the cost of re-generating captions.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from modules.extract import extract_mentions, extract_unique_canonicals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caption-csv", type=Path, required=True)
    ap.add_argument("--out-caption", type=Path, required=True)
    ap.add_argument("--out-mentions", type=Path, required=True)
    ap.add_argument("--vocab", choices=["strict", "aggressive"], default="strict")
    args = ap.parse_args()

    caption_rows: list[tuple] = []
    mention_rows: list[tuple] = []

    with args.caption_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = int(row["image_id"])
            prompt = row["prompt"]
            caption = row["caption"]
            canonicals = extract_unique_canonicals(caption, vocab=args.vocab)
            caption_rows.append((image_id, prompt, caption, ",".join(canonicals)))
            for m in extract_mentions(caption, vocab=args.vocab):
                mention_rows.append((image_id, m.surface, m.canonical, m.start, m.end))

    args.out_caption.parent.mkdir(parents=True, exist_ok=True)
    args.out_mentions.parent.mkdir(parents=True, exist_ok=True)

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

    print(f"Saved -> {args.out_caption}")
    print(f"Saved -> {args.out_mentions}")
    print(f"images={len(caption_rows)}  total_mentions={len(mention_rows)}")


if __name__ == "__main__":
    main()
