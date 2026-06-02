"""Regenerate visualization overlays from existing grounding CSV without re-running OD.

Useful when only specific images are needed (e.g. for paper figures) and the
original viz dir was not transferred between machines.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image

from modules.extract import Mention
from modules.verify import Verdict
from modules.visualize import draw_verdicts


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounding-csv", type=Path, required=True)
    ap.add_argument("--caption-csv", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "coco_val")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", type=int, required=True,
                    help="Image IDs to regenerate (int form, e.g. 253433)")
    args = ap.parse_args()

    captions: dict[int, str] = {}
    with args.caption_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            captions[int(row["image_id"])] = row["caption"]

    by_image: dict[int, list[Verdict]] = defaultdict(list)
    with args.grounding_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = int(row["image_id"])
            if iid not in args.ids:
                continue
            bbox = None
            try:
                bbox = (
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                )
            except (TypeError, ValueError):
                pass
            mention = Mention(
                surface=row["surface"], canonical=row["canonical"], start=0, end=0
            )
            verdict = Verdict(
                mention=mention,
                supported=(row["status"] == "supported"),
                bbox=bbox,
            )
            by_image[iid].append(verdict)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for iid in args.ids:
        path = args.image_dir / f"{iid:012d}.jpg"
        if not path.exists():
            print(f"MISS: {path}")
            continue
        image = Image.open(path).convert("RGB")
        viz = draw_verdicts(image, by_image.get(iid, []), caption=captions.get(iid))
        out = args.out_dir / f"{iid:012d}_verified.jpg"
        viz.save(out, quality=92)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
