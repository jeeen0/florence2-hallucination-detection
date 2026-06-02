"""Cross-model verification with LLaVA-1.5-7B captions + Florence-2 OD.

Same comparison as run_cross.py but with a modern LVLM captioner. Requires
a GPU with ~14 GB VRAM (RTX 4090 or better).

Usage:
  python code/run_cross_llava.py --n 500
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from florence2 import Florence2Runner
from llava_runner import LlavaCaptioner
from modules.extract import extract_mentions
from modules.verify import detect_objects, verify_with_detections


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_caption_csv(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append((int(row["image_id"]), row["caption"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--florence-caption-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "captions" / "large_strict_caption_500.csv",
        help="Existing Florence-2 captions CSV; used as the 'self' source and to fix image order",
    )
    ap.add_argument("--n", type=int, default=500, help="First N images from the caption CSV")
    ap.add_argument(
        "--image-dir", type=Path, default=REPO_ROOT / "data" / "coco_val"
    )
    ap.add_argument("--llava-model", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--florence-model", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--vocab", choices=["strict", "aggressive"], default="strict")
    ap.add_argument(
        "--out-llava-caption",
        type=Path,
        default=REPO_ROOT / "outputs" / "cross" / "llava_caption.csv",
    )
    ap.add_argument(
        "--out-self-grounding",
        type=Path,
        default=REPO_ROOT / "outputs" / "cross" / "llava_self_grounding.csv",
    )
    ap.add_argument(
        "--out-cross-grounding",
        type=Path,
        default=REPO_ROOT / "outputs" / "cross" / "llava_cross_grounding.csv",
    )
    args = ap.parse_args()

    florence_rows = _load_caption_csv(args.florence_caption_csv)[: args.n]
    print(f"[llava-cross] running on {len(florence_rows)} images from {args.florence_caption_csv}")

    # 1. LLaVA captioning
    print(f"[llava-cross] loading {args.llava_model}")
    llava = LlavaCaptioner(model_id=args.llava_model)
    llava_caps: dict[int, str] = {}
    for i, (img_id, _) in enumerate(florence_rows, 1):
        path = args.image_dir / f"{img_id:012d}.jpg"
        image = Image.open(path).convert("RGB")
        llava_caps[img_id] = llava.caption(image)
        if i % 25 == 0 or i == len(florence_rows):
            print(f"  [{i}/{len(florence_rows)}] llava ok")
    del llava
    torch.cuda.empty_cache()

    args.out_llava_caption.parent.mkdir(parents=True, exist_ok=True)
    with args.out_llava_caption.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "caption"])
        for img_id, _ in florence_rows:
            w.writerow([img_id, llava_caps[img_id]])
    print(f"Saved -> {args.out_llava_caption}")

    # 2. Florence-2 OD on the same images
    print(f"[llava-cross] loading {args.florence_model}")
    runner = Florence2Runner(model_id=args.florence_model)

    self_rows: list[tuple] = []
    cross_rows: list[tuple] = []

    for i, (img_id, florence_cap) in enumerate(florence_rows, 1):
        path = args.image_dir / f"{img_id:012d}.jpg"
        image = Image.open(path).convert("RGB")
        detections = detect_objects(runner, image, vocab=args.vocab)

        for src_text, target in ((florence_cap, self_rows), (llava_caps[img_id], cross_rows)):
            mentions = extract_mentions(src_text, vocab=args.vocab)
            verdicts = verify_with_detections(mentions, detections)
            for v in verdicts:
                bb = v.bbox if v.bbox else (None, None, None, None)
                target.append(
                    (
                        img_id,
                        v.mention.surface,
                        v.mention.canonical,
                        "supported" if v.supported else "unsupported",
                        bb[0],
                        bb[1],
                        bb[2],
                        bb[3],
                    )
                )

        if i % 25 == 0 or i == len(florence_rows):
            print(f"  [{i}/{len(florence_rows)}] od ok")

    args.out_self_grounding.parent.mkdir(parents=True, exist_ok=True)
    args.out_cross_grounding.parent.mkdir(parents=True, exist_ok=True)

    for path, rows in (
        (args.out_self_grounding, self_rows),
        (args.out_cross_grounding, cross_rows),
    ):
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
