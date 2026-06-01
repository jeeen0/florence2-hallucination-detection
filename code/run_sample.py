"""Step 1 smoke test.

Loads Florence-2-{base,large}-ft and runs three prompts on one image:
  <CAPTION>, <DETAILED_CAPTION>, <OD>

Outputs:
  outputs/captions/sample_caption.csv      # caption rows
  outputs/captions/sample_od.csv           # one row per detection
  outputs/captions/sample_raw.json         # full raw outputs for traceability
  outputs/visualizations/sample_od.jpg     # bboxes overlaid on image

Usage:
  python code/run_sample.py --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from florence2 import Florence2Runner


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CAPTION_CSV = REPO_ROOT / "outputs" / "captions" / "sample_caption.csv"
OUT_OD_CSV = REPO_ROOT / "outputs" / "captions" / "sample_od.csv"
OUT_RAW_JSON = REPO_ROOT / "outputs" / "captions" / "sample_raw.json"
OUT_OD_VIZ = REPO_ROOT / "outputs" / "visualizations" / "sample_od.jpg"


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_od(image: Image.Image, od: dict) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    bboxes = od.get("bboxes", [])
    labels = od.get("labels", [])
    for (x1, y1, x2, y2), label in zip(bboxes, labels):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 4, max(0, y1 - 18)), str(label), fill=(255, 0, 0), font=font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument("--model", default="microsoft/Florence-2-base-ft")
    ap.add_argument("--image-id", default="sample")
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    image = Image.open(image_path).convert("RGB")

    runner = Florence2Runner(model_id=args.model)

    caption = runner.run("<CAPTION>", image)["<CAPTION>"]
    detailed = runner.run("<DETAILED_CAPTION>", image)["<DETAILED_CAPTION>"]
    od = runner.run("<OD>", image)["<OD>"]

    OUT_CAPTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_OD_VIZ.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CAPTION_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "prompt", "output_text"])
        w.writerow([args.image_id, "<CAPTION>", caption])
        w.writerow([args.image_id, "<DETAILED_CAPTION>", detailed])

    with OUT_OD_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label", "x1", "y1", "x2", "y2"])
        for (x1, y1, x2, y2), label in zip(od.get("bboxes", []), od.get("labels", [])):
            w.writerow([args.image_id, label, x1, y1, x2, y2])

    with OUT_RAW_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "image_id": args.image_id,
                "image_path": str(image_path),
                "model": args.model,
                "caption": caption,
                "detailed_caption": detailed,
                "od": od,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    draw_od(image, od).save(OUT_OD_VIZ, quality=92)

    print(f"[CAPTION]          {caption}")
    print(f"[DETAILED_CAPTION] {detailed}")
    print(f"[OD]               {len(od.get('bboxes', []))} detections")
    print(f"Saved -> {OUT_CAPTION_CSV.relative_to(REPO_ROOT)}")
    print(f"Saved -> {OUT_OD_CSV.relative_to(REPO_ROOT)}")
    print(f"Saved -> {OUT_RAW_JSON.relative_to(REPO_ROOT)}")
    print(f"Saved -> {OUT_OD_VIZ.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
