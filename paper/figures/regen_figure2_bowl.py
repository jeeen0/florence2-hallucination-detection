"""Regenerate Figure 2 — the Joint Failure (FN) qualitative example.

Selected case: image_id 509131.
    caption  : "A bowl of oranges, apples and bananas on a table."
    COCO GT  : apple, banana, orange, potted plant     (no 'bowl')
    System   : bowl, orange, apple, banana — all marked supported
    Audit    : 'bowl' verdict = not_visible (real joint failure)

The fruits sit on an open wire fruit stand, not in a bowl. Both the captioner
and the OD inferred 'bowl' from the cluster of fruit — a textbook joint
failure where the verifier fails because it shares the captioner's bias.

Output: paper/figures/000000509131_verified.jpg  (matches style of the
existing knife / NO TRUCKS verified figures used as Fig 3 / Fig 4).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
IMAGE_ID = 509131
SRC_IMAGE = REPO / "data" / "coco_val" / f"{IMAGE_ID:012d}.jpg"
GROUNDING = REPO / "outputs" / "grounding" / "large_strict_2000_grounding.csv"
CAPTIONS  = REPO / "outputs" / "captions"  / "large_strict_caption_2000.csv"
OUT_FILE  = REPO / "paper" / "figures" / f"{IMAGE_ID:012d}_verified.jpg"

SUPPORTED_COLOR  = (40, 180, 60)
UNSUPPORTED_COLOR = (220, 40, 40)
TEXT_BG          = (0, 0, 0)


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    img = Image.open(SRC_IMAGE).convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    big = _font(16)
    small = _font(14)

    g = pd.read_csv(GROUNDING)
    rows = g[g["image_id"] == IMAGE_ID]
    caption = pd.read_csv(CAPTIONS).set_index("image_id").loc[IMAGE_ID, "caption"]

    supported_labels = []
    unsupported_labels = []

    for _, r in rows.iterrows():
        label = r["canonical"]
        if r["status"] == "supported":
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            draw.rectangle([x1, y1, x2, y2],
                           outline=SUPPORTED_COLOR, width=3)
            draw.text((x1 + 4, max(0, y1 - 18)), label,
                      fill=SUPPORTED_COLOR, font=big)
            supported_labels.append(label)
        else:
            unsupported_labels.append(label)

    margin, y = 6, 6
    draw.text((margin, y), f"caption: {caption}",
              fill=(255, 255, 255), font=small,
              stroke_width=2, stroke_fill=TEXT_BG)
    y += 20

    if supported_labels:
        draw.text((margin, y),
                  "supported: " + ", ".join(supported_labels),
                  fill=SUPPORTED_COLOR, font=small,
                  stroke_width=2, stroke_fill=TEXT_BG)
        y += 20
    if unsupported_labels:
        draw.text((margin, y),
                  "unsupported: " + ", ".join(unsupported_labels),
                  fill=UNSUPPORTED_COLOR, font=small,
                  stroke_width=2, stroke_fill=TEXT_BG)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_FILE, quality=92)
    print(f"saved: {OUT_FILE}  ({OUT_FILE.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
