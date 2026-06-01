"""Visualize caption-grounding verification results."""
from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


SUPPORTED_COLOR = (40, 180, 60)     # green
UNSUPPORTED_COLOR = (220, 40, 40)   # red
TEXT_BG = (0, 0, 0)


def _font(size: int = 16) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_verdicts(
    image: Image.Image,
    verdicts: Iterable,
    caption: str | None = None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(14)

    supported_labels: list[str] = []
    unsupported_labels: list[str] = []

    for v in verdicts:
        label = v.mention.canonical
        if v.supported and v.bbox is not None:
            x1, y1, x2, y2 = v.bbox
            draw.rectangle([x1, y1, x2, y2], outline=SUPPORTED_COLOR, width=3)
            draw.text((x1 + 4, max(0, y1 - 18)), label, fill=SUPPORTED_COLOR, font=font)
            supported_labels.append(label)
        else:
            unsupported_labels.append(label)

    margin = 6
    line_y = margin
    if caption:
        text = f"caption: {caption}"
        draw.text((margin, line_y), text, fill=(255, 255, 255), font=small,
                  stroke_width=2, stroke_fill=TEXT_BG)
        line_y += 20

    if supported_labels:
        draw.text(
            (margin, line_y),
            "supported: " + ", ".join(supported_labels),
            fill=SUPPORTED_COLOR, font=small,
            stroke_width=2, stroke_fill=TEXT_BG,
        )
        line_y += 20
    if unsupported_labels:
        draw.text(
            (margin, line_y),
            "unsupported: " + ", ".join(unsupported_labels),
            fill=UNSUPPORTED_COLOR, font=small,
            stroke_width=2, stroke_fill=TEXT_BG,
        )

    return canvas
