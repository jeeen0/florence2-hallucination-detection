"""Caption generation wrapper around Florence2Runner."""
from __future__ import annotations

from PIL import Image

from florence2 import Florence2Runner


def generate_caption(
    runner: Florence2Runner,
    image: Image.Image,
    detailed: bool = False,
) -> str:
    prompt = "<DETAILED_CAPTION>" if detailed else "<CAPTION>"
    return runner.run(prompt, image)[prompt].strip()
