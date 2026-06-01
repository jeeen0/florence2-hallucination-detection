"""BLIP caption backend for cross-model verification.

Default model: ``Salesforce/blip-image-captioning-large`` (~470M params, ~1.7 GB
fp16). The original BLIP-2 OPT-2.7B exceeds the 8 GB VRAM of an RTX 3070 when
co-loaded with Florence-2-large-ft, so we fall back to the lighter BLIP.
The class name is preserved for backward compatibility with the script."""
from __future__ import annotations

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class Blip2Captioner:
    def __init__(
        self,
        model_id: str = "Salesforce/blip-image-captioning-large",
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = (
            BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def caption(
        self,
        image: Image.Image,
        max_new_tokens: int = 50,
        num_beams: int = 3,
        prompt: str | None = None,
    ) -> str:
        if prompt is None:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.dtype)
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, self.dtype)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        if prompt is not None and text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].lstrip(": ").strip()
        return text
