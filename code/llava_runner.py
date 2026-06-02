"""LLaVA-1.5-7B caption backend for cross-model verification.

Requires ~14 GB VRAM in fp16 (works on RTX 4090; not on 3070).
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


class LlavaCaptioner:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
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
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def caption(self, image: Image.Image, max_new_tokens: int = 80) -> str:
        prompt = "USER: <image>\nDescribe this image briefly.\nASSISTANT:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device, self.dtype
        )
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1].strip()
        return text
