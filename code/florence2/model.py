"""Thin wrapper around microsoft/Florence-2-{base,large}-ft for prompt-based inference."""
from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class Florence2Runner:
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base-ft",
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

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    @torch.inference_mode()
    def run(
        self,
        task_prompt: str,
        image: Image.Image,
        text_input: str | None = None,
        max_new_tokens: int = 1024,
        num_beams: int = 3,
    ) -> dict[str, Any]:
        prompt = task_prompt if text_input is None else task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.dtype
        )
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        return self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
