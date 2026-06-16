"""Open-vocabulary detector back-ends used as caption-mention verifiers.

Each detector exposes ``detect(image, queries, ...) -> list[Detection]`` where
``queries`` are the candidate object labels (the mention canonicals for that
image). A ``Detection`` is emitted for every query the detector localizes above
threshold, with ``canonical`` set to the matched query so the existing
``modules.verify.verify_with_detections`` and ``run_step4`` metric code can be
reused unchanged. This isolates the *verifier* as the only variable when
comparing against Florence-2 ``<OD>``.

Runs on a GPU (RTX 4090-class). Requires ``transformers>=4.40`` (OWLv2 and
Grounding DINO are both included in 4.49.0).
"""
from __future__ import annotations

import torch
from PIL import Image

from modules.verify import Detection
from modules.extract import normalize


def _device_dtype():
    # float32 on both detectors: small models, negligible speed cost on a 4090,
    # and it avoids fp16 input/weight dtype-mismatch issues (esp. Grounding DINO).
    return ("cuda" if torch.cuda.is_available() else "cpu"), torch.float32


class OwlV2Detector:
    """OWLv2 open-vocabulary detector (google/owlv2-base-patch16-ensemble)."""

    def __init__(self, model_id: str = "google/owlv2-base-patch16-ensemble"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device, self.dtype = _device_dtype()
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = (
            Owlv2ForObjectDetection.from_pretrained(model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.name = "owlv2"

    @torch.no_grad()
    def detect(self, image: Image.Image, queries: list[str], threshold: float = 0.10) -> list[Detection]:
        if not queries:
            return []
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target = torch.tensor([image.size[::-1]], device=self.device)  # (h, w)
        res = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target
        )[0]
        dets: list[Detection] = []
        for score, label_idx, box in zip(res["scores"], res["labels"], res["boxes"]):
            q = queries[int(label_idx)]
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            dets.append(Detection(label=q, canonical=q, bbox=(x1, y1, x2, y2)))
        return dets

    @torch.no_grad()
    def detect_best(self, image: Image.Image, queries: list[str], threshold: float = 0.20):
        """Return {query: (score, bbox)} keeping only the highest-scoring box per
        query. Used for clean single-box-per-object visualizations."""
        if not queries:
            return {}
        inputs = self.processor(text=[queries], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target = torch.tensor([image.size[::-1]], device=self.device)
        res = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target
        )[0]
        best: dict[str, tuple] = {}
        for score, label_idx, box in zip(res["scores"], res["labels"], res["boxes"]):
            q = queries[int(label_idx)]
            s = float(score)
            if q not in best or s > best[q][0]:
                best[q] = (s, tuple(float(v) for v in box.tolist()))
        return best


class GroundingDinoDetector:
    """Grounding DINO open-vocabulary detector (IDEA-Research/grounding-dino-base)."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-base"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device, self.dtype = _device_dtype()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id, torch_dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.name = "gdino"

    @staticmethod
    def _match_query(returned_label: str, queries: list[str]) -> str | None:
        r = normalize(returned_label)
        if not r:
            return None
        # exact, then containment (longest query wins) so "dining table" beats "table"
        for q in sorted(queries, key=len, reverse=True):
            nq = normalize(q)
            if r == nq or nq in r or r in nq:
                return q
        return None

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        queries: list[str],
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        if not queries:
            return []
        text = " . ".join(normalize(q) for q in queries) + " ."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target = [image.size[::-1]]  # (h, w)
        res = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target,
        )[0]
        labels = res.get("text_labels", res.get("labels", []))
        dets: list[Detection] = []
        for score, label, box in zip(res["scores"], labels, res["boxes"]):
            canon = self._match_query(str(label), queries)
            if canon is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            dets.append(Detection(label=str(label), canonical=canon, bbox=(x1, y1, x2, y2)))
        return dets


def build_detector(name: str):
    if name == "owlv2":
        return OwlV2Detector()
    if name == "gdino":
        return GroundingDinoDetector()
    raise ValueError(f"unknown detector '{name}' (expected 'owlv2' or 'gdino')")
