"""Loader for COCO instances_val2017.json. Exposes per-image GT category + bbox."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GtBox:
    category_id: int
    category_name: str
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)


class CocoGt:
    def __init__(self, instances_path: Path) -> None:
        with instances_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.categories: dict[int, str] = {c["id"]: c["name"] for c in data["categories"]}
        self._boxes: dict[int, list[GtBox]] = defaultdict(list)
        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            cid = ann["category_id"]
            self._boxes[ann["image_id"]].append(
                GtBox(
                    category_id=cid,
                    category_name=self.categories[cid],
                    bbox=(x, y, x + w, y + h),
                )
            )

    def boxes(self, image_id: int) -> list[GtBox]:
        return self._boxes.get(image_id, [])

    def categories_in(self, image_id: int) -> set[str]:
        return {b.category_name for b in self.boxes(image_id)}
