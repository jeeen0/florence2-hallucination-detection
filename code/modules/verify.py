"""Verify caption mentions against Florence-2 visual outputs.

Default strategy: one <OD> call per image → match mention canonicals to detected
labels. Falls back to <CAPTION_TO_PHRASE_GROUNDING> per mention if requested.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from florence2 import Florence2Runner
from .extract import (
    COCO80,
    SYNONYM_MAP_AGGRESSIVE,
    SYNONYM_MAP_STRICT,
    Mention,
    normalize,
)


def _label_to_canonical_table(synonym_map: dict[str, list[str]]) -> dict[str, str]:
    table: dict[str, str] = {}
    for canonical in COCO80:
        table[canonical] = canonical
        for syn in synonym_map.get(canonical, []):
            table[syn] = canonical
    return table


_LABEL_STRICT = _label_to_canonical_table(SYNONYM_MAP_STRICT)
_LABEL_AGGRESSIVE = _label_to_canonical_table(SYNONYM_MAP_AGGRESSIVE)


def label_to_canonical(label: str, vocab: str = "strict") -> str | None:
    table = _LABEL_AGGRESSIVE if vocab == "aggressive" else _LABEL_STRICT
    return table.get(normalize(label))


@dataclass
class Detection:
    label: str
    canonical: str | None
    bbox: tuple[float, float, float, float]


@dataclass
class Verdict:
    mention: Mention
    supported: bool
    bbox: tuple[float, float, float, float] | None


def detect_objects(
    runner: Florence2Runner,
    image: Image.Image,
    vocab: str = "strict",
) -> list[Detection]:
    out = runner.run("<OD>", image)["<OD>"]
    bboxes = out.get("bboxes", [])
    labels = out.get("labels", [])
    detections: list[Detection] = []
    for bbox, label in zip(bboxes, labels):
        detections.append(
            Detection(
                label=str(label),
                canonical=label_to_canonical(str(label), vocab=vocab),
                bbox=tuple(bbox),
            )
        )
    return detections


def verify_with_detections(
    mentions: list[Mention],
    detections: list[Detection],
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for mention in mentions:
        match = next(
            (d for d in detections if d.canonical == mention.canonical),
            None,
        )
        verdicts.append(
            Verdict(
                mention=mention,
                supported=match is not None,
                bbox=match.bbox if match else None,
            )
        )
    return verdicts


def phrase_ground(
    runner: Florence2Runner,
    image: Image.Image,
    phrase: str,
) -> list[tuple[float, float, float, float]]:
    out = runner.run("<CAPTION_TO_PHRASE_GROUNDING>", image, text_input=phrase)
    parsed = out.get("<CAPTION_TO_PHRASE_GROUNDING>", {})
    return [tuple(b) for b in parsed.get("bboxes", [])]


def verify_with_phrase_grounding(
    runner: Florence2Runner,
    image: Image.Image,
    mentions: list[Mention],
) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for mention in mentions:
        bboxes = phrase_ground(runner, image, mention.canonical)
        verdicts.append(
            Verdict(
                mention=mention,
                supported=bool(bboxes),
                bbox=bboxes[0] if bboxes else None,
            )
        )
    return verdicts
