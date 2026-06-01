from .caption import generate_caption
from .extract import (
    COCO80,
    SYNONYM_MAP,
    SYNONYM_MAP_AGGRESSIVE,
    SYNONYM_MAP_STRICT,
    Mention,
    extract_mentions,
    extract_unique_canonicals,
)
from .verify import (
    Detection,
    Verdict,
    detect_objects,
    label_to_canonical,
    phrase_ground,
    verify_with_detections,
    verify_with_phrase_grounding,
)
from .visualize import draw_verdicts

__all__ = [
    "COCO80",
    "SYNONYM_MAP",
    "SYNONYM_MAP_AGGRESSIVE",
    "SYNONYM_MAP_STRICT",
    "Mention",
    "Detection",
    "Verdict",
    "extract_mentions",
    "extract_unique_canonicals",
    "generate_caption",
    "label_to_canonical",
    "detect_objects",
    "verify_with_detections",
    "phrase_ground",
    "verify_with_phrase_grounding",
    "draw_verdicts",
]
