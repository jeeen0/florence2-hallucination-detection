"""COCO-80 object-mention extraction from free-form captions.

Two synonym vocabularies are maintained for ablation:

- ``SYNONYM_MAP_AGGRESSIVE`` (legacy): includes generic-to-specific mappings such
  as ``table → dining table``, ``baseball → sports ball``, ``monitor → tv``.
  Treats a wide surface form vocabulary, at the cost of over-flagging mentions
  that the captioner did not literally make.

- ``SYNONYM_MAP_STRICT`` (default): drops mappings whose surface form is more
  general than the canonical category (``table``, ``plant``, ``stove``,
  ``monitor``, generic ``glove``) or whose canonical is a different physical
  object than the surface form's natural reading (``baseball`` → sport context,
  not the ball). Pilot analysis showed the aggressive map turned the apparent
  hallucination rate into a measure of vocabulary alignment rather than visual
  consistency.

Both maps share the same matching algorithm:
- Normalize caption (lowercase, strip non-letters except space/hyphen).
- Whole-word match against a flat (surface, canonical) lookup table.
- Multi-word surfaces are matched first (length-descending).
- Overlapping matches are dropped (first/longest wins).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


COCO80: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


SYNONYM_MAP_AGGRESSIVE: dict[str, list[str]] = {
    "person": [
        "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
        "kid", "kids", "child", "children", "people", "guy", "guys",
        "lady", "ladies", "human", "humans", "pedestrian", "pedestrians",
    ],
    "bicycle": ["bike", "bikes", "bicycles"],
    "car": ["cars", "automobile", "automobiles", "sedan", "suv"],
    "motorcycle": ["motorbike", "motorbikes", "motorcycles"],
    "airplane": ["plane", "planes", "aeroplane", "airplanes", "jet", "aircraft"],
    "bus": ["buses"],
    "train": ["trains"],
    "truck": ["trucks", "lorry"],
    "boat": ["boats", "ship", "ships"],
    "traffic light": ["traffic lights", "stoplight", "stoplights", "traffic signal"],
    "fire hydrant": ["fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop signs"],
    "parking meter": ["parking meters"],
    "bench": ["benches"],
    "bird": ["birds"],
    "cat": ["cats", "kitten", "kittens"],
    "dog": ["dogs", "puppy", "puppies"],
    "horse": ["horses"],
    "sheep": ["lamb", "lambs"],
    "cow": ["cows", "cattle"],
    "elephant": ["elephants"],
    "bear": ["bears"],
    "zebra": ["zebras"],
    "giraffe": ["giraffes"],
    "backpack": ["backpacks", "rucksack"],
    "umbrella": ["umbrellas"],
    "handbag": ["handbags", "purse", "purses"],
    "tie": ["ties", "necktie", "neckties"],
    "suitcase": ["suitcases", "luggage"],
    "frisbee": ["frisbees"],
    "skis": ["ski"],
    "snowboard": ["snowboards"],
    "sports ball": ["soccer ball", "basketball", "football", "tennis ball", "baseball"],
    "kite": ["kites"],
    "baseball bat": ["bats"],
    "baseball glove": ["mitt", "glove", "gloves"],
    "skateboard": ["skateboards"],
    "surfboard": ["surfboards"],
    "tennis racket": ["racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottles"],
    "wine glass": ["wine glasses", "glass of wine"],
    "cup": ["cups", "mug", "mugs"],
    "fork": ["forks"],
    "knife": ["knives"],
    "spoon": ["spoons"],
    "bowl": ["bowls"],
    "banana": ["bananas"],
    "apple": ["apples"],
    "sandwich": ["sandwiches"],
    "orange": ["oranges"],
    "broccoli": [],
    "carrot": ["carrots"],
    "hot dog": ["hotdog", "hotdogs", "hot dogs"],
    "pizza": ["pizzas"],
    "donut": ["donuts", "doughnut", "doughnuts"],
    "cake": ["cakes"],
    "chair": ["chairs"],
    "couch": ["sofa", "sofas", "couches"],
    "potted plant": [
        "potted plants", "plant", "plants", "houseplant", "houseplants",
        "flower pot", "flower pots",
    ],
    "bed": ["beds"],
    "dining table": ["table", "tables"],
    "toilet": ["toilets"],
    "tv": ["television", "televisions", "tvs", "monitor"],
    "laptop": ["laptops", "laptop computer", "notebook computer"],
    "mouse": ["computer mouse"],
    "remote": ["remotes", "remote control", "remote controls"],
    "keyboard": ["keyboards"],
    "cell phone": [
        "cellphone", "cellphones", "phone", "phones",
        "mobile phone", "smartphone", "smartphones",
    ],
    "microwave": ["microwaves"],
    "oven": ["ovens", "stove", "stoves"],
    "toaster": ["toasters"],
    "sink": ["sinks"],
    "refrigerator": ["refrigerators", "fridge", "fridges"],
    "book": ["books"],
    "clock": ["clocks"],
    "vase": ["vases"],
    "scissors": [],
    "teddy bear": ["teddy bears"],
    "hair drier": ["hair dryer", "hair dryers", "hairdryer", "hairdryers", "blow dryer"],
    "toothbrush": ["toothbrushes"],
}


SYNONYM_MAP_STRICT: dict[str, list[str]] = {
    "person": [
        "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
        "kid", "kids", "child", "children", "people", "guy", "guys",
        "lady", "ladies", "human", "humans", "pedestrian", "pedestrians",
    ],
    "bicycle": ["bike", "bikes", "bicycles"],
    "car": ["cars", "automobile", "automobiles", "sedan", "suv"],
    "motorcycle": ["motorbike", "motorbikes", "motorcycles"],
    "airplane": ["plane", "planes", "aeroplane", "airplanes", "jet", "aircraft"],
    "bus": ["buses"],
    "train": ["trains"],
    "truck": ["trucks", "lorry"],
    "boat": ["boats", "ship", "ships"],
    "traffic light": ["traffic lights", "stoplight", "stoplights", "traffic signal"],
    "fire hydrant": ["fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop signs"],
    "parking meter": ["parking meters"],
    "bench": ["benches"],
    "bird": ["birds"],
    "cat": ["cats", "kitten", "kittens"],
    "dog": ["dogs", "puppy", "puppies"],
    "horse": ["horses"],
    "sheep": ["lamb", "lambs"],
    "cow": ["cows", "cattle"],
    "elephant": ["elephants"],
    "bear": ["bears"],
    "zebra": ["zebras"],
    "giraffe": ["giraffes"],
    "backpack": ["backpacks", "rucksack"],
    "umbrella": ["umbrellas"],
    "handbag": ["handbags", "purse", "purses"],
    "tie": ["ties", "necktie", "neckties"],
    "suitcase": ["suitcases", "luggage"],
    "frisbee": ["frisbees"],
    "skis": ["ski"],
    "snowboard": ["snowboards"],
    # STRICT: only surface forms that literally contain "ball".
    "sports ball": ["soccer ball", "tennis ball"],
    "kite": ["kites"],
    # STRICT: drop generic "bats" (could be animal); only canonical "baseball bat".
    "baseball bat": [],
    # STRICT: drop generic "glove"/"gloves"; keep only specific "mitt".
    "baseball glove": ["mitt"],
    "skateboard": ["skateboards"],
    "surfboard": ["surfboards"],
    "tennis racket": ["racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottles"],
    "wine glass": ["wine glasses", "glass of wine"],
    "cup": ["cups", "mug", "mugs"],
    "fork": ["forks"],
    "knife": ["knives"],
    "spoon": ["spoons"],
    "bowl": ["bowls"],
    "banana": ["bananas"],
    "apple": ["apples"],
    "sandwich": ["sandwiches"],
    "orange": ["oranges"],
    "broccoli": [],
    "carrot": ["carrots"],
    "hot dog": ["hotdog", "hotdogs", "hot dogs"],
    "pizza": ["pizzas"],
    "donut": ["donuts", "doughnut", "doughnuts"],
    "cake": ["cakes"],
    "chair": ["chairs"],
    "couch": ["sofa", "sofas", "couches"],
    # STRICT: drop generic "plant"/"plants".
    "potted plant": [
        "potted plants", "houseplant", "houseplants",
        "flower pot", "flower pots",
    ],
    "bed": ["beds"],
    # STRICT: drop generic "table"/"tables".
    "dining table": [],
    "toilet": ["toilets"],
    # STRICT: drop "monitor" (could be computer display).
    "tv": ["television", "televisions", "tvs"],
    "laptop": ["laptops", "laptop computer", "notebook computer"],
    "mouse": ["computer mouse"],
    "remote": ["remotes", "remote control", "remote controls"],
    "keyboard": ["keyboards"],
    "cell phone": [
        "cellphone", "cellphones", "phone", "phones",
        "mobile phone", "smartphone", "smartphones",
    ],
    "microwave": ["microwaves"],
    # STRICT: drop "stove"/"stoves" (separate appliance in COCO context).
    "oven": ["ovens"],
    "toaster": ["toasters"],
    "sink": ["sinks"],
    "refrigerator": ["refrigerators", "fridge", "fridges"],
    "book": ["books"],
    "clock": ["clocks"],
    "vase": ["vases"],
    "scissors": [],
    "teddy bear": ["teddy bears"],
    "hair drier": ["hair dryer", "hair dryers", "hairdryer", "hairdryers", "blow dryer"],
    "toothbrush": ["toothbrushes"],
}


SYNONYM_MAP = SYNONYM_MAP_STRICT  # default for backwards-compat imports


def _build_lookup(synonym_map: dict[str, list[str]]) -> list[tuple[str, str]]:
    table: list[tuple[str, str]] = []
    for canonical in COCO80:
        table.append((canonical, canonical))
        for syn in synonym_map.get(canonical, []):
            table.append((syn, canonical))
    table.sort(key=lambda x: -len(x[0]))
    return table


_LOOKUP_STRICT = _build_lookup(SYNONYM_MAP_STRICT)
_LOOKUP_AGGRESSIVE = _build_lookup(SYNONYM_MAP_AGGRESSIVE)


def _select_lookup(vocab: str) -> list[tuple[str, str]]:
    if vocab == "aggressive":
        return _LOOKUP_AGGRESSIVE
    if vocab == "strict":
        return _LOOKUP_STRICT
    raise ValueError(f"Unknown vocab '{vocab}' (expected 'strict' or 'aggressive')")


@dataclass
class Mention:
    surface: str
    canonical: str
    start: int
    end: int


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 \-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_mentions(caption: str, vocab: str = "strict") -> list[Mention]:
    text = normalize(caption)
    lookup = _select_lookup(vocab)
    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    for surface, canonical in lookup:
        pat = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
        for m in re.finditer(pat, text):
            if any(s <= m.start() < e or s < m.end() <= e for s, e in occupied):
                continue
            occupied.append((m.start(), m.end()))
            mentions.append(
                Mention(surface=surface, canonical=canonical, start=m.start(), end=m.end())
            )
    mentions.sort(key=lambda m: m.start)
    return mentions


def extract_unique_canonicals(caption: str, vocab: str = "strict") -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in extract_mentions(caption, vocab=vocab):
        if m.canonical not in seen:
            seen.add(m.canonical)
            out.append(m.canonical)
    return out
