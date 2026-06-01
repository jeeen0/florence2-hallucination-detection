"""COCO-80 object-mention extraction from free-form captions.

Approach:
- Normalize caption (lowercase, strip non-letters except space/hyphen).
- Whole-word match against a flat (surface, canonical) lookup table.
- Multi-word surfaces (e.g. "dining table") are matched before single words to
  avoid being shadowed by a substring match.
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


SYNONYM_MAP: dict[str, list[str]] = {
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


def _build_lookup() -> list[tuple[str, str]]:
    table: list[tuple[str, str]] = []
    for canonical in COCO80:
        table.append((canonical, canonical))
        for syn in SYNONYM_MAP.get(canonical, []):
            table.append((syn, canonical))
    table.sort(key=lambda x: -len(x[0]))
    return table


_LOOKUP = _build_lookup()


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


def extract_mentions(caption: str) -> list[Mention]:
    text = normalize(caption)
    mentions: list[Mention] = []
    occupied: list[tuple[int, int]] = []
    for surface, canonical in _LOOKUP:
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


def extract_unique_canonicals(caption: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in extract_mentions(caption):
        if m.canonical not in seen:
            seen.add(m.canonical)
            out.append(m.canonical)
    return out
