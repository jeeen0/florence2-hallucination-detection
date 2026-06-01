"""Metrics for caption-grounding verification against COCO GT."""
from __future__ import annotations

import random
from typing import Callable, Sequence


def iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def confusion_unsupported(rows: list[dict]) -> dict[str, int]:
    """
    Treat the system's job as flagging unsupported mentions.
    Ground truth flag is "canonical is NOT in image GT categories".

    TP: predicted unsupported AND not in GT  (correctly flagged hallucination)
    FP: predicted unsupported AND in GT      (wrongly flagged a real object)
    FN: predicted supported AND not in GT    (missed hallucination)
    TN: predicted supported AND in GT        (correctly passed real object)
    """
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for r in rows:
        pred_unsupp = (r["status"] == "unsupported")
        in_gt = r["in_gt"]
        if pred_unsupp and not in_gt:
            cm["TP"] += 1
        elif pred_unsupp and in_gt:
            cm["FP"] += 1
        elif not pred_unsupp and not in_gt:
            cm["FN"] += 1
        else:
            cm["TN"] += 1
    return cm


def precision_recall_f1(cm: dict[str, int]) -> tuple[float, float, float]:
    p = safe_div(cm["TP"], cm["TP"] + cm["FP"])
    r = safe_div(cm["TP"], cm["TP"] + cm["FN"])
    f1 = safe_div(2 * p * r, p + r)
    return p, r, f1


def bootstrap_ci(
    values: Sequence,
    statistic: Callable,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point_estimate, lower, upper)."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    point = statistic(values)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        try:
            estimates.append(statistic(sample))
        except (ZeroDivisionError, ValueError):
            continue
    estimates.sort()
    if not estimates:
        return point, point, point
    alpha = (1 - confidence) / 2
    lo_idx = int(alpha * len(estimates))
    hi_idx = int((1 - alpha) * len(estimates)) - 1
    return point, estimates[lo_idx], estimates[max(hi_idx, 0)]


def mention_rate(values: Sequence[bool]) -> float:
    """Given a sequence of booleans (e.g., 'mention not in GT'), return fraction True."""
    if not values:
        return 0.0
    return sum(1 for v in values if v) / len(values)


def f1_from_flags(rows: Sequence[dict]) -> float:
    cm = confusion_unsupported(list(rows))
    _, _, f1 = precision_recall_f1(cm)
    return f1
