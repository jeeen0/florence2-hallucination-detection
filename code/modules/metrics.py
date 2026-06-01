"""Metrics for caption-grounding verification against COCO GT."""
from __future__ import annotations


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
