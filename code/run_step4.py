"""Step 4: compute metrics over Step 3 verification results vs COCO GT.

Reads:
  outputs/grounding/base_grounding_results.csv  (image_id, surface, canonical, status, x1..y4)
  data/coco_annotations/instances_val2017.json

Outputs:
  outputs/metrics/base_metrics.json
  outputs/metrics/base_result_table.csv

Metrics:
  Hallucination Rate          (CHAIR-like; mentions not in GT / total mentions)
  Confusion matrix on "predict unsupported" task
  Precision / Recall / F1
  Grounding Acc@0.5           (IoU vs GT boxes of same category)
  Two-row result table         (Baseline vs Verified)

Usage:
  python code/run_step4.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from coco_gt import CocoGt
from modules.metrics import (
    confusion_unsupported,
    iou,
    precision_recall_f1,
    safe_div,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_bbox(row: dict) -> tuple[float, float, float, float] | None:
    try:
        return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounding-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "grounding" / "base_grounding_results.csv")
    ap.add_argument("--gt-json", type=Path,
                    default=REPO_ROOT / "data" / "coco_annotations" / "instances_val2017.json")
    ap.add_argument("--out-json", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "base_metrics.json")
    ap.add_argument("--out-table", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "base_result_table.csv")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--label", default="Florence-2-base-ft")
    args = ap.parse_args()

    print(f"[step4] loading GT from {args.gt_json}")
    gt = CocoGt(args.gt_json)

    with args.grounding_csv.open("r", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    rows: list[dict] = []
    for r in raw:
        image_id = int(r["image_id"])
        gt_cats = gt.categories_in(image_id)
        in_gt = r["canonical"] in gt_cats
        bbox = _parse_bbox(r)
        rows.append({
            "image_id": image_id,
            "canonical": r["canonical"],
            "surface": r["surface"],
            "status": r["status"],
            "bbox": bbox,
            "in_gt": in_gt,
        })

    n_images = len({r["image_id"] for r in rows})
    n_mentions = len(rows)
    supported = [r for r in rows if r["status"] == "supported"]
    unsupported = [r for r in rows if r["status"] == "unsupported"]

    baseline_hallu = sum(1 for r in rows if not r["in_gt"])
    verified_hallu = sum(1 for r in supported if not r["in_gt"])
    baseline_rate = safe_div(baseline_hallu, n_mentions)
    verified_rate = safe_div(verified_hallu, len(supported))

    cm = confusion_unsupported(rows)
    precision, recall, f1 = precision_recall_f1(cm)

    grd_hits = 0
    grd_total = 0
    for r in supported:
        if r["bbox"] is None:
            continue
        gt_same = [b.bbox for b in gt.boxes(r["image_id"]) if b.category_name == r["canonical"]]
        if not gt_same:
            continue
        grd_total += 1
        best = max((iou(r["bbox"], gb) for gb in gt_same), default=0.0)
        if best >= args.iou_thr:
            grd_hits += 1
    grd_acc = safe_div(grd_hits, grd_total)

    metrics = {
        "model": args.label,
        "n_images": n_images,
        "n_mentions": n_mentions,
        "supported": len(supported),
        "unsupported": len(unsupported),
        "baseline_hallucinated": baseline_hallu,
        "baseline_hallucination_rate": baseline_rate,
        "verified_hallucinated": verified_hallu,
        "verified_hallucination_rate": verified_rate,
        "unsupported_confusion": cm,
        "unsupported_precision": precision,
        "unsupported_recall": recall,
        "unsupported_f1": f1,
        "grounding_iou_thr": args.iou_thr,
        "grounding_total_evaluable": grd_total,
        "grounding_hits": grd_hits,
        f"grounding_acc_at_{args.iou_thr}": grd_acc,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    with args.out_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Method", "Mentions", "Hallucinated",
            "HallucinationRate", "UnsupportedF1", f"GroundingAcc@{args.iou_thr}",
        ])
        w.writerow([
            f"{args.label} caption (baseline)",
            n_mentions, baseline_hallu,
            f"{baseline_rate:.4f}", "-", "-",
        ])
        w.writerow([
            f"{args.label} verified (ours)",
            len(supported), verified_hallu,
            f"{verified_rate:.4f}", f"{f1:.4f}", f"{grd_acc:.4f}",
        ])

    print(f"\nSaved -> {args.out_json}")
    print(f"Saved -> {args.out_table}")
    print("\n--- Summary ---")
    print(f"images={n_images}  mentions={n_mentions}  supported={len(supported)}  unsupported={len(unsupported)}")
    print(f"Baseline:  hallucinated={baseline_hallu}/{n_mentions}  rate={baseline_rate:.4f}")
    print(f"Verified:  hallucinated={verified_hallu}/{len(supported)}  rate={verified_rate:.4f}")
    print(f"Unsupported P/R/F1:  {precision:.4f} / {recall:.4f} / {f1:.4f}  (cm={cm})")
    print(f"Grounding Acc@{args.iou_thr}:  {grd_hits}/{grd_total}  =  {grd_acc:.4f}")


if __name__ == "__main__":
    main()
