"""Recompute confusion matrix & metrics after manual audit.

Reads `outputs/metrics/manual_audit.csv` (produced by `run_manual_audit.py`)
and applies the verdicts to the original grounding result to produce an
augmented-GT confusion matrix.

Logic:
    - FN + verdict 'visible'      → object IS present, GT missed it
                                  → reclassify: TN  (system correctly let through)
    - FN + verdict 'not_visible'  → real hallucination, system missed
                                  → keep:        FN
    - FP + verdict 'visible'      → real over-flag, system OD missed visible obj
                                  → keep:        FP
    - FP + verdict 'not_visible'  → GT labelled essentially-invisible object
                                  → reclassify: TP  (system correctly flagged)
    - verdict 'borderline'        → reported separately, also as a sensitivity
                                    interval (best/worst case).
    - verdict 'skip'              → ignored (acts like 'no audit performed')

Outputs:
    outputs/metrics/large_strict_2000_audited_metrics.json
    outputs/metrics/large_strict_2000_audited_result_table.csv
    console summary

Usage:
    python code/run_audit_recompute.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
GROUNDING_CSV = REPO / "outputs" / "grounding" / "large_strict_2000_grounding.csv"
COCO_GT       = REPO / "data" / "coco_annotations" / "instances_val2017.json"
AUDIT_CSV     = REPO / "outputs" / "metrics" / "manual_audit.csv"
RAW_METRICS   = REPO / "outputs" / "metrics" / "large_strict_2000_metrics.json"
OUT_JSON      = REPO / "outputs" / "metrics" / "large_strict_2000_audited_metrics.json"
OUT_TABLE     = REPO / "outputs" / "metrics" / "large_strict_2000_audited_result_table.csv"

RNG_SEED = 7
N_BOOT   = 1000


def load_gt() -> dict[int, set[str]]:
    with open(COCO_GT, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    out: dict[int, set[str]] = {}
    for a in coco["annotations"]:
        out.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
    return out


def label_row(status: str, in_gt: bool) -> str:
    """Raw CHAIR-style classification, before audit."""
    if not in_gt and status == "unsupported":
        return "TP"
    if in_gt     and status == "unsupported":
        return "FP"
    if not in_gt and status == "supported":
        return "FN"
    return "TN"


def confusion(labels: pd.Series) -> dict[str, int]:
    c = labels.value_counts()
    return {k: int(c.get(k, 0)) for k in ("TP", "FP", "FN", "TN")}


def prf(c: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = c["TP"], c["FP"], c["FN"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f}


def bootstrap_ci(labels: list[str], stat: str,
                 n_boot: int = N_BOOT, seed: int = RNG_SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for precision / recall / f1 / hall_rate."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    arr = np.array(labels)
    out = []
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        c = {k: int((sample == k).sum()) for k in ("TP", "FP", "FN", "TN")}
        if stat == "hall_rate_baseline":
            denom = sum(c.values())
            num   = c["TP"] + c["FN"]  # GT-absent total
            out.append(num / denom if denom else 0.0)
        elif stat == "hall_rate_verified":
            denom = sum(c.values())
            num   = c["FN"]  # GT-absent AND system-passed
            out.append(num / denom if denom else 0.0)
        else:
            out.append(prf(c)[stat])
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def apply_audit(orig_label: str, case_type: str, verdict: str) -> str:
    """Apply the verdict to one case to produce the audited label."""
    if case_type == "FN":
        if verdict == "visible":      return "TN"  # GT noise corrected
        if verdict == "not_visible":  return "FN"
    if case_type == "FP":
        if verdict == "visible":      return "FP"
        if verdict == "not_visible":  return "TP"  # COCO over-label corrected
    # borderline / skip → keep original
    return orig_label


def main():
    if not AUDIT_CSV.exists():
        sys.exit(f"missing: {AUDIT_CSV}  (run code/run_manual_audit.py first)")

    g = pd.read_csv(GROUNDING_CSV)
    img_gt = load_gt()
    g["in_gt"]    = g.apply(lambda r: r["canonical"] in img_gt.get(int(r["image_id"]), set()), axis=1)
    g["raw_label"] = g.apply(lambda r: label_row(r["status"], r["in_gt"]), axis=1)
    g["row_key"]   = g.apply(
        lambda r: f"{int(r['image_id'])}_{r['canonical']}_{r['raw_label']}",
        axis=1,
    )

    audit = pd.read_csv(AUDIT_CSV)
    # latest verdict per row_key (in case of re-audits / overwrites)
    audit = audit.sort_values("timestamp").drop_duplicates("row_key", keep="last")

    audit_map = dict(zip(audit["row_key"], audit["verdict"]))
    g["verdict"] = g["row_key"].map(audit_map).fillna("none")

    # Audited label — only FN / FP rows can change.
    def audited(row):
        if row["raw_label"] not in ("FN", "FP"):
            return row["raw_label"]
        return apply_audit(row["raw_label"], row["raw_label"], row["verdict"])

    g["audited_label"]      = g.apply(audited, axis=1)
    # Best-case: borderlines treated favourably for the system
    # Worst-case: borderlines treated unfavourably
    def audited_best(row):
        if row["verdict"] != "borderline":
            return row["audited_label"]
        if row["raw_label"] == "FN": return "TN"
        if row["raw_label"] == "FP": return "TP"
        return row["audited_label"]

    def audited_worst(row):
        if row["verdict"] != "borderline":
            return row["audited_label"]
        return row["raw_label"]  # borderline kept as raw (pessimistic)

    g["audited_best"]  = g.apply(audited_best, axis=1)
    g["audited_worst"] = g.apply(audited_worst, axis=1)

    raw_conf      = confusion(g["raw_label"])
    audited_conf  = confusion(g["audited_label"])
    best_conf     = confusion(g["audited_best"])
    worst_conf    = confusion(g["audited_worst"])

    def with_ci(labels_series, label):
        labels_list = labels_series.tolist()
        c = confusion(labels_series)
        m = prf(c)
        n = sum(c.values())
        hall_base = (c["TP"] + c["FN"]) / n if n else 0.0
        hall_ver  = c["FN"] / n if n else 0.0
        return {
            "label": label,
            "n_mentions": n,
            "TP": c["TP"], "FP": c["FP"], "FN": c["FN"], "TN": c["TN"],
            "precision": m["precision"],
            "precision_95ci": list(bootstrap_ci(labels_list, "precision")),
            "recall":    m["recall"],
            "recall_95ci":    list(bootstrap_ci(labels_list, "recall")),
            "f1":        m["f1"],
            "f1_95ci":        list(bootstrap_ci(labels_list, "f1")),
            "hall_rate_baseline": hall_base,
            "hall_rate_baseline_95ci": list(bootstrap_ci(labels_list, "hall_rate_baseline")),
            "hall_rate_verified": hall_ver,
            "hall_rate_verified_95ci": list(bootstrap_ci(labels_list, "hall_rate_verified")),
        }

    print("\n=== Audit summary ===")
    audited_subset = audit[audit["verdict"].isin(["visible", "not_visible", "borderline"])]
    summary = (audited_subset
               .groupby(["case_type", "verdict"])
               .size()
               .unstack(fill_value=0))
    print(summary.to_string())
    print(f"\nTotal audited (non-skip): {len(audited_subset)}")
    print(f"Skipped:                  {(audit['verdict'] == 'skip').sum()}")

    rows = []
    for tag, series in [
        ("raw (COCO val2017)",          g["raw_label"]),
        ("audited (point estimate)",    g["audited_label"]),
        ("audited (best, borderline→ok)",  g["audited_best"]),
        ("audited (worst, borderline→bad)", g["audited_worst"]),
    ]:
        rows.append(with_ci(series, tag))

    for r in rows:
        print(f"\n--- {r['label']} ---")
        print(f"  n={r['n_mentions']}  "
              f"TP/FP/FN/TN = {r['TP']}/{r['FP']}/{r['FN']}/{r['TN']}")
        print(f"  Hall%(base) = {r['hall_rate_baseline']*100:.2f} "
              f"[{r['hall_rate_baseline_95ci'][0]*100:.2f}, "
              f"{r['hall_rate_baseline_95ci'][1]*100:.2f}]")
        print(f"  Hall%(ver)  = {r['hall_rate_verified']*100:.2f} "
              f"[{r['hall_rate_verified_95ci'][0]*100:.2f}, "
              f"{r['hall_rate_verified_95ci'][1]*100:.2f}]")
        print(f"  P = {r['precision']:.3f}  "
              f"R = {r['recall']:.3f}  "
              f"F1 = {r['f1']:.3f} "
              f"[{r['f1_95ci'][0]:.3f}, {r['f1_95ci'][1]:.3f}]")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    pd.DataFrame(rows)[
        ["label", "n_mentions", "TP", "FP", "FN", "TN",
         "hall_rate_baseline", "hall_rate_verified",
         "precision", "recall", "f1"]
    ].to_csv(OUT_TABLE, index=False)

    print(f"\nwrote: {OUT_JSON}")
    print(f"wrote: {OUT_TABLE}")


if __name__ == "__main__":
    main()
