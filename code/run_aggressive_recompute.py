"""Recompute aggressive vocab ablation breakdown using the manual audit.

Inputs:
    outputs/metrics/aggressive_audit.csv      (from run_aggressive_audit.py)
    outputs/grounding/large_aggressive_500_grounding.csv
    outputs/grounding/large_strict_500_grounding.csv
    outputs/metrics/large_aggressive_500_metrics.json
    outputs/metrics/large_strict_500_metrics.json

Outputs:
    outputs/metrics/aggressive_audit_breakdown.json
    outputs/metrics/aggressive_audit_breakdown.csv
    console summary suitable for §5.5 of the paper

Per the audit, each of the 50 aggressive-only mentions receives a verdict:
    - 'visible'      → the object IS in the image (GT noise on the aggressive
                       canonical). Aggressive's "extra mention" is not a real
                       hallucination — strict policy is over-conservative.
    - 'not_visible'  → the object is NOT in the image. Aggressive's "extra
                       mention" is a real hallucination that strict misses.
    - 'borderline'   → ambiguous.
    - 'skip' / none  → un-audited; counted separately.

Headline number for §5.5:
    "Of the 50 mentions that the aggressive vocab adds, X were visible
     (GT noise) and Y were not visible (real hallucinations the strict vocab
     silently drops)."
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
AGG_GROUNDING = REPO / "outputs" / "grounding" / "large_aggressive_500_grounding.csv"
STRICT_GROUNDING = REPO / "outputs" / "grounding" / "large_strict_500_grounding.csv"
AUDIT_CSV     = REPO / "outputs" / "metrics" / "aggressive_audit.csv"
AGG_METRICS   = REPO / "outputs" / "metrics" / "large_aggressive_500_metrics.json"
STRICT_METRICS = REPO / "outputs" / "metrics" / "large_strict_500_metrics.json"
OUT_JSON      = REPO / "outputs" / "metrics" / "aggressive_audit_breakdown.json"
OUT_CSV       = REPO / "outputs" / "metrics" / "aggressive_audit_breakdown.csv"


def main():
    if not AUDIT_CSV.exists():
        sys.exit(f"missing: {AUDIT_CSV}  (run code/run_aggressive_audit.py first)")

    agg = pd.read_csv(AGG_GROUNDING)
    strict = pd.read_csv(STRICT_GROUNDING)
    audit = pd.read_csv(AUDIT_CSV)
    audit = audit.sort_values("timestamp").drop_duplicates("row_key", keep="last")

    strict_keys = set(zip(strict.image_id, strict.surface))
    extras = agg[~agg.apply(lambda r: (r.image_id, r.surface) in strict_keys, axis=1)].copy()
    extras["row_key"] = extras.apply(
        lambda r: f"agg_{int(r['image_id'])}_{r['surface']}_{r['canonical']}",
        axis=1,
    )
    extras = extras.merge(audit[["row_key", "verdict"]], on="row_key", how="left")
    extras["verdict"] = extras["verdict"].fillna("not_audited")

    print(f"Aggressive-only mentions: {len(extras)}")
    print()
    print("=== Verdict counts ===")
    print(extras["verdict"].value_counts().to_string())
    print()

    print("=== Verdict × original aggressive status ===")
    cross = (extras.groupby(["verdict", "status"]).size().unstack(fill_value=0))
    print(cross.to_string())
    print()

    print("=== Verdict × canonical (top 15) ===")
    canon = (extras.groupby(["canonical", "verdict"]).size().unstack(fill_value=0)
             .sort_values(by=list(extras["verdict"].unique()), ascending=False))
    print(canon.head(15).to_string())
    print()

    # Headline numbers for §5.5
    n_total       = len(extras)
    n_visible     = int((extras["verdict"] == "visible").sum())
    n_notvisible  = int((extras["verdict"] == "not_visible").sum())
    n_borderline  = int((extras["verdict"] == "borderline").sum())
    n_unaudited   = int(extras["verdict"].isin(["not_audited", "skip"]).sum())

    print("=== Headline (paper §5.5) ===")
    print(f"  aggressive-only mentions:   {n_total}")
    print(f"    visible (GT noise):        {n_visible}  "
          f"({100.0 * n_visible / n_total:.1f}%)")
    print(f"    not_visible (real hall):   {n_notvisible}  "
          f"({100.0 * n_notvisible / n_total:.1f}%)")
    print(f"    borderline:                {n_borderline}")
    print(f"    unaudited / skipped:       {n_unaudited}")
    print()

    # Effect on baseline hall%
    # strict baseline = 18 hallucinations / 732 mentions = 2.46%
    # aggressive baseline raw = 35 / 782 = 4.48% (per existing metric)
    try:
        with open(STRICT_METRICS, "r", encoding="utf-8") as f:
            strict_m = json.load(f)
        with open(AGG_METRICS, "r", encoding="utf-8") as f:
            agg_m = json.load(f)
    except Exception:
        strict_m = agg_m = {}

    n_agg     = len(agg)
    raw_agg_hall = (~extras["status"].isna() & (extras.get("status") == "unsupported")).sum() \
                  if False else None  # placeholder

    # Compute aggressive baseline hall% raw vs audit-corrected.
    # raw: any extra mention where canonical not in COCO GT counts as hall.
    # audited: extra mention where verdict='not_visible' counts as real hall;
    #          extra mention where verdict='visible'   counts as TN (no hall).
    # Borderline kept as raw (worst-case).
    g = agg.copy()
    # Compute in_gt
    from run_manual_audit import load_coco_gt
    img_gt = load_coco_gt()
    g["in_gt"] = g.apply(
        lambda r: r["canonical"] in img_gt.get(int(r["image_id"]), set()),
        axis=1,
    )
    g["raw_hall"] = ~g["in_gt"]   # CHAIR-style: not in GT → counted as hall mention

    # Merge audit verdict into the full aggressive grounding for the EXTRAS only.
    g["row_key"] = g.apply(
        lambda r: f"agg_{int(r['image_id'])}_{r['surface']}_{r['canonical']}",
        axis=1,
    )
    g = g.merge(audit[["row_key", "verdict"]], on="row_key", how="left")

    def audited_hall(r) -> bool:
        # Only extras have a verdict. Non-extras keep raw classification.
        v = r["verdict"]
        if pd.isna(v) or v in ("not_audited", "skip"):
            return bool(r["raw_hall"])
        if v == "visible":
            return False
        if v == "not_visible":
            return True
        if v == "borderline":
            return bool(r["raw_hall"])
        return bool(r["raw_hall"])

    g["audited_hall"] = g.apply(audited_hall, axis=1)
    n_mentions = len(g)
    raw_hall_count     = int(g["raw_hall"].sum())
    audit_hall_count   = int(g["audited_hall"].sum())
    raw_hall_pct       = 100.0 * raw_hall_count   / n_mentions
    audit_hall_pct     = 100.0 * audit_hall_count / n_mentions

    print(f"=== Aggressive baseline hall% (this experiment, n_mentions={n_mentions}) ===")
    print(f"  raw                : {raw_hall_count:3d} / {n_mentions} = {raw_hall_pct:.2f}%")
    print(f"  audit-corrected    : {audit_hall_count:3d} / {n_mentions} = {audit_hall_pct:.2f}%")
    print()

    # Recall the strict baseline for §5.5 comparison
    print("=== For §5.5 comparison ===")
    print(f"  strict baseline (raw, from paper)   : ~2.46% (18 / 732)")
    print(f"  aggressive baseline (raw)           : {raw_hall_pct:.2f}% "
          f"({raw_hall_count} / {n_mentions})")
    print(f"  aggressive baseline (audit-corrected): {audit_hall_pct:.2f}% "
          f"({audit_hall_count} / {n_mentions})")
    print()

    # Save outputs
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_aggressive_only": n_total,
        "verdict_counts": {
            "visible":      n_visible,
            "not_visible":  n_notvisible,
            "borderline":   n_borderline,
            "unaudited":    n_unaudited,
        },
        "n_mentions_total":           n_mentions,
        "raw_hall_count":             raw_hall_count,
        "raw_hall_pct":               raw_hall_pct,
        "audit_corrected_hall_count": audit_hall_count,
        "audit_corrected_hall_pct":   audit_hall_pct,
        "per_canonical_audit": {
            k: {kk: int(vv) for kk, vv in v.items()}
            for k, v in canon.to_dict("index").items()
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame([{
        "n_aggressive_only":           n_total,
        "verdict_visible":             n_visible,
        "verdict_not_visible":         n_notvisible,
        "verdict_borderline":          n_borderline,
        "verdict_unaudited":           n_unaudited,
        "n_mentions_total":            n_mentions,
        "raw_hall_count":              raw_hall_count,
        "raw_hall_pct":                raw_hall_pct,
        "audit_corrected_hall_count":  audit_hall_count,
        "audit_corrected_hall_pct":    audit_hall_pct,
    }]).to_csv(OUT_CSV, index=False)

    print(f"wrote: {OUT_JSON}")
    print(f"wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
