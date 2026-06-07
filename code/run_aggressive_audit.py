"""Manual audit of the 50 mentions that the *aggressive* vocab adds
on top of the *strict* vocab (same 500 images, same captions, same seed).

Goal: §5.5 Vocab Ablation claims that aggressive synonym mapping inflates the
baseline hallucination rate by ~80% (2.46% → 4.48%). The audit asks, per each
of those 50 extra mentions: *is the underlying object visible in the image?*

  - 'visible'      → aggressive mapping is *not* introducing a hallucination,
                    just a legitimate object that the COCO GT does not label
                    under the canonical category.  The "+80% inflation" is
                    then GT-noise sensitivity, not real over-claiming.
  - 'not_visible'  → aggressive mapping is genuinely surfacing a hallucination
                    that the strict vocab silently dropped.  Aggressive is
                    catching MORE real hallucinations than strict.

This script reuses the matplotlib UI from `run_manual_audit.py` via the
`Auditor` class.  Output is written to
`outputs/metrics/aggressive_audit.csv` (separate from the main 2000 audit).

Usage:
    python code/run_aggressive_audit.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from run_manual_audit import (
    Auditor, REPO, COCO_GT, load_coco_gt, download_missing_images,
)


AGGRESSIVE_GROUNDING = REPO / "outputs" / "grounding" / "large_aggressive_500_grounding.csv"
STRICT_GROUNDING     = REPO / "outputs" / "grounding" / "large_strict_500_grounding.csv"
AGGRESSIVE_CAPTION   = REPO / "outputs" / "captions"  / "large_aggressive_caption_500.csv"
OUT_CSV              = REPO / "outputs" / "metrics"   / "aggressive_audit.csv"


def build_aggressive_queue() -> pd.DataFrame:
    """Return the 50 mentions that exist in aggressive but not in strict.

    These are the mentions where aggressive's broader synonym table produced
    a canonical (e.g. `table → dining table`, `baseball → sports ball`,
    `stove → oven`, `monitor → tv`) that the strict policy intentionally
    refused.
    """
    if not AGGRESSIVE_GROUNDING.exists():
        sys.exit(f"missing: {AGGRESSIVE_GROUNDING}")
    if not STRICT_GROUNDING.exists():
        sys.exit(f"missing: {STRICT_GROUNDING}")

    agg    = pd.read_csv(AGGRESSIVE_GROUNDING)
    strict = pd.read_csv(STRICT_GROUNDING)
    caps   = pd.read_csv(AGGRESSIVE_CAPTION).set_index("image_id")["caption"].to_dict()
    img_gt = load_coco_gt()

    strict_keys = set(zip(strict.image_id, strict.surface))
    extras = agg[~agg.apply(lambda r: (r.image_id, r.surface) in strict_keys, axis=1)].copy()

    extras["in_gt"]    = extras.apply(
        lambda r: r["canonical"] in img_gt.get(int(r["image_id"]), set()),
        axis=1,
    )
    # Single case type — these are all "aggressive-only" candidates.
    # We tag with the system's classification on aggressive for downstream
    # interpretation (so it appears in `case_type` column on the CSV).
    def _case(row):
        if row["in_gt"]:
            # mention's canonical is in COCO GT; aggressive's verdict either
            # confirmed (supported) or over-flagged (unsupported)
            return "agg_in_gt"
        # canonical NOT in COCO GT
        return "agg_not_in_gt_supp" if row["status"] == "supported" \
               else "agg_not_in_gt_unsupp"

    extras["case_type"] = extras.apply(_case, axis=1)
    extras["caption"]   = extras["image_id"].map(caps).fillna("")
    extras["coco_gt"]   = extras["image_id"].map(
        lambda i: ", ".join(sorted(img_gt.get(int(i), set()))) or "(empty)"
    )
    extras["row_key"]   = extras.apply(
        lambda r: f"agg_{int(r['image_id'])}_{r['surface']}_{r['canonical']}",
        axis=1,
    )
    return extras.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo-skips", action="store_true")
    parser.add_argument("--redo-borderline", action="store_true")
    parser.add_argument("--redo-all", action="store_true")
    parser.add_argument("--download-missing", action="store_true",
                        help="download missing COCO val2017 images for the queue")
    args = parser.parse_args()

    if not COCO_GT.exists():
        sys.exit(f"missing: {COCO_GT}")

    cases = build_aggressive_queue()
    print(f"\nAggressive-only mentions: {len(cases)}")
    print(cases["case_type"].value_counts().to_string())
    print()
    print("Sample (first 10):")
    print(cases[["image_id", "surface", "canonical", "status", "in_gt", "case_type"]]
          .head(10).to_string(index=False))
    print()

    if args.download_missing:
        download_missing_images(cases)

    Auditor(
        cases,
        out_csv=OUT_CSV,
        redo_skips=args.redo_skips,
        redo_borderline=args.redo_borderline,
        redo_all=args.redo_all,
        title_prefix="Aggressive Vocab Audit",
        finish_message=(
            "Aggressive 50 extras audit 완료!\n\n"
            "다음 단계: python code/run_aggressive_recompute.py\n"
            "창을 닫고 [q] 또는 콘솔에서 Ctrl+C."
        ),
    ).run()


if __name__ == "__main__":
    main()
