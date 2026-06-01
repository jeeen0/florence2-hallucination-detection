"""Phase 5 ablation: strict vs aggressive synonym vocab on the same 500 images.

We reuse the existing ``large_strict_caption_500.csv`` captions (caption text is
vocab-independent) and produce a parallel aggressive-vocab variant. Then run a
fresh Florence-2 OD verification pass for the aggressive variant and compare.

Outputs:
  outputs/captions/large_aggressive_caption_500.csv
  outputs/captions/large_aggressive_extracted_500.csv
  outputs/grounding/large_aggressive_500_grounding.csv
  outputs/metrics/large_aggressive_500_metrics.json
  outputs/metrics/large_aggressive_500_result_table.csv
  outputs/visualizations/large_aggressive_500/
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict-caption-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "large_strict_caption_500.csv")
    ap.add_argument("--model", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    py = args.python
    code_dir = REPO_ROOT / "code"

    out_cap = REPO_ROOT / "outputs" / "captions" / "large_aggressive_caption_500.csv"
    out_men = REPO_ROOT / "outputs" / "captions" / "large_aggressive_extracted_500.csv"
    out_grd = REPO_ROOT / "outputs" / "grounding" / "large_aggressive_500_grounding.csv"
    out_viz = REPO_ROOT / "outputs" / "visualizations" / "large_aggressive_500"
    out_json = REPO_ROOT / "outputs" / "metrics" / "large_aggressive_500_metrics.json"
    out_table = REPO_ROOT / "outputs" / "metrics" / "large_aggressive_500_result_table.csv"

    # 1. Re-extract mentions with aggressive vocab (no model call)
    run([
        py, str(code_dir / "extract_only.py"),
        "--caption-csv", str(args.strict_caption_csv),
        "--out-caption", str(out_cap),
        "--out-mentions", str(out_men),
        "--vocab", "aggressive",
    ])

    # 2. Step 3 OD verification with aggressive vocab
    out_viz.mkdir(parents=True, exist_ok=True)
    run([
        py, str(code_dir / "run_step3.py"),
        "--model", args.model,
        "--vocab", "aggressive",
        "--caption-csv", str(out_cap),
        "--mentions-csv", str(out_men),
        "--out-csv", str(out_grd),
        "--out-viz-dir", str(out_viz),
    ])

    # 3. Step 4 metrics
    run([
        py, str(code_dir / "run_step4.py"),
        "--grounding-csv", str(out_grd),
        "--out-json", str(out_json),
        "--out-table", str(out_table),
        "--label", "Florence-2-large-ft (aggressive, 500, CAPTION)",
    ])

    # 4. Refresh combined compare
    run([py, str(code_dir / "run_compare_all.py")])


if __name__ == "__main__":
    main()
