"""Compare all metric JSONs in outputs/metrics into a single table.

Reads every *_metrics.json file and produces:
  outputs/metrics/all_compare.csv
  outputs/metrics/all_compare.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics")
    ap.add_argument("--out-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "all_compare.csv")
    ap.add_argument("--out-md", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "all_compare.md")
    args = ap.parse_args()

    rows: list[dict] = []
    for jpath in sorted(args.metrics_dir.glob("*_metrics.json")):
        m = json.loads(jpath.read_text(encoding="utf-8"))
        rows.append({"file": jpath.name, **m})

    if not rows:
        print(f"No *_metrics.json under {args.metrics_dir}")
        return

    fieldnames = [
        "label", "n_images", "n_mentions", "supported", "unsupported",
        "baseline_hallucinated", "baseline_hallucination_rate",
        "verified_hallucinated", "verified_hallucination_rate",
        "unsupported_precision", "unsupported_recall", "unsupported_f1",
        "grounding_acc_at_0.5", "file",
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get("model" if k == "label" else k, "") for k in fieldnames}
            w.writerow(row)

    lines: list[str] = []
    lines.append("# All experiments comparison\n")
    lines.append("| Label | n_img | n_mention | Baseline Hall% | Verified Hall% | P | R | F1 | GrdAcc@0.5 |")
    lines.append("| --- | --: | --: | --: | --: | --: | --: | --: | --: |")
    for r in rows:
        lines.append(
            f"| {r.get('model', '')} | {r.get('n_images', '')} | {r.get('n_mentions', '')} "
            f"| {r.get('baseline_hallucination_rate', 0)*100:.2f}% "
            f"| {r.get('verified_hallucination_rate', 0)*100:.2f}% "
            f"| {r.get('unsupported_precision', 0):.4f} "
            f"| {r.get('unsupported_recall', 0):.4f} "
            f"| {r.get('unsupported_f1', 0):.4f} "
            f"| {r.get('grounding_acc_at_0.5', 0)*100:.2f}% |"
        )

    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved -> {args.out_csv}")
    print(f"Saved -> {args.out_md}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
