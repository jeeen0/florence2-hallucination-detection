"""Combine base-ft and large-ft metric JSONs into a single comparison CSV/MD."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "base_metrics.json")
    ap.add_argument("--large", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "large_metrics.json")
    ap.add_argument("--out-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "compare_table.csv")
    ap.add_argument("--out-md", type=Path,
                    default=REPO_ROOT / "outputs" / "metrics" / "compare_table.md")
    args = ap.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    large = json.loads(args.large.read_text(encoding="utf-8"))

    rows = [
        ("Florence-2-base-ft caption (baseline)", base, "baseline"),
        ("Florence-2-base-ft verified (ours)", base, "verified"),
        ("Florence-2-large-ft caption (baseline)", large, "baseline"),
        ("Florence-2-large-ft verified (ours)", large, "verified"),
    ]

    def fmt(m: dict, kind: str) -> tuple:
        if kind == "baseline":
            return (
                m["n_mentions"], m["baseline_hallucinated"],
                f"{m['baseline_hallucination_rate']*100:.2f}%",
                "-", "-",
            )
        else:
            return (
                m["supported"], m["verified_hallucinated"],
                f"{m['verified_hallucination_rate']*100:.2f}%",
                f"{m['unsupported_f1']:.4f}",
                f"{m['grounding_acc_at_0.5']*100:.2f}%",
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Mentions", "Hallucinated",
                    "HallucinationRate", "UnsupportedF1", "GroundingAcc@0.5"])
        for label, m, kind in rows:
            w.writerow([label, *fmt(m, kind)])

    lines: list[str] = []
    lines.append("# Base-ft vs Large-ft Comparison\n")
    lines.append(f"- base-ft: n_images={base['n_images']}, n_mentions={base['n_mentions']}")
    lines.append(f"- large-ft: n_images={large['n_images']}, n_mentions={large['n_mentions']}\n")
    lines.append("| Method | Mentions | Hallucinated ↓ | Hall. Rate ↓ | Unsupp. F1 ↑ | Grd Acc@0.5 ↑ |")
    lines.append("| --- | --: | --: | --: | --: | --: |")
    for label, m, kind in rows:
        cells = fmt(m, kind)
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved -> {args.out_csv}")
    print(f"Saved -> {args.out_md}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
