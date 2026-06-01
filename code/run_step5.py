"""Step 5: pick qualitative success / failure cases for the paper.

Logic:
  Per (image, mention) compute:
    pred  = supported | unsupported  (from Step 3 result)
    in_gt = canonical ∈ image GT categories (from COCO instances_val2017.json)

  Classification:
    TP  = pred=unsupp, in_gt=F   ("our system correctly flagged a hallucination")
    FN  = pred=supp,   in_gt=F   ("missed: caption AND OD both hallucinated")
    FP  = pred=unsupp, in_gt=T   ("over-flagged: real object was rejected")

Outputs:
  outputs/qualitative/success/<id>_TP_<canonical>.jpg          (Step 3 viz copy)
  outputs/qualitative/failure_FN/<id>_FN_<canonical>.jpg
  outputs/qualitative/failure_FP/<id>_FP_<canonical>.jpg
  outputs/qualitative/cases.md                                  Markdown summary
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from coco_gt import CocoGt


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_captions(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["image_id"])] = row["caption"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounding-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "grounding" / "base_grounding_results.csv")
    ap.add_argument("--caption-csv", type=Path,
                    default=REPO_ROOT / "outputs" / "captions" / "base_caption_50.csv")
    ap.add_argument("--viz-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "visualizations" / "base")
    ap.add_argument("--gt-json", type=Path,
                    default=REPO_ROOT / "data" / "coco_annotations" / "instances_val2017.json")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs" / "qualitative")
    args = ap.parse_args()

    gt = CocoGt(args.gt_json)
    captions = _load_captions(args.caption_csv)

    cases: dict[str, list[dict]] = defaultdict(list)

    with args.grounding_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = int(row["image_id"])
            canonical = row["canonical"]
            pred_unsupp = (row["status"] == "unsupported")
            in_gt = canonical in gt.categories_in(image_id)

            if pred_unsupp and not in_gt:
                tag = "TP"
            elif pred_unsupp and in_gt:
                tag = "FP"
            elif not pred_unsupp and not in_gt:
                tag = "FN"
            else:
                tag = "TN"

            cases[tag].append({
                "image_id": image_id,
                "surface": row["surface"],
                "canonical": canonical,
                "caption": captions.get(image_id, ""),
                "gt_categories": sorted(gt.categories_in(image_id)),
            })

    # copy viz files into folders
    out_success = args.out_dir / "success"
    out_fn = args.out_dir / "failure_FN"
    out_fp = args.out_dir / "failure_FP"
    for d in (out_success, out_fn, out_fp):
        d.mkdir(parents=True, exist_ok=True)

    def _copy(tag: str, dest: Path) -> None:
        for c in cases[tag]:
            src = args.viz_dir / f"{c['image_id']:012d}_verified.jpg"
            if not src.exists():
                continue
            shutil.copy(src, dest / f"{c['image_id']:012d}_{tag}_{c['canonical'].replace(' ', '_')}.jpg")

    _copy("TP", out_success)
    _copy("FN", out_fn)
    _copy("FP", out_fp)

    # markdown summary
    md_path = args.out_dir / "cases.md"
    lines: list[str] = []
    lines.append("# Qualitative Cases — Florence-2-base-ft, 50 COCO val2017 images\n")
    lines.append(f"- TP (success): {len(cases['TP'])}")
    lines.append(f"- FN (missed hallucination): {len(cases['FN'])}")
    lines.append(f"- FP (over-flag): {len(cases['FP'])}")
    lines.append(f"- TN (correctly passed): {len(cases['TN'])}\n")

    for tag, label in (("TP", "## ✅ Success cases (predicted unsupported AND not in GT)"),
                      ("FN", "## ❌ Failure cases — missed (supported BUT not in GT)"),
                      ("FP", "## ⚠️ Failure cases — over-flag (unsupported BUT in GT)")):
        lines.append(label)
        if not cases[tag]:
            lines.append("(none)\n")
            continue
        for c in cases[tag]:
            lines.append(f"### image_id `{c['image_id']:012d}` — mention `{c['canonical']}` (surface: `{c['surface']}`)")
            lines.append(f"- Caption: _{c['caption']}_")
            lines.append(f"- GT categories: `{', '.join(c['gt_categories']) or '(none)'}`")
            lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"TP={len(cases['TP'])}  FN={len(cases['FN'])}  FP={len(cases['FP'])}  TN={len(cases['TN'])}")
    print(f"Saved -> {out_success.relative_to(REPO_ROOT)}/")
    print(f"Saved -> {out_fn.relative_to(REPO_ROOT)}/")
    print(f"Saved -> {out_fp.relative_to(REPO_ROOT)}/")
    print(f"Saved -> {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
