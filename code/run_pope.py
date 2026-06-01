"""POPE benchmark evaluation for Florence-2.

POPE (Polling-based Object Probing Evaluation) asks yes/no questions of the form
"Is there a <object> in the image?" for COCO val2014 images.

Three splits are evaluated independently:
- random:        negative objects drawn uniformly from COCO80
- popular:       negative objects biased toward common categories
- adversarial:   negative objects biased toward objects co-occurring with the
                 image's positive ones

Pipeline:
1. Download 500 unique val2014 images referenced by POPE.
2. Run Florence-2 <OD> on each image once. Cache raw detections to JSON.
3. For each (image, object_name, label) question, predict "yes" if the OD
   canonical labels contain object_name; otherwise "no".
4. Compute accuracy, precision, recall, F1 per split.

Outputs:
  outputs/pope/od_cache.json
  outputs/pope/pope_per_question.csv
  outputs/pope/pope_metrics.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from florence2 import Florence2Runner
from modules.verify import detect_objects


REPO_ROOT = Path(__file__).resolve().parents[1]
POPE_DIR = REPO_ROOT / "data" / "pope"
SPLITS = ("random", "popular", "adversarial")


def _fetch_val2014(filename: str, target_dir: Path) -> Path | None:
    out = target_dir / filename
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"http://images.cocodataset.org/val2014/{filename}"
    try:
        urllib.request.urlretrieve(url, out)
        return out
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass
        return None


def _parse_object(text: str) -> str | None:
    m = re.match(r"Is there a (.+?) in the image\?", text)
    return m.group(1).lower().strip() if m else None


def _load_split(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="microsoft/Florence-2-large-ft")
    ap.add_argument("--vocab", choices=["strict", "aggressive"], default="strict")
    ap.add_argument("--pope-dir", type=Path, default=POPE_DIR)
    ap.add_argument("--image-dir", type=Path,
                    default=REPO_ROOT / "data" / "coco_val2014_pope")
    ap.add_argument("--od-cache", type=Path,
                    default=REPO_ROOT / "outputs" / "pope" / "od_cache.json")
    ap.add_argument("--out-per-q", type=Path,
                    default=REPO_ROOT / "outputs" / "pope" / "pope_per_question.csv")
    ap.add_argument("--out-metrics", type=Path,
                    default=REPO_ROOT / "outputs" / "pope" / "pope_metrics.json")
    args = ap.parse_args()

    splits: dict[str, list[dict]] = {}
    for s in SPLITS:
        path = args.pope_dir / f"coco_pope_{s}.json"
        splits[s] = _load_split(path)

    all_questions: list[dict] = []
    for s, rows in splits.items():
        for r in rows:
            obj = _parse_object(r["text"])
            if obj is None:
                continue
            all_questions.append({
                "split": s,
                "question_id": r["question_id"],
                "image": r["image"],
                "object": obj,
                "label": r["label"].lower(),
            })

    images = sorted({q["image"] for q in all_questions})
    print(f"[pope] {len(all_questions)} questions across 3 splits, {len(images)} unique images")

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.od_cache.parent.mkdir(parents=True, exist_ok=True)
    args.out_per_q.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pope] downloading missing val2014 images to {args.image_dir}")
    paths: dict[str, Path] = {}
    for i, fname in enumerate(images, 1):
        p = _fetch_val2014(fname, args.image_dir)
        if p is None:
            print(f"  [{i}/{len(images)}] MISS {fname}")
            continue
        paths[fname] = p
        if i % 50 == 0:
            print(f"  [{i}/{len(images)}] downloaded")
    print(f"[pope] {len(paths)}/{len(images)} images available")

    od_cache: dict[str, list[dict]] = {}
    if args.od_cache.exists():
        od_cache = json.loads(args.od_cache.read_text(encoding="utf-8"))
        print(f"[pope] loaded OD cache: {len(od_cache)} entries")

    todo = [(f, p) for f, p in paths.items() if f not in od_cache]
    if todo:
        print(f"[pope] running <OD> on {len(todo)} new images")
        runner = Florence2Runner(model_id=args.model)
        for i, (fname, path) in enumerate(todo, 1):
            image = Image.open(path).convert("RGB")
            dets = detect_objects(runner, image, vocab=args.vocab)
            od_cache[fname] = [
                {"label": d.label, "canonical": d.canonical, "bbox": list(d.bbox)}
                for d in dets
            ]
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {fname}")
        args.od_cache.write_text(json.dumps(od_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[pope] OD cache saved -> {args.od_cache}")

    per_q_rows: list[tuple] = []
    cm_by_split: dict[str, dict[str, int]] = {s: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for s in SPLITS}

    for q in all_questions:
        if q["image"] not in od_cache:
            continue
        cans = {d.get("canonical") for d in od_cache[q["image"]]}
        cans.discard(None)
        pred = "yes" if q["object"] in cans else "no"
        gt = q["label"]
        per_q_rows.append((q["split"], q["question_id"], q["image"], q["object"], gt, pred))

        # POPE convention: positive class is "yes" (object truly present)
        if pred == "yes" and gt == "yes":
            cm_by_split[q["split"]]["TP"] += 1
        elif pred == "yes" and gt == "no":
            cm_by_split[q["split"]]["FP"] += 1
        elif pred == "no" and gt == "yes":
            cm_by_split[q["split"]]["FN"] += 1
        else:
            cm_by_split[q["split"]]["TN"] += 1

    def _metrics(cm: dict[str, int]) -> dict[str, float]:
        total = sum(cm.values())
        acc = (cm["TP"] + cm["TN"]) / total if total else 0.0
        p = cm["TP"] / (cm["TP"] + cm["FP"]) if cm["TP"] + cm["FP"] else 0.0
        r = cm["TP"] / (cm["TP"] + cm["FN"]) if cm["TP"] + cm["FN"] else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        yes_rate = (cm["TP"] + cm["FP"]) / total if total else 0.0
        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "yes_ratio": yes_rate,
            **cm,
        }

    summary = {s: _metrics(cm_by_split[s]) for s in SPLITS}

    args.out_metrics.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.out_per_q.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "question_id", "image", "object", "label", "pred"])
        for row in per_q_rows:
            w.writerow(row)

    print(f"\nSaved -> {args.out_metrics}")
    print(f"Saved -> {args.out_per_q}")
    print()
    for s in SPLITS:
        m = summary[s]
        print(f"  {s:13s}  acc={m['accuracy']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}  yes-ratio={m['yes_ratio']:.4f}")


if __name__ == "__main__":
    main()
