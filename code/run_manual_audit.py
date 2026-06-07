"""Manual audit of FN / FP boundary cases against COCO GT.

For the 2000-image main experiment, COCO `instances_val2017` is known to have
systematic annotation gaps (small / accessory / non-human objects). Many
cases that the CHAIR-style metric counts as FN (system passed it but GT
absent) are in fact GT annotation gaps — the object IS visible. Similarly,
some FP cases (system flagged but GT present) may correspond to GT objects
that are essentially invisible.

This tool walks the user through every FN and FP case and asks ONE question
per case:

    "Is this object visibly present in the image?"

The verdict (visible / not-visible / borderline / skip) is recorded to
`outputs/metrics/manual_audit.csv`. The file is updated after each keystroke
so the user can quit and resume at any time.

Downstream: `run_audit_recompute.py` reads this CSV, recomputes confusion
matrix with augmented GT, and produces metrics with 95% bootstrap CIs.

Usage:
    python code/run_manual_audit.py

Keys (shown on-screen for every case):
    y  — object is visibly present
    n  — object is NOT visibly present
    m  — borderline / ambiguous
    s  — skip (re-show later)
    q  — quit and save
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

from data_utils import fetch_val2017_image


# --------------------------------------------------------------------------- #
# Paths (relative to repo root; the script can be run from anywhere).
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
GROUNDING_CSV = REPO / "outputs" / "grounding" / "large_strict_2000_grounding.csv"
CAPTION_CSV   = REPO / "outputs" / "captions" / "large_strict_caption_2000.csv"
COCO_GT       = REPO / "data" / "coco_annotations" / "instances_val2017.json"
RAW_IMG_DIR   = REPO / "data" / "coco_val"
VIZ_DIR_PREF  = [
    REPO / "outputs" / "visualizations" / "large_strict_2000",
    REPO / "outputs" / "visualizations" / "large_strict_1000",
    REPO / "outputs" / "visualizations" / "large_strict",
]
OUT_CSV       = REPO / "outputs" / "metrics" / "manual_audit.csv"


# --------------------------------------------------------------------------- #
# Korean font (Windows: Malgun Gothic). Falls back silently.
# --------------------------------------------------------------------------- #
def _pick_korean_font() -> str | None:
    for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if cand in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return cand
    return None


# --------------------------------------------------------------------------- #
# Data loaders.
# --------------------------------------------------------------------------- #
def load_coco_gt():
    with open(COCO_GT, "r", encoding="utf-8") as f:
        coco = json.load(f)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_gt: dict[int, set[str]] = {}
    for ann in coco["annotations"]:
        img_gt.setdefault(ann["image_id"], set()).add(cats[ann["category_id"]])
    return img_gt


def build_audit_queue() -> pd.DataFrame:
    """Return cases (FN + FP) joined with caption + COCO GT for display."""
    g = pd.read_csv(GROUNDING_CSV)
    caps = pd.read_csv(CAPTION_CSV).set_index("image_id")["caption"].to_dict()
    img_gt = load_coco_gt()

    g["in_gt"] = g.apply(
        lambda r: r["canonical"] in img_gt.get(int(r["image_id"]), set()),
        axis=1,
    )
    # FN: status=supported & not in GT
    fn = g[(g["status"] == "supported") & (~g["in_gt"])].copy()
    fn["case_type"] = "FN"
    # FP: status=unsupported & in GT
    fp = g[(g["status"] == "unsupported") & (g["in_gt"])].copy()
    fp["case_type"] = "FP"

    cases = pd.concat([fn, fp], ignore_index=True)
    cases["caption"]  = cases["image_id"].map(caps).fillna("")
    cases["coco_gt"]  = cases["image_id"].map(
        lambda i: ", ".join(sorted(img_gt.get(int(i), set()))) or "(empty)"
    )
    cases["row_key"]  = cases.apply(
        lambda r: f"{int(r['image_id'])}_{r['canonical']}_{r['case_type']}",
        axis=1,
    )
    return cases.reset_index(drop=True)


def load_existing_verdicts(out_csv: Path | None = None) -> dict[str, str]:
    path = out_csv or OUT_CSV
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["row_key"], df["verdict"]))


def append_verdict(row_key: str, image_id: int, canonical: str,
                   case_type: str, verdict: str,
                   out_csv: Path | None = None):
    """Append one verdict to the CSV (creates with header if missing)."""
    path = out_csv or OUT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{
        "row_key":    row_key,
        "image_id":   int(image_id),
        "canonical":  canonical,
        "case_type":  case_type,
        "verdict":    verdict,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    }])
    new.to_csv(path, mode="a", header=not path.exists(), index=False)


def overwrite_verdict(row_key: str, image_id: int, canonical: str,
                      case_type: str, verdict: str,
                      out_csv: Path | None = None):
    """Used when re-auditing a skipped case. Rewrites the CSV without that key."""
    path = out_csv or OUT_CSV
    if not path.exists():
        append_verdict(row_key, image_id, canonical, case_type, verdict,
                       out_csv=path)
        return
    df = pd.read_csv(path)
    df = df[df["row_key"] != row_key]
    df = pd.concat([df, pd.DataFrame([{
        "row_key":    row_key,
        "image_id":   int(image_id),
        "canonical":  canonical,
        "case_type":  case_type,
        "verdict":    verdict,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    }])], ignore_index=True)
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Image lookup.
# --------------------------------------------------------------------------- #
def find_viz_path(image_id: int) -> Path | None:
    fname = f"{int(image_id):012d}_verified.jpg"
    for d in VIZ_DIR_PREF:
        p = d / fname
        if p.exists():
            return p
    return None


def raw_image_path(image_id: int) -> Path:
    return RAW_IMG_DIR / f"{int(image_id):012d}.jpg"


def image_path(image_id: int) -> Path | None:
    viz = find_viz_path(image_id)
    if viz is not None:
        return viz

    raw = raw_image_path(image_id)
    if raw.exists():
        return raw

    return None


def download_missing_images(cases: pd.DataFrame) -> None:
    missing_ids = sorted({
        int(r.image_id)
        for r in cases.itertuples()
        if image_path(int(r.image_id)) is None
    })
    if not missing_ids:
        print("[download] no missing audit images.")
        return

    RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] missing audit images: {len(missing_ids)}")
    ok = 0
    for i, image_id in enumerate(missing_ids, start=1):
        path = fetch_val2017_image(image_id, RAW_IMG_DIR)
        if path is None:
            print(f"  [{i}/{len(missing_ids)}] failed: {image_id:012d}.jpg")
        else:
            ok += 1
            print(f"  [{i}/{len(missing_ids)}] ok: {path.name}")
    print(f"[download] downloaded {ok}/{len(missing_ids)} images.")


def case_context(case_type: str, status: str | None = None) -> tuple[str, str]:
    if case_type == "FN":
        return (
            "FN  (시스템: supported · GT: 없음)\n"
            "→ 'y' (보임) = GT 누락. 보정 대상 (TN 으로 이동)\n"
            "→ 'n' (안 보임) = 진짜 환각. real FN, Joint failure",
            "#B25A1B",
        )
    if case_type == "FP":
        return (
            "FP  (시스템: unsupported · GT: 있음)\n"
            "→ 'y' (보임) = 시스템 OD 진짜 놓침. real FP\n"
            "→ 'n' (안 보임) = GT 가 사실상 안 보이는 객체 라벨. 보정 대상",
            "#2F4A7C",
        )

    if case_type == "agg_in_gt":
        if status == "supported":
            return (
                "Aggressive-only  (시스템: supported · GT: 있음)\n"
                "→ 'y' (보임) = aggressive 매핑이 타당한 정상 케이스\n"
                "→ 'n' (안 보임) = GT/box/crop 기준상 식별 어려움",
                "#2F4A7C",
            )
        return (
            "Aggressive-only  (시스템: unsupported · GT: 있음)\n"
            "→ 'y' (보임) = 객체는 맞지만 시스템이 놓친 케이스\n"
            "→ 'n' (안 보임) = GT/box/crop 기준상 식별 어려움",
            "#2F4A7C",
        )

    if case_type == "agg_not_in_gt_supp":
        return (
            "Aggressive-only  (시스템: supported · GT: 없음)\n"
            "→ 'y' (보임) = COCO GT 누락/카테고리 mismatch 가능성\n"
            "→ 'n' (안 보임) = aggressive 매핑이 실제 환각을 만든 케이스",
            "#B25A1B",
        )

    if case_type == "agg_not_in_gt_unsupp":
        return (
            "Aggressive-only  (시스템: unsupported · GT: 없음)\n"
            "→ 'y' (보임) = 시스템은 막았지만 실제 객체는 보임\n"
            "→ 'n' (안 보임) = aggressive 매핑이 환각 후보를 추가한 케이스",
            "#B25A1B",
        )

    return (
        f"{case_type}\n"
        "→ 'y' (보임) = crop 기준으로 객체 식별 가능\n"
        "→ 'n' (안 보임) = crop 기준으로 객체 식별 불가",
        "#555555",
    )


# --------------------------------------------------------------------------- #
# Main interactive loop.
# --------------------------------------------------------------------------- #
class Auditor:
    def __init__(self, cases: pd.DataFrame, *,
                 redo_skips: bool = False,
                 redo_borderline: bool = False,
                 redo_all: bool = False,
                 out_csv: Path | None = None,
                 finish_message: str | None = None,
                 title_prefix: str = "Manual Audit"):
        self.cases = cases
        self.out_csv = out_csv or OUT_CSV
        self.verdicts = load_existing_verdicts(self.out_csv)
        self.redo_skips = redo_skips
        self.redo_borderline = redo_borderline
        self.redo_all = redo_all
        self.finish_message = finish_message or (
            "모든 케이스 audit 완료!\n\n"
            "다음 단계: python code/run_audit_recompute.py\n"
            "창을 닫고 [q] 또는 콘솔에서 Ctrl+C."
        )
        self.title_prefix = title_prefix
        self.idx = 0
        self.fig = None
        self.ui_font: str | None = None
        self.last_action: str | None = None
        self.skipped_keys: list[str] = []

    def _to_audit(self) -> pd.DataFrame:
        """Cases still needing a verdict."""
        if self.redo_all:
            return self.cases.reset_index(drop=True)
        revisit_statuses: set[str] = set()
        if self.redo_skips:
            revisit_statuses.add("skip")
        if self.redo_borderline:
            revisit_statuses.add("borderline")
        done_keys = {k for k, v in self.verdicts.items()
                     if v not in revisit_statuses}
        return self.cases[~self.cases["row_key"].isin(done_keys)].reset_index(drop=True)

    def run(self):
        plt.rcParams["toolbar"] = "None"
        self.ui_font = _pick_korean_font()
        queue = self._to_audit()
        if queue.empty:
            print(f"\nAll {len(self.cases)} cases already audited. "
                  f"Re-run with --redo-skips to revisit skipped cases.\n")
            self._print_summary()
            return

        print(f"\n=== Manual Audit ===")
        print(f"Cases to review: {len(queue)} "
              f"(FN={int((queue['case_type'] == 'FN').sum())}, "
              f"FP={int((queue['case_type'] == 'FP').sum())})")
        print(f"Already done:    {len(self.verdicts)} / {len(self.cases)}")
        print(f"Output CSV:      {OUT_CSV}\n")
        print("Keys: y=visible  n=not-visible  m=borderline  s=skip  q=quit\n")

        self.fig = plt.figure(figsize=(11.5, 7.2))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.queue = queue
        self.idx = 0
        self._render()
        plt.show()

        self._print_summary()

    def _render(self):
        if self.idx >= len(self.queue):
            self._finish_message()
            return
        row = self.queue.iloc[self.idx]

        self.fig.clear()
        # Two panels: left = image, right = metadata + key guide.
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.05)
        ax_img  = self.fig.add_subplot(gs[0, 0])
        ax_info = self.fig.add_subplot(gs[0, 1])
        ax_img.axis("off")
        ax_info.axis("off")

        # ---- image ----
        img_path = image_path(int(row["image_id"]))
        if img_path is None:
            missing = raw_image_path(int(row["image_id"]))
            ax_img.text(
                0.5, 0.5,
                "[image missing]\n\n"
                f"{missing.name}\n\n"
                "이미지를 내려받은 뒤 다시 실행하거나\n"
                "[s]로 skip 하세요.",
                transform=ax_img.transAxes,
                ha="center", va="center", color="#A00000",
                fontsize=12, wrap=True,
            )
        else:
            try:
                img = mpimg.imread(str(img_path))
                ax_img.imshow(img)
                ax_img.set_title(
                    f"{img_path.name}    ({self.idx + 1}/{len(self.queue)})",
                    fontsize=10,
                )
            except Exception as e:
                ax_img.text(
                    0.5, 0.5,
                    "[image load failed]\n\n"
                    f"{img_path.name}\n\n{e}",
                    transform=ax_img.transAxes,
                    ha="center", va="center", color="#A00000",
                    fontsize=12, wrap=True,
                )

        # ---- info panel ----
        case_type = row["case_type"]
        cap       = row["caption"]
        canon     = row["canonical"]
        gt        = row["coco_gt"]
        status    = row["status"] if "status" in row.index else None
        ctx, ctx_color = case_context(case_type, status)

        prior = self.verdicts.get(row["row_key"])
        prior_str = f"  (이전 verdict: {prior})" if prior else ""

        lines = [
            (f"Case {self.idx + 1} / {len(self.queue)}    "
             f"({case_type}){prior_str}", 12, "bold", "black"),
            ("", 6, "normal", "black"),
            (f"image_id : {int(row['image_id'])}", 10, "normal", "black"),
            (f"canonical: ", 10, "normal", "black"),
            (f"   ▶ {canon} ◀", 14, "bold", "#A00000"),
            ("", 6, "normal", "black"),
            (f"caption  : {cap}", 9, "italic", "#333333"),
            ("", 4, "normal", "black"),
            (f"COCO GT  : {gt}", 9, "normal", "#444444"),
            ("", 6, "normal", "black"),
            ("질문 (crop test):", 11, "bold", "black"),
            (f"  '{canon}' 영역만 잘라서 모르는 사람에게 보여줘도",
             10, "normal", "black"),
            (f"  그 사람이 '{canon}' 라고 식별할 수 있는가?",
             10, "normal", "black"),
            ("  → 그렇다: y    영역 내 픽셀만으로 식별 불가: n", 9, "italic", "#555555"),
            ("  → 영역만 보면 애매: m", 9, "italic", "#555555"),
            ("", 6, "normal", "black"),
            ("Context (현재 케이스 유형):", 10, "bold", ctx_color),
        ]

        y = 0.97
        for text, size, weight, color in lines:
            style = "italic" if weight == "italic" else "normal"
            weight = "normal" if weight == "italic" else weight
            ax_info.text(0.02, y, text, transform=ax_info.transAxes,
                         fontsize=size, fontweight=weight, fontstyle=style, color=color,
                         va="top", ha="left", wrap=True)
            y -= 0.045 if text else 0.020

        # context block (multi-line)
        ax_info.text(0.02, y - 0.005, ctx, transform=ax_info.transAxes,
                     fontsize=10, color=ctx_color, va="top",
                     fontfamily=self.ui_font)
        y -= 0.18

        # keys
        keys_block = (
            "  [y]  보임  (visible)\n"
            "  [n]  안 보임  (not visible)\n"
            "  [m]  애매함  (borderline)\n"
            "  [s]  skip — 나중에 다시 보기\n"
            "  [q]  저장하고 종료"
        )
        ax_info.text(0.02, y, "키 안내:", transform=ax_info.transAxes,
                     fontsize=10, fontweight="bold", va="top")
        ax_info.text(0.02, y - 0.05, keys_block, transform=ax_info.transAxes,
                     fontsize=11, va="top", fontfamily=self.ui_font,
                     color="#222222")

        # progress
        if self.redo_all:
            done = self.idx
            total = len(self.queue)
            label = "이번 세션"
        else:
            done = len(self.verdicts) + self.idx
            total = len(self.cases)
            label = "진행"
        ax_info.text(0.02, 0.02,
                     f"{label}: {done}/{total}  "
                     f"({100.0 * done / total:.1f}%)" if total else "",
                     transform=ax_info.transAxes,
                     fontsize=9, color="#555555")

        if self.last_action:
            ax_info.text(0.98, 0.02, f"직전: {self.last_action}",
                         transform=ax_info.transAxes,
                         fontsize=9, color="#555555", ha="right")

        self.fig.canvas.draw_idle()

    def _record(self, verdict: str):
        row = self.queue.iloc[self.idx]
        rk  = row["row_key"]
        existing = self.verdicts.get(rk)
        if existing is not None:
            overwrite_verdict(rk, int(row["image_id"]), row["canonical"],
                              row["case_type"], verdict,
                              out_csv=self.out_csv)
        else:
            append_verdict(rk, int(row["image_id"]), row["canonical"],
                           row["case_type"], verdict,
                           out_csv=self.out_csv)
        self.verdicts[rk] = verdict
        self.last_action = f"{row['canonical']} ({row['case_type']}) → {verdict}"
        self.idx += 1
        self._render()

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "y":
            self._record("visible")
        elif k == "n":
            self._record("not_visible")
        elif k == "m":
            self._record("borderline")
        elif k == "s":
            self._record("skip")
        elif k == "q":
            plt.close(self.fig)
        # any other key: ignore

    def _on_close(self, event):
        pass  # CSV is already up to date

    def _finish_message(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, self.finish_message,
                ha="center", va="center", fontsize=14, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _print_summary(self):
        if not self.verdicts:
            return
        df = pd.read_csv(self.out_csv)
        counts = df.groupby(["case_type", "verdict"]).size().unstack(fill_value=0)
        print("\n--- audit summary ---")
        print(counts.to_string())
        print(f"\nTotal recorded: {len(df)} / {len(self.cases)}")
        print(f"CSV: {self.out_csv}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo-skips", action="store_true",
                        help="re-show previously skipped cases")
    parser.add_argument("--redo-borderline", action="store_true",
                        help="re-show previously borderline cases to finalize a verdict")
    parser.add_argument("--redo-all", action="store_true",
                        help="re-show ALL cases (re-audit existing verdicts under a "
                             "new criterion; existing verdicts are overwritten)")
    parser.add_argument("--download-missing", action="store_true",
                        help="download missing COCO val2017 images used by audit cases")
    args = parser.parse_args()

    if not GROUNDING_CSV.exists():
        sys.exit(f"missing: {GROUNDING_CSV}")
    if not COCO_GT.exists():
        sys.exit(f"missing: {COCO_GT}")

    cases = build_audit_queue()
    print(f"FN cases: {(cases['case_type'] == 'FN').sum()}")
    print(f"FP cases: {(cases['case_type'] == 'FP').sum()}")

    if args.download_missing:
        download_missing_images(cases)

    Auditor(cases,
            redo_skips=args.redo_skips,
            redo_borderline=args.redo_borderline,
            redo_all=args.redo_all).run()


if __name__ == "__main__":
    main()
