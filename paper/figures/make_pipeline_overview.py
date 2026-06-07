"""
make_pipeline_overview.py
=========================

Generates `pipeline_overview.png` (and `.pdf`) — the Florence-2
caption-detection consistency verification pipeline figure for the IPIU2026
paper (§3.1 in `paper/draft_ko.md`).

Layout is a Y-shape that visually conveys the central architectural point:
`<CAPTION>` and `<OD>` are two *parallel* prompt calls on the same image
that converge at the matching stage.

Run:
    python paper/figures/make_pipeline_overview.py

Dependencies: matplotlib (already in the project env).

Suggested figure caption (Korean) for the paper:
    그림 1. Florence-2 단일 모델 기반 caption-detection consistency 검증
    파이프라인. <CAPTION> 으로 추출된 객체 mention 을 동일 이미지에 대한
    <OD> 결과와 strict synonym 매핑 하에 매칭하여 supported / unsupported
    로 분류한다.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _try_korean_font() -> bool:
    from matplotlib import font_manager
    candidates = ["Malgun Gothic", "MalgunGothic", "NanumGothic", "Nanum Gothic",
                  "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


USE_KOREAN = _try_korean_font()


# (kind, primary Korean label, English sub-label, prompt/short hint or None)
# Bottom row labels are kept short to avoid the overflow that occurred in the
# earlier U-shaped layout (Korean+English combined exceeded box width).
NODES = {
    "input":    ("input",    "입력 이미지",   "Input Image",        "COCO val2017"),
    "caption":  ("florence", "캡션 생성",     "Caption Generation", "<CAPTION>"),
    "extract":  ("logic",    "Mention 추출",  "Mention Extraction", "COCO80 + strict"),
    "od":       ("florence", "객체 검출",     "Object Detection",   "<OD>"),
    "match":    ("logic",    "매칭 & 분류",   "Match & Classify",   "canonical label"),
    "output":   ("output",   "검증된 캡션",   "Verified Caption",   "supported only"),
}

COLOR = {
    "input":    {"fc": "#E8EEF7", "ec": "#2F4A7C"},
    "florence": {"fc": "#FDE9D9", "ec": "#B25A1B"},
    "logic":    {"fc": "#FFFFFF", "ec": "#444444"},
    "output":   {"fc": "#E2F0E2", "ec": "#2F6B2F"},
}

BOX_W = 2.6
BOX_H = 1.30

# Y-shape positions (center x, center y) on a 16 × 7 canvas.
POS = {
    "input":   (1.7,  3.5),
    "caption": (5.4,  5.4),
    "extract": (8.9,  5.4),
    "od":      (5.4,  1.6),
    "match":   (12.2, 3.5),
    "output":  (14.9, 3.5),
}


def draw_stage(ax, key):
    kind, primary_ko, primary_en, sub = NODES[key]
    cx, cy = POS[key]
    style = COLOR[kind]

    box = FancyBboxPatch(
        (cx - BOX_W / 2, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=style["ec"], facecolor=style["fc"], zorder=2,
    )
    ax.add_patch(box)

    primary = primary_ko if USE_KOREAN else primary_en
    ax.text(cx, cy + 0.22, primary,
            ha="center", va="center", fontsize=9.0, fontweight="bold",
            color="black", zorder=3)
    if USE_KOREAN:
        ax.text(cx, cy - 0.02, primary_en,
                ha="center", va="center", fontsize=6.8, fontstyle="italic",
                color="#222222", zorder=3)
    if sub:
        ax.text(cx, cy - 0.30, sub,
                ha="center", va="center", fontsize=6.4, color="#333333",
                family="monospace" if sub.startswith("<") else None,
                zorder=3)


def draw_arrow(ax, src_key, dst_key, *, src_side=None, dst_side=None):
    """Arrow from edge of src box to edge of dst box.

    src_side / dst_side may be 'r','l','t','b'. If None, auto-pick based on
    relative position.
    """
    sx, sy = POS[src_key]
    dx, dy = POS[dst_key]

    def edge(cx, cy, side):
        if side == "r": return (cx + BOX_W / 2, cy)
        if side == "l": return (cx - BOX_W / 2, cy)
        if side == "t": return (cx, cy + BOX_H / 2)
        if side == "b": return (cx, cy - BOX_H / 2)
        # auto
        if dx > cx + 0.1: return (cx + BOX_W / 2, cy) if abs(dy - cy) < 0.5 else None
        if dx < cx - 0.1: return (cx - BOX_W / 2, cy) if abs(dy - cy) < 0.5 else None
        return None

    if src_side is None or dst_side is None:
        # auto pick based on relative positions
        dxdir = "r" if dx > sx else ("l" if dx < sx else "c")
        dydir = "t" if dy > sy else ("b" if dy < sy else "c")
        if dydir == "c":
            src_side, dst_side = "r" if dxdir == "r" else "l", "l" if dxdir == "r" else "r"
        elif dxdir == "c":
            src_side, dst_side = dydir, "b" if dydir == "t" else "t"
        else:
            # diagonal — exit from horizontal edge, enter at vertical edge
            src_side = "r" if dxdir == "r" else "l"
            dst_side = "b" if dydir == "t" else "t"

    s = {"r": (sx + BOX_W / 2, sy), "l": (sx - BOX_W / 2, sy),
         "t": (sx, sy + BOX_H / 2), "b": (sx, sy - BOX_H / 2)}[src_side]
    d = {"r": (dx + BOX_W / 2, dy), "l": (dx - BOX_W / 2, dy),
         "t": (dx, dy + BOX_H / 2), "b": (dx, dy - BOX_H / 2)}[dst_side]

    arrow = FancyArrowPatch(
        s, d, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color="#222222", zorder=1,
    )
    ax.add_patch(arrow)


def build_figure():
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=300)
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    for k in NODES:
        draw_stage(ax, k)

    # Input branches up to caption and down to OD (parallel calls).
    draw_arrow(ax, "input",   "caption", src_side="r", dst_side="l")
    draw_arrow(ax, "input",   "od",      src_side="r", dst_side="l")
    # Top branch continues right to mention extraction.
    draw_arrow(ax, "caption", "extract", src_side="r", dst_side="l")
    # Both branches converge into match & classify.
    draw_arrow(ax, "extract", "match",   src_side="r", dst_side="l")
    draw_arrow(ax, "od",      "match",   src_side="r", dst_side="l")
    # Final output.
    draw_arrow(ax, "match",   "output",  src_side="r", dst_side="l")

    legend_handles = [
        mpatches.Patch(
            facecolor=COLOR["florence"]["fc"], edgecolor=COLOR["florence"]["ec"],
            label=("Florence-2 호출" if USE_KOREAN else "Florence-2 call"),
        ),
        mpatches.Patch(
            facecolor=COLOR["logic"]["fc"], edgecolor=COLOR["logic"]["ec"],
            label=("로직 / 데이터" if USE_KOREAN else "Logic / Data"),
        ),
        mpatches.Patch(
            facecolor=COLOR["input"]["fc"], edgecolor=COLOR["input"]["ec"],
            label=("입력" if USE_KOREAN else "Input"),
        ),
        mpatches.Patch(
            facecolor=COLOR["output"]["fc"], edgecolor=COLOR["output"]["ec"],
            label=("출력" if USE_KOREAN else "Output"),
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.04), ncol=4,
        fontsize=7.5, frameon=False,
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.10)
    return fig


def main():
    out_dir = Path(__file__).resolve().parent
    png_path = out_dir / "pipeline_overview.png"
    pdf_path = out_dir / "pipeline_overview.pdf"

    fig = build_figure()
    fig.savefig(png_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    try:
        fig.savefig(pdf_path, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    except Exception as e:
        print(f"[warn] PDF save failed ({e!r}); PNG was saved.", file=sys.stderr)
    plt.close(fig)

    size = os.path.getsize(png_path)
    print(f"saved: {png_path}  ({size} bytes)")
    if pdf_path.exists():
        print(f"saved: {pdf_path}  ({os.path.getsize(pdf_path)} bytes)")
    print(f"Korean font used: {USE_KOREAN}")
    if size < 20_000:
        raise SystemExit(f"PNG too small ({size} bytes) — likely empty plot")


if __name__ == "__main__":
    main()
