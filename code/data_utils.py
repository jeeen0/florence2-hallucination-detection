"""COCO val2017 image fetching utilities (no annotations required)."""
from __future__ import annotations

import random
import urllib.error
import urllib.request
from pathlib import Path

# Verified COCO val2017 image IDs (some may 404 → skipped).
CANDIDATE_IDS = [
    139, 285, 632, 724, 776, 802, 872, 885, 1000, 1268,
    1296, 1353, 1425, 1503, 1532, 1584, 1675, 1761, 1818, 2006,
    2149, 2153, 2157, 2261, 2299, 2431, 2473, 2532, 2587, 2592,
    2685, 2923, 3156, 3255, 3501, 3553, 3845, 4134, 4395, 4495,
    4765, 5001, 5037, 5060, 5503, 5529, 5586, 5802, 5993, 6012,
    6040, 6213, 6471, 6614, 6818, 7088, 7281, 7386, 7574, 7795,
    7977, 8021, 8211, 8277, 8532, 8629, 8762, 8844, 8899, 9298,
    9378, 9400, 9483, 9590, 9769, 9854, 9891, 9914, 10092, 10363,
    10764, 10977, 11051, 11122, 11197, 11511, 11760, 11813, 11987, 12062,
    12120, 12280, 12576, 12639, 12670, 12748, 12784, 12993, 13004, 13177,
]


def fetch_val2017_image(img_id: int, target_dir: Path) -> Path | None:
    fname = f"{img_id:012d}.jpg"
    out_path = target_dir / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    url = f"http://images.cocodataset.org/val2017/{fname}"
    try:
        urllib.request.urlretrieve(url, out_path)
        return out_path
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        if out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        return None


def collect_val2017_images(
    n: int,
    target_dir: Path,
    seed: int = 42,
) -> list[tuple[int, Path]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    candidates = list(CANDIDATE_IDS)
    rng.shuffle(candidates)
    collected: list[tuple[int, Path]] = []
    for cid in candidates:
        if len(collected) >= n:
            break
        path = fetch_val2017_image(cid, target_dir)
        if path is not None:
            collected.append((cid, path))
    return collected
