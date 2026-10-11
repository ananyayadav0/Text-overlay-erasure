"""Shared data loading, splits, mask handling and run I/O."""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
RESULTS = ROOT / "results"
WORK_SIZE = 512          # long side used for every method and every metric
SPLIT_SEED = 6127
N_DEV, N_TEST = 40, 160  # dev: hyper-parameter sweeps, test: final comparison


def load_annotations():
    with open(DATA / "annotations.jsonl") as f:
        return {d["id"]: d for d in map(json.loads, f)}


def splits():
    """Frozen split from dataset/splits.json (written by finalize_dataset.py); seeded shuffle otherwise."""
    p = DATA / "splits.json"
    if p.exists():
        return json.loads(p.read_text())
    ids = sorted(load_annotations())
    random.Random(SPLIT_SEED).shuffle(ids)
    return {"dev": ids[:N_DEV], "test": ids[N_DEV:N_DEV + N_TEST]}


def work_size(w, h, long_side=WORK_SIZE, multiple=8):
    s = long_side / max(w, h)
    return (max(multiple, round(w * s / multiple) * multiple),
            max(multiple, round(h * s / multiple) * multiple))


def load_sample(pid, long_side=WORK_SIZE):
    """clean, violated (RGB PIL) and GT mask (L, 0/255) at working resolution."""
    clean = Image.open(DATA / "clean" / f"{pid}.jpg").convert("RGB")
    size = work_size(*clean.size, long_side)
    violated = Image.open(DATA / "violated" / f"{pid}.jpg").convert("RGB").resize(size, Image.LANCZOS)
    mask = Image.open(DATA / "mask" / f"{pid}.png").convert("L").resize(size, Image.BOX)
    mask = mask.point(lambda v: 255 if v > 0 else 0)
    return clean.resize(size, Image.LANCZOS), violated, mask


def dilate(mask, px):
    """Grow a binary mask by px pixels (square structuring element)."""
    if px <= 0:
        return mask
    return mask.filter(ImageFilter.MaxFilter(2 * px + 1))


def paste_back(original, generated, mask, feather=0):
    """Keep original pixels outside the mask; optional Gaussian feather on the seam."""
    m = mask
    if feather > 0:
        m = mask.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(generated.resize(original.size, Image.LANCZOS), original, m)


def to_np(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def run_dir(exp, run):
    d = RESULTS / exp / run
    (d / "images").mkdir(parents=True, exist_ok=True)
    return d


def save_meta(d, meta):
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    tmp.replace(d / "meta.json")


def load_meta(d):
    p = Path(d) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}
