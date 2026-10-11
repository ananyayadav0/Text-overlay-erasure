"""Metrics for one run directory, or for every run under results/.

    python src/evaluate.py results/main/sd15
    python src/evaluate.py --all

Quality is measured against the clean ground truth, split by the ground-truth mask M:
  *_in   only the erased region changes  (output inside M, clean outside)
  *_out  only the background changes     (clean inside M, output outside)
Erasure is measured with OCR / a QR decoder on the output:
  text_res  share of ground-truth violation characters OCR can still read (lower is better)
  qr_res    share of QR codes that still decode (lower is better)
  new_text  OCR strings in the output that are neither in the clean image nor ground-truth text
"""
import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_annotations, load_meta, load_sample, to_np  # noqa: E402
from ocr import box_iou, find_qr, poly_bbox, poly_center, read_text  # noqa: E402

CACHE = RESULTS / "_cache"
TEXT_TYPES = ["promo_claim", "rank_badge", "store_watermark", "contact_info", "exaggerated_effect"]


@lru_cache(maxsize=1)
def lpips_net():
    import lpips
    return lpips.LPIPS(net="alex", verbose=False).eval()


def lpips_dist(a, b):
    t = lambda im: torch.from_numpy(to_np(im)).permute(2, 0, 1)[None] * 2 - 1  # noqa: E731
    with torch.no_grad():
        return float(lpips_net()(t(a), t(b)))


def psnr_masked(a, b, m):
    if m.sum() == 0:
        return float("nan")
    mse = ((to_np(a) - to_np(b)) ** 2)[m].mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def char_recall(gt, found):
    g, f = norm(gt), norm(found)
    if not g:
        return 0.0
    blocks = SequenceMatcher(None, g, f, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(g)


def scaled_violations(ann, size):
    sx, sy = size[0] / ann["width"], size[1] / ann["height"]
    out = []
    for v in ann["violations"]:
        x0, y0, x1, y1 = v["bbox"]
        out.append({**v, "bbox": [x0 * sx - 4, y0 * sy - 4, x1 * sx + 4, y1 * sy + 4]})
    return out


def clean_ocr(pid, clean):
    p = CACHE / "clean_ocr" / f"{pid}.json"
    if p.exists():
        return json.loads(p.read_text())
    p.parent.mkdir(parents=True, exist_ok=True)
    dets = [{"bbox": poly_bbox(poly), "text": t, "conf": c} for poly, t, c in read_text(clean)]
    p.write_text(json.dumps(dets))
    return dets


def erasure_scores(out, ann, clean_dets):
    viol = scaled_violations(ann, out.size)
    dets = read_text(out)
    text_rec, qr_left, qr_total = [], 0, 0
    per_type = {t: [] for t in TEXT_TYPES}
    for v in viol:
        if v["type"] == "qr_code":
            continue
        x0, y0, x1, y1 = v["bbox"]
        inside = [t for poly, t, _ in dets if x0 <= poly_center(poly)[0] <= x1 and y0 <= poly_center(poly)[1] <= y1]
        rec = char_recall(v["text"], " ".join(inside))
        text_rec.append(rec)
        per_type[v["type"]].append(rec)
    qrs = [v for v in viol if v["type"] == "qr_code"]
    if qrs:
        decoded = {t for _, t in find_qr(out) if t}
        qr_total = len(qrs)
        qr_left = sum(v["text"] in decoded for v in qrs)
    gt_texts = [v["text"] for v in viol] + [v.get("caption", "") for v in viol if v.get("caption")]
    new = 0
    for poly, t, c in dets:
        if c < 0.4 or len(norm(t)) < 3:
            continue
        b = poly_bbox(poly)
        if any(box_iou(b, cd["bbox"]) > 0.3 for cd in clean_dets):
            continue
        # misreads of a planted string are residue, not new text: most of their characters appear in it in order
        if any(char_recall(t, g) >= 0.6 for g in gt_texts):
            continue
        new += 1
    return {
        "text_res": float(np.mean(text_rec)) if text_rec else float("nan"),
        "qr_res": qr_left / qr_total if qr_total else float("nan"),
        "new_text": new,
        **{f"tr_{t}": (float(np.mean(r)) if r else float("nan")) for t, r in per_type.items()},
    }


def evaluate_run(d, force=False):
    d = Path(d)
    meta = load_meta(d)
    imgs = sorted((d / "images").glob("*.png"))
    csv_path = d / "metrics.csv"
    if not force and csv_path.exists():
        with open(csv_path) as f:
            if sum(1 for _ in f) - 1 == len(imgs):
                return summarize(d)
    anns = load_annotations()
    rows = []
    for p in tqdm(imgs, desc=f"eval {d.parent.name}/{d.name}", leave=False):
        pid = p.stem
        clean, violated, gt = load_sample(pid)
        out = Image.open(p).convert("RGB").resize(clean.size)
        m = np.asarray(gt) > 0
        comp_in = Image.composite(out, clean, gt)
        comp_out = Image.composite(clean, out, gt)
        r = {
            "id": pid,
            "psnr_in": psnr_masked(out, clean, m),
            "psnr_out": psnr_masked(out, clean, ~m),
            "lpips_in": lpips_dist(comp_in, clean),
            "lpips_out": lpips_dist(comp_out, clean),
            "lpips": lpips_dist(out, clean),
            "ssim": float(structural_similarity(to_np(out), to_np(clean), channel_axis=2, data_range=1.0)),
            "mask_frac": float(m.mean()),
            "time": meta.get("times", {}).get(pid, float("nan")),
        }
        r.update(erasure_scores(out, anns[pid], clean_ocr(pid, clean)))
        rows.append(r)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return summarize(d)


def summarize(d):
    d = Path(d)
    with open(d / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    keys = [k for k in rows[0] if k != "id"]
    s = {k: float(np.nanmean([float(r[k]) for r in rows])) for k in keys}
    s["new_text_img"] = float(np.mean([float(r["new_text"]) > 0 for r in rows]))
    s["n"] = len(rows)
    s["config"] = load_meta(d).get("config", {})
    (d / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    runs = a.runs or sorted(p.parent for p in RESULTS.glob("*/*/images/") if not p.parts[-3].startswith("_"))
    for r in runs:
        s = evaluate_run(r, a.force)
        print(f"{r}: " + ", ".join(f"{k}={v:.4f}" for k, v in s.items() if isinstance(v, float)))


if __name__ == "__main__":
    main()
