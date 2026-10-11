"""Run one method configuration over a split and save outputs (resumable).

    python src/run_method.py --exp main --run sd15 --method sd15 --split test --mask gt --dilate 8
    python src/run_method.py --exp sweep_cfg --run g3 --method sd15 --split dev --param guidance=3
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import dilate, load_meta, load_sample, run_dir, save_meta, splits  # noqa: E402


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None, "": ""}.get(v.lower(), v)


def get_mask(source, pid, gt):
    if source == "gt":
        return gt
    if source == "none":
        return Image.new("L", gt.size, 0)
    raise ValueError(source)


def run(exp, run_name, method, split="test", mask="gt", dilate_px=8, paste=True, feather=0,
        limit=None, **params):
    from methods import apply
    d = run_dir(exp, run_name)
    ids = splits()[split][:limit]
    meta = load_meta(d)
    times = meta.get("times", {})
    cfg = {"method": method, "split": split, "mask": mask, "dilate": dilate_px, "paste": paste,
           "feather": feather, "params": params}
    for pid in tqdm(ids, desc=f"{exp}/{run_name}", leave=False):
        out_path = d / "images" / f"{pid}.png"
        if out_path.exists():
            continue
        _, violated, gt = load_sample(pid)
        m = dilate(get_mask(mask, pid, gt), dilate_px)
        t = time.time()
        out = apply(method, violated, m, paste=paste, feather=feather, **params)
        times[pid] = time.time() - t
        tmp = out_path.parent / (out_path.name + ".tmp")
        out.save(tmp, format="PNG")
        tmp.replace(out_path)          # atomic: a shutdown never leaves a half-written image
        save_meta(d, {"config": cfg, "ids": ids, "times": times})
    save_meta(d, {"config": cfg, "ids": ids, "times": times})
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mask", default="gt", choices=["gt", "none"])
    ap.add_argument("--dilate", type=int, default=8)
    ap.add_argument("--no-paste", action="store_true")
    ap.add_argument("--feather", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--param", action="append", default=[], help="key=value passed to the method")
    a = ap.parse_args()
    params = dict(kv.split("=", 1) for kv in a.param)
    params = {k: parse_value(v) for k, v in params.items()}
    run(a.exp, a.run, a.method, a.split, a.mask, a.dilate, not a.no_paste, a.feather, a.limit, **params)


if __name__ == "__main__":
    main()
