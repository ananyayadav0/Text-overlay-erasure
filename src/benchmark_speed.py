"""Seconds per image for every method on ONE machine, same 10 test images (the report's speed table).

    python src/benchmark_speed.py                       # all methods available on this device
    python src/benchmark_speed.py --methods lama sd15   # a subset
Writes results/speed/<device>.json.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, dilate, load_sample, splits  # noqa: E402
from methods import DEVICE, apply  # noqa: E402

ALL = ["telea", "lama", "sd15", "hybrid", "sdxl", "ip2p", "kontext"]


def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if DEVICE == "mps":
        return platform.processor() or "Apple MPS"
    return "CPU"


def free_models():
    import gc
    import methods
    for f in ("_lama", "_sd15", "_sdxl", "_ip2p", "_kontext"):
        getattr(methods, f).cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=ALL)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    ids = splits()["test"][:a.n + 1]           # first image is a warm-up (model load, kernels)
    hp = RESULTS / "selected_hparams.json"
    h = json.loads(hp.read_text()) if hp.exists() else {"oracle_dilate": 4, "hybrid_strength": 0.5}
    d = RESULTS / "speed"
    d.mkdir(parents=True, exist_ok=True)
    path = d / (device_name().replace(" ", "_").replace("/", "_") + ".json")
    out = json.loads(path.read_text()) if path.exists() else {"device": device_name(), "seconds_per_image": {}, "n": {}}
    for m in a.methods:
        if m == "kontext" and not torch.cuda.is_available():
            print("skip kontext (CUDA only)")
            continue
        params = {"strength": h["hybrid_strength"]} if m == "hybrid" else {}
        ts = []
        for i, pid in enumerate(ids):
            _, violated, gt = load_sample(pid)
            mask = dilate(gt, h["oracle_dilate"])
            sync()
            t = time.time()
            apply(m, violated, mask, **params)
            sync()
            if i:
                ts.append(time.time() - t)
        free_models()                                   # each model alone on the GPU (Kontext needs ~33 GB)
        out["seconds_per_image"][m] = sum(ts) / len(ts)
        out["n"][m] = len(ts)
        print(f"{m:8s} {out['seconds_per_image'][m]:.2f} s/img", flush=True)
        path.write_text(json.dumps(out, indent=2))     # merged per method, so runs can be split up


if __name__ == "__main__":
    main()
