"""Every experiment in the report, in execution order. Resumable; re-run to continue.

    python src/run_all.py --phase dev      # sweeps + ablations on dev (40 imgs)
    python src/run_all.py --phase test     # main comparison on test (160 imgs)
    python src/run_all.py --phase sdxl     # SDXL-Inpaint on test
    python src/run_all.py --phase kontext  # FLUX.1 Kontext: guidance sweep (dev) + test; CUDA GPU >= 20 GB, HF token
    python src/run_all.py --phase all      # dev, test, sdxl (kontext is opt-in)

Hyper-parameters are not hand-picked: they are selected on dev by the rules in the report (Sec. 3.3) and
written to results/selected_hparams.json, which the report quotes.
"""
import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS  # noqa: E402
from run_method import run  # noqa: E402

FALLBACK = {"oracle_dilate": 4, "hybrid_strength": 0.5}
HPARAMS = RESULTS / "selected_hparams.json"


# (exp, run, method, split, mask, dilate, paste, feather, limit, params)
def R(exp, run_name, method, split="dev", mask="gt", dilate=8, paste=True, feather=0, limit=None, **params):
    return dict(exp=exp, run_name=run_name, method=method, split=split, mask=mask, dilate_px=dilate,
                paste=paste, feather=feather, limit=limit, **params)


# ------------------------------------------------------------------ dev: sweeps and ablations
# defaults while sweeping one factor: dilation 8, 30 steps, guidance 7.5, seed 0
DEV = [
    *[R("sweep_dilation", f"lama_d{d}", "lama", dilate=d) for d in (0, 4, 8, 16, 32)],
    R("dev_ref", "identity", "identity"),
    R("dev_ref", "telea", "telea"),
    *[R("sweep_guidance", f"g{g}", "sd15", guidance=g) for g in (1, 3, 5, 7.5, 12)],
    *[R("sweep_steps", f"s{s}", "sd15", steps=s) for s in (10, 20, 30, 50)],
    *[R("sweep_dilation", f"sd15_d{d}", "sd15", dilate=d) for d in (0, 4, 8, 16, 32)],
    R("ablation", "default", "sd15"),
    R("ablation", "no_paste", "sd15", paste=False),
    R("ablation", "no_negative", "sd15", negative=""),
    R("ablation", "empty_prompt", "sd15", prompt=""),
    R("ablation", "feather4", "sd15", feather=4),
    *[R("sweep_strength", f"hy{s}", "hybrid", strength=s) for s in (0.3, 0.5, 0.7, 0.9)],
    *[R("sweep_ip2p_img", f"ig{ig}", "ip2p", image_guidance=ig) for ig in (1.0, 1.25, 1.5, 2.0)],
    *[R("sweep_ip2p_txt", f"g{g}", "ip2p", guidance=g) for g in (3, 7.5, 12)],
]

# identical configurations share outputs instead of being generated twice
ALIASES = {
    ("sweep_steps", "s30"): ("sweep_guidance", "g7.5"),
    ("sweep_dilation", "sd15_d8"): ("sweep_guidance", "g7.5"),
    ("ablation", "default"): ("sweep_guidance", "g7.5"),
    ("sweep_ip2p_img", "ig1.5"): ("sweep_ip2p_txt", "g7.5"),
}


# ------------------------------------------------------------------ selection rules (report Sec. 3.3)

def _summary(exp, run_name):
    from evaluate import evaluate_run
    return evaluate_run(RESULTS / exp / run_name)


def select_hparams():
    """Dilation of the ground-truth mask: lowest LaMa LPIPS-in among dilations within 0.5 pt of the best text
    residual. Hybrid strength: lowest LPIPS-in among strengths whose text residual and new-text rate are no worse than LaMa's."""
    sel = dict(FALLBACK)
    try:
        c = {d: _summary("sweep_dilation", f"lama_d{d}") for d in (0, 4, 8, 16, 32)}
        best_tr = min(s["text_res"] for s in c.values())
        sel["oracle_dilate"] = min((s["lpips_in"], d) for d, s in c.items() if s["text_res"] <= best_tr + 0.005)[1]
    except Exception as e:
        print("oracle dilation: sweep unavailable, fallback", e, flush=True)
    try:
        lama = _summary("sweep_dilation", "lama_d8")
        cands = []
        for st in (0.3, 0.5, 0.7, 0.9):
            s = _summary("sweep_strength", f"hy{st}")
            if s["text_res"] <= lama["text_res"] + 0.005 and s["new_text_img"] <= lama["new_text_img"] + 1e-9:
                cands.append((s["lpips_in"], st))
        if cands:
            sel["hybrid_strength"] = min(cands)[1]
    except Exception as e:
        print("hybrid strength: sweep unavailable, fallback", e, flush=True)
    HPARAMS.write_text(json.dumps(sel, indent=2))
    print("selected on dev:", sel, flush=True)
    return sel


def hparams():
    return json.loads(HPARAMS.read_text()) if HPARAMS.exists() else select_hparams()


# ------------------------------------------------------------------ test

def test_jobs(h):
    od, st = h["oracle_dilate"], h["hybrid_strength"]
    return [
        R("main", "identity", "identity", "test"),
        R("main", "telea", "telea", "test", dilate=od),
        R("main", "lama", "lama", "test", dilate=od),
        R("main", "sd15", "sd15", "test", dilate=od),
        R("main", "hybrid", "hybrid", "test", dilate=od, strength=st),
        R("main", "ip2p", "ip2p", "test"),
    ]


# SDXL at native 1024 px: ~3.5 min/img on an M2 Pro, ~10 s on an A100 -> full test set on CUDA
SDXL_LIMIT = int(os.environ.get("SDXL_LIMIT", 160))


def sdxl_jobs(h):
    return [R("main", "sdxl", "sdxl", "test", dilate=h["oracle_dilate"], limit=SDXL_LIMIT)]


KONTEXT_DEV = [R("sweep_kontext", f"g{g}", "kontext", guidance=g) for g in (1.5, 2.5, 4.0)]
KONTEXT_TEST = [R("main", "kontext", "kontext", "test")]

# ------------------------------------------------------------------ driver

def execute(jobs):
    for j in jobs:
        key = (j["exp"], j["run_name"])
        if key in ALIASES:
            src = RESULTS / ALIASES[key][0] / ALIASES[key][1]
            dst = RESULTS / j["exp"] / j["run_name"]
            run(**next(x for x in DEV if (x["exp"], x["run_name"]) == ALIASES[key]))  # no-op if complete
            (dst / "images").mkdir(parents=True, exist_ok=True)
            for f in (src / "images").glob("*.png"):
                if not (dst / "images" / f.name).exists():
                    shutil.copy2(f, dst / "images" / f.name)
            shutil.copy2(src / "meta.json", dst / "meta.json")
            continue
        print(f"== {j['exp']}/{j['run_name']}", flush=True)
        run(**j)
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test", "sdxl", "kontext", "all"], default="all")
    a = ap.parse_args()
    if a.phase in ("dev", "all"):
        execute(DEV)
        select_hparams()
    if a.phase in ("test", "all"):
        execute(test_jobs(hparams()))
    if a.phase in ("sdxl", "all"):
        execute(sdxl_jobs(hparams()))
    if a.phase == "kontext":
        execute(KONTEXT_DEV + KONTEXT_TEST)
    print("done", flush=True)


if __name__ == "__main__":
    main()
