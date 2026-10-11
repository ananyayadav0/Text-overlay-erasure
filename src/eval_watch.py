"""Evaluate each run as soon as it has finished generating (runs next to run_all.py).

    python src/eval_watch.py            # loop until results/.all_done exists
    python src/eval_watch.py --once
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, load_meta  # noqa: E402
from evaluate import evaluate_run  # noqa: E402


def finished_runs():
    for img_dir in sorted(RESULTS.glob("*/*/images")):
        d = img_dir.parent
        if d.parent.name.startswith("_"):
            continue
        ids = load_meta(d).get("ids", [])
        if ids and len(list(img_dir.glob("*.png"))) >= len(ids) and not (d / "summary.json").exists():
            yield d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        for d in finished_runs():
            s = evaluate_run(d)
            print(f"{d.parent.name}/{d.name}: text_res={s['text_res']:.3f} lpips_in={s['lpips_in']:.4f} "
                  f"lpips_out={s['lpips_out']:.4f} new_text_img={s['new_text_img']:.3f}", flush=True)
        if a.once or (RESULTS / ".all_done").exists():
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
