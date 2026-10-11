"""Fix the study's 200 images (40 dev + 160 test) and write dataset/splits.json.

The generator renders a pool of candidates (make_violation_dataset.py --n 400); this step draws the
seeded random 200 used in the study, freezes the split, and moves the other candidates to
dataset_unused/ so that dataset/ holds exactly the 200 images of the report.

    python src/finalize_dataset.py
"""
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, N_DEV, N_TEST, SPLIT_SEED  # noqa: E402

UNUSED = DATA.parent / "dataset_unused"


def main():
    if (DATA / "splits.json").exists():
        print("dataset already finalized:", DATA / "splits.json")
        return
    with open(DATA / "annotations.jsonl") as f:
        anns = [json.loads(l) for l in f]
    ids = sorted(a["id"] for a in anns)
    random.Random(SPLIT_SEED).shuffle(ids)
    split = {"dev": ids[:N_DEV], "test": ids[N_DEV:N_DEV + N_TEST]}
    keep = set(split["dev"]) | set(split["test"])
    for sub, ext in (("clean", "jpg"), ("violated", "jpg"), ("mask", "png")):
        (UNUSED / sub).mkdir(parents=True, exist_ok=True)
        for a in anns:
            if a["id"] not in keep:
                p = DATA / sub / f"{a['id']}.{ext}"
                if p.exists():
                    shutil.move(str(p), UNUSED / sub / p.name)
    with open(DATA / "annotations.jsonl", "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in anns if a["id"] in keep)
    with open(UNUSED / "annotations.jsonl", "w") as f:
        f.writelines(json.dumps(a) + "\n" for a in anns if a["id"] not in keep)
    (DATA / "splits.json").write_text(json.dumps(split, indent=1))
    print(f"kept {len(keep)} images (dev {len(split['dev'])}, test {len(split['test'])}); "
          f"moved {len(anns) - len(keep)} to {UNUSED}")


if __name__ == "__main__":
    main()
