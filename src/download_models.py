"""Pre-download every model used in the experiments (run once per machine).

    python src/download_models.py             # SD1.5-inpaint, SDXL-inpaint, InstructPix2Pix, LaMa, EasyOCR (~12 GB)
    python src/download_models.py --kontext   # + FLUX.1 Kontext [dev] (~34 GB, gated: needs HF_TOKEN + accepted licence)
"""
import argparse

from huggingface_hub import snapshot_download

FP16 = ["*.json", "*.txt", "*fp16.safetensors", "tokenizer*/*", "*.model"]
MODELS = {
    "stable-diffusion-v1-5/stable-diffusion-inpainting": FP16,
    "timbrooks/instruct-pix2pix": FP16,
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": FP16,
}
KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"
# diffusers folders only; skips the 24 GB single-file checkpoint at the repo root
KONTEXT_FILES = ["model_index.json", "*/*.json", "*/*.safetensors", "*/*.txt", "*/*.model", "tokenizer*/*"]


def check_kontext_access():
    from huggingface_hub import hf_hub_download, whoami
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        print("HF user:", whoami()["name"])
        hf_hub_download(KONTEXT, "model_index.json")
    except GatedRepoError:
        raise SystemExit("403: accept the FLUX.1 Kontext [dev] licence on its Hugging Face page first")
    except HfHubHTTPError as e:
        raise SystemExit(f"401/HTTP error: HF_TOKEN missing or invalid ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kontext", action="store_true")
    a = ap.parse_args()
    for repo, pats in MODELS.items():
        print("downloading", repo, flush=True)
        print("  ->", snapshot_download(repo, allow_patterns=pats), flush=True)
    if a.kontext:
        check_kontext_access()
        print("downloading", KONTEXT, flush=True)
        print("  ->", snapshot_download(KONTEXT, allow_patterns=KONTEXT_FILES), flush=True)
    import easyocr
    easyocr.Reader(["en"], gpu=False)
    from simple_lama_inpainting import SimpleLama
    SimpleLama()
    print("all models ready", flush=True)
