"""Erasure methods behind one interface:  fn(violated: PIL RGB, mask: PIL L, **params) -> PIL RGB.

mask-based : telea, lama, sd15, sdxl, hybrid (LaMa init + SD1.5 refinement, our proposal)
mask-free  : ip2p (InstructPix2Pix), kontext (FLUX.1 Kontext [dev], 12B rectified-flow transformer; CUDA only)
"""
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

from common import paste_back, work_size

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

SD15 = "stable-diffusion-v1-5/stable-diffusion-inpainting"
SDXL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
IP2P = "timbrooks/instruct-pix2pix"
KONTEXT = "black-forest-labs/FLUX.1-Kontext-dev"   # gated: accept the licence on HF and set HF_TOKEN

PROMPT = "a clean professional product photo, plain background, high quality, sharp"
NEG = "text, letters, words, caption, watermark, logo, signature, qr code, sticker, badge, blurry, low quality"
INSTRUCTION = "remove all the text, watermarks, stickers, badges and QR codes from the image"


def _gen(seed):
    return torch.Generator("cpu").manual_seed(seed)


# ---------------------------------------------------------------- classical / feed-forward

def identity(img, mask, **_):
    return img


def telea(img, mask, radius=5, **_):
    out = cv2.inpaint(np.asarray(img), np.asarray(mask), radius, cv2.INPAINT_TELEA)
    return Image.fromarray(out)


@lru_cache(maxsize=1)
def _lama():
    from simple_lama_inpainting import SimpleLama
    # the TorchScript LaMa graph has ops MPS lacks: CUDA if present, otherwise CPU (fast enough at 512 px)
    return SimpleLama(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def lama(img, mask, **_):
    out = _lama()(img, mask)
    return out.crop((0, 0, *img.size)) if out.size != img.size else out


# ---------------------------------------------------------------- diffusion inpainting

@lru_cache(maxsize=1)
def _sd15():
    from diffusers import StableDiffusionInpaintPipeline
    p = StableDiffusionInpaintPipeline.from_pretrained(SD15, torch_dtype=DTYPE, variant="fp16", safety_checker=None)
    p.set_progress_bar_config(disable=True)
    return p.to(DEVICE)


def sd15(img, mask, steps=30, guidance=7.5, strength=1.0, prompt=PROMPT, negative=NEG, seed=0, **_):
    w, h = img.size
    return _sd15()(prompt=prompt, negative_prompt=negative or None, image=img, mask_image=mask,
                   height=h, width=w, num_inference_steps=steps, guidance_scale=guidance,
                   strength=strength, generator=_gen(seed)).images[0]


@lru_cache(maxsize=1)
def _sdxl():
    from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline
    p = StableDiffusionXLInpaintPipeline.from_pretrained(SDXL, torch_dtype=DTYPE, variant="fp16")
    p.set_progress_bar_config(disable=True)
    p = p.to(DEVICE)
    p.vae.to(torch.float32)          # fp16 SDXL VAE overflows to NaN; decode in fp32
    p.vae.config.force_upcast = True
    return p


def sdxl(img, mask, steps=30, guidance=8.0, strength=0.99, prompt=PROMPT, negative=NEG, seed=0, res=1024, **_):
    """SDXL-inpaint is trained at 1024 px: upsample, inpaint, come back to working size."""
    size = work_size(*img.size, res)
    out = _sdxl()(prompt=prompt, negative_prompt=negative or None, image=img.resize(size, Image.LANCZOS),
                  mask_image=mask.resize(size, Image.NEAREST), height=size[1], width=size[0],
                  num_inference_steps=steps, guidance_scale=guidance, strength=strength,
                  generator=_gen(seed)).images[0]
    return out.resize(img.size, Image.LANCZOS)


def hybrid(img, mask, steps=30, guidance=7.5, strength=0.5, prompt=PROMPT, negative=NEG, seed=0, **_):
    """Proposed: LaMa gives a structure-consistent, text-free fill; SD1.5 re-noises it only
    partially (strength<1) and restores texture, so it cannot invent new text from scratch."""
    base = lama(img, mask)
    w, h = img.size
    return _sd15()(prompt=prompt, negative_prompt=negative or None, image=base, mask_image=mask,
                   height=h, width=w, num_inference_steps=steps, guidance_scale=guidance,
                   strength=strength, generator=_gen(seed)).images[0]


# ---------------------------------------------------------------- mask-free instruction editing

@lru_cache(maxsize=1)
def _ip2p():
    from diffusers import StableDiffusionInstructPix2PixPipeline
    p = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P, torch_dtype=DTYPE, variant="fp16", safety_checker=None)
    p.set_progress_bar_config(disable=True)
    return p.to(DEVICE)


def ip2p(img, mask, steps=30, guidance=7.5, image_guidance=1.5, instruction=INSTRUCTION, seed=0, **_):
    return _ip2p()(prompt=instruction, image=img, num_inference_steps=steps, guidance_scale=guidance,
                   image_guidance_scale=image_guidance, generator=_gen(seed)).images[0].resize(img.size)


@lru_cache(maxsize=1)
def _kontext():
    """bf16 on GPUs with >= 38 GB (A100/H100); otherwise the transformer is loaded in 4-bit NF4 and
    components are offloaded to CPU between uses (L4 24 GB). T4 / Apple MPS are not supported."""
    from diffusers import FluxKontextPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("FLUX.1 Kontext needs a CUDA GPU (run it on Colab, see colab/run_on_colab.ipynb)")
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    if vram < 20:
        raise RuntimeError(f"FLUX.1 Kontext needs >= 20 GB VRAM (L4/A100); this GPU has {vram:.0f} GB")
    if vram >= 38:
        p = FluxKontextPipeline.from_pretrained(KONTEXT, torch_dtype=torch.bfloat16).to("cuda")
    else:
        from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        tr = FluxTransformer2DModel.from_pretrained(KONTEXT, subfolder="transformer", quantization_config=q,
                                                    torch_dtype=torch.bfloat16)
        p = FluxKontextPipeline.from_pretrained(KONTEXT, transformer=tr, torch_dtype=torch.bfloat16)
        p.enable_model_cpu_offload()
    p.set_progress_bar_config(disable=True)
    return p


def kontext(img, mask, steps=28, guidance=2.5, instruction=INSTRUCTION, seed=0, res=1024, **_):
    """Kontext is trained around 1 MP: upsample to 1024 px long side, edit, resize back to working size."""
    size = work_size(*img.size, res, multiple=16)   # FLUX packs 2x2 latent patches: sides % 16 == 0
    out = _kontext()(image=img.resize(size, Image.LANCZOS), prompt=instruction, guidance_scale=guidance,
                     num_inference_steps=steps, height=size[1], width=size[0], max_area=size[0] * size[1],
                     _auto_resize=False, generator=_gen(seed)).images[0]
    return out.resize(img.size, Image.LANCZOS)


METHODS = {"identity": identity, "telea": telea, "lama": lama, "sd15": sd15, "sdxl": sdxl,
           "hybrid": hybrid, "ip2p": ip2p, "kontext": kontext}
MASK_FREE = {"identity", "ip2p", "kontext"}


def apply(method, img, mask, paste=True, feather=0, **params):
    out = METHODS[method](img, mask, **params)
    if out.size != img.size:
        out = out.resize(img.size, Image.LANCZOS)
    if paste and method not in MASK_FREE:
        out = paste_back(img, out, mask, feather=feather)
    return out
