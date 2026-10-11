# Erasing Text-Overlay Violations from E-commerce Product Images

A technical review of **inpainting** and **instruction-based editing** models for automatically removing
policy-violating overlays — promo slogans, ranking badges, store watermarks, contact details and QR codes —
from product "hero" images, while leaving the product itself untouched.

Group project for **CA6127 Transformer and Generative Adversarial Networks** (NTU) by Chen Yu Heng and
Ananya Yadav.

## The question

Marketplaces normally reject a hero image that carries disallowed text and ask the seller for a new one.
Can the offending elements instead be erased automatically? We compare six public models from four families,
plus a training-free pipeline of our own:

| Method | Family | Needs a mask? |
|---|---|---|
| Telea | Classical inpainting (no learning) | yes |
| LaMa | GAN inpainting (Fourier convolutions) | yes |
| Stable Diffusion 1.5 Inpainting | Latent diffusion (U-Net) | yes |
| SDXL Inpainting | Latent diffusion (larger U-Net) | yes |
| InstructPix2Pix | Instruction-based editor (U-Net) | no |
| FLUX.1 Kontext [dev] | Instruction-based editor (12B diffusion transformer) | no |
| **P1: LaMa → SD1.5 (ours)** | LaMa fill, then SDEdit-style diffusion refinement inside the mask | yes |

## Dataset

200 hero images from [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
across 30 product types, each with one or two **synthetic violations** of six kinds (promo claim, rank badge,
store watermark, contact info, effect claim, QR code). Fonts, colours, rotation, opacity and JPEG
re-compression all vary. Because the violations are generated, every sample comes with the clean original and
a pixel-exact mask — so erasure quality can be measured exactly.

Split once (seed 6127) into **40 dev** images for sweeps and ablations and **160 test** images for the final
comparison. No hyper-parameter was chosen by looking at test results.

## Evaluation

Erasure, fill quality and background damage are scored separately rather than merged into one number:

- **Text residual** — share of planted characters EasyOCR can still read
- **QR residual** — share of planted QR codes that still decode
- **New text** — share of images where the model invented lettering
- **PSNR / LPIPS inside the mask** — is the erased region what was underneath?
- **PSNR / LPIPS outside the mask** — was the rest of the image left alone?
- **SSIM** (whole image) and **seconds per image**

For the mask-free editors we additionally measure *where* they change the image relative to the ground-truth
mask, separating "finding" the violation from "filling" it.

## Results (test set, n = 160, NVIDIA A100)

| Method | Text res. % ↓ | New text % ↓ | PSNR-in ↑ | LPIPS-in ↓ | PSNR-out ↑ | LPIPS-out ↓ | SSIM ↑ |
|---|---|---|---|---|---|---|---|
| Input (no edit) | 79.6 | 0.6 | 8.32 | 0.1028 | 48.18 | 0.0064 | 0.955 |
| Telea | 0.3 | 0.0 | 25.26 | 0.0214 | 40.22 | 0.0142 | 0.978 |
| **LaMa** | **0.1** | **0.0** | 27.72 | 0.0076 | **42.98** | **0.0055** | **0.984** |
| SD1.5-Inpaint | 0.2 | 0.6 | 22.87 | 0.0231 | 39.83 | 0.0251 | 0.978 |
| SDXL-Inpaint | 2.3 | 4.4 | 20.74 | 0.0399 | 38.75 | 0.0384 | 0.968 |
| P1: LaMa → SD1.5 (ours) | 0.2 | 0.0 | 26.85 | 0.0079 | 42.46 | 0.0083 | 0.983 |
| InstructPix2Pix | 30.6 | 8.1 | 7.44 | 0.1064 | 16.09 | 0.3624 | 0.728 |
| FLUX.1 Kontext | **0.1** | 0.6 | **32.54** | **0.0052** | 35.22 | 0.0289 | 0.964 |

Inpainters receive the ground-truth mask (dilated by 4 px); the editors receive no mask. No method left a
decodable QR code.

**Key findings**

1. **Given the mask, erasure is essentially solved** — and the lightweight GAN **LaMa** beats both diffusion
   inpainters on quality while running about 30× faster. SDXL is the weakest learned inpainter because it
   tends to repaint a new badge where the old one was.
2. **Mask dilation (4 px) matters more than any other hyper-parameter.** For diffusion models the negative
   prompt is the most important component, and pasting the original pixels back outside the mask is essential.
3. **Our P1 pipeline** removes SD1.5's invented text and colour shift by starting from LaMa's fill, but does not
   beat LaMa itself — diffusion refinement adds little on this task.
4. **FLUX.1 Kontext** finds 99.5% of violation pixels on its own and produces the most accurate fills of any
   method, but almost a third of the pixels it changes lie outside the violations — and it is ~500× slower
   than LaMa.
5. LaMa and Kontext are the two ends of a **fidelity–throughput trade-off**: LaMa for high-volume moderation
   once violations are located, Kontext when fill fidelity matters more than latency.

**Limitations:** detection is out of scope (inpainters are given the exact mask); violations are synthetic,
English-only and use a limited set of fonts.

## Reproducing

Development sweeps were run on an Apple M2 Pro (PyTorch MPS); final numbers on one Colab A100. Every step is
resumable — finished images are skipped.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_models.py [--kontext]                 # ~12 GB; --kontext adds ~34 GB, needs HF_TOKEN + accepted licence
.venv/bin/python make_violation_dataset.py --out ./dataset --n 400  # render candidate images with synthetic violations
.venv/bin/python src/finalize_dataset.py                            # keep 200 images (40 dev + 160 test), freeze the split
.venv/bin/python src/run_all.py --phase dev                         # sweeps + ablations; selects hyper-parameters
.venv/bin/python src/run_all.py --phase test                        # main comparison on 160 test images
.venv/bin/python src/run_all.py --phase sdxl                        # SDXL-Inpaint on test
.venv/bin/python src/run_all.py --phase kontext                     # FLUX.1 Kontext sweep + test (CUDA, >= 20 GB VRAM)
.venv/bin/python src/benchmark_speed.py                             # seconds/image for every method
.venv/bin/python src/evaluate.py --force                            # score every run on one machine
.venv/bin/python src/analysis.py                                    # figures and tables -> report/figures, report/tables
```

`./resume.sh` launches the remaining dev/test/SDXL phases plus evaluation in the background on a Mac
(`./resume.sh status` shows progress). To run on Colab, build `colab/ca6127_colab.zip` with
`colab/pack_for_colab.sh`, upload it to Google Drive and follow `colab/run_on_colab.ipynb`.

## Repository layout

| Path | Purpose |
|---|---|
| `make_violation_dataset.py` | Downloads ABO hero images, draws the six violation types, writes clean / violated / mask / annotations |
| `src/common.py` | Frozen split, 512 px working resolution, mask dilation, paste-back |
| `src/finalize_dataset.py` | Reduces rendered candidates to the 200 images used and writes `dataset/splits.json` |
| `src/methods.py` | Telea, LaMa, SD1.5, SDXL, InstructPix2Pix, FLUX.1 Kontext and the LaMa→SD1.5 pipeline behind one interface |
| `src/run_method.py` | One configuration over one split |
| `src/run_all.py` | Every experiment, plus dev-based hyper-parameter selection |
| `src/ocr.py` | EasyOCR and OpenCV QR helpers |
| `src/evaluate.py`, `src/eval_watch.py` | Metrics (text/QR residual, new text, PSNR/LPIPS in/out of mask, SSIM); background scorer |
| `src/benchmark_speed.py` | Single-GPU speed table |
| `src/analysis.py` | Figures and tables, including where the editors change the image |
| `colab/` | Colab notebook and upload packer |
| `report/` | Report build tooling (Markdown → styled .docx / PDF) |

## Tech stack

PyTorch · Hugging Face Diffusers & Transformers · LaMa · EasyOCR · OpenCV · LPIPS · scikit-image · pandas · Matplotlib
