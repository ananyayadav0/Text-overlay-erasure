#!/usr/bin/env python3
"""
Build a synthetic "text overlay violation" dataset from fashion product images.

For each clean product image the script writes:
    clean/<id>.jpg        original image (resized so the long side <= --size)
    violated/<id>.jpg     same image with 1-2 violations drawn on top, re-saved as JPEG
    mask/<id>.png         binary mask, 255 where the overlay touched pixels
    annotations.jsonl     one line per image: violation type, text, bbox, params
    preview.jpg           contact sheet of the first 20 pairs, for a quick eyeball

Image sources (pick with --source):
    kaggle   Kaggle "Fashion Product Images Dataset" (paramaggarwal), full-res,
             downloaded one file at a time through the kaggle CLI.
             Needs ~/.kaggle/kaggle.json set up by you beforehand.
    abo      Amazon Berkeley Objects main (hero) images from the public S3 bucket,
             full resolution, no credentials. Default.
    hf       Hugging Face mirror ceyda/fashion-products-small (max side 512),
             no credentials needed. Lower resolution but zero setup.

Usage:
    python make_violation_dataset.py --out ./dataset --n 300 --source kaggle
    python make_violation_dataset.py --out ./dataset --n 300 --source hf
"""

import argparse
import glob
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

# --------------------------------------------------------------------------
# Violation catalogue. Five types, all text-based, all common in apparel listings.
# --------------------------------------------------------------------------

VIOLATIONS = {
    "promo_claim": {
        "desc": "Promotional text over the product",
        "texts": [
            "LIMITED TIME 50% OFF", "BUY 1 GET 1 FREE", "TODAY ONLY $9.99",
            "FLASH SALE 70% OFF", "FREE SHIPPING NOW", "LAST 3 LEFT!",
            "HOT DEAL - SAVE $40", "CLEARANCE 80% OFF", "BEST SELLER 2026",
            "NEW ARRIVAL - 30% OFF", "SALE ENDS TONIGHT", "LOWEST PRICE EVER",
        ],
        "texts_zh": [
            "限时5折", "买一送一", "全网最低价", "今日特价9.9元", "清仓甩卖",
            "仅剩3件", "包邮到家", "爆款直降200", "秒杀价", "年度销量冠军",
        ],
    },
    "rank_badge": {
        "desc": "Ranking or guarantee badge with short text",
        "texts": [
            "No.1", "#1 CHOICE", "TOP RATED", "OFFICIAL", "100% ORIGINAL",
            "BEST SELLER", "5 STAR", "VERIFIED", "TOP 10", "MUST HAVE",
        ],
        "texts_zh": ["销量第一", "正品保证", "官方认证", "热卖", "第一名", "国家级", "顶级"],
    },
    "store_watermark": {
        "desc": "Semi-transparent tiled store name",
        "texts": [
            "@FashionHub_SG", "StyleNest Store", "www.cheapwear.shop",
            "SHOP ID 88213", "LuxeCloset Official", "TrendyThreads",
            "@dailydrip_apparel", "MegaMall Fashion",
        ],
        "texts_zh": ["优品家居旗舰店", "好物小铺", "潮流百货专营店", "店铺ID 88213"],
    },
    "contact_info": {
        "desc": "Off-platform contact details",
        "texts": [
            "WhatsApp +65 9123 4567", "Add WeChat: fashion_88", "Telegram @stylezone",
            "Call 9876 5432 for bulk", "DM Instagram @nestwear", "Line ID: cloth_deal",
            "WhatsApp for lower price", "WeChat: ss_outlet2026",
        ],
        "texts_zh": ["加微信 fashion_88 更便宜", "私聊QQ 38291764", "电话 138 0013 8000", "加V领优惠券"],
    },
    "exaggerated_effect": {
        "desc": "Unverifiable body or effect claim",
        "texts": [
            "SLIMMING 10 LBS INSTANTLY", "LOOKS 5 YEARS YOUNGER", "INSTANT WAIST SHAPER",
            "HIDES BELLY 100%", "LEGS LOOK 3 INCHES LONGER", "ANTI-SWEAT FOREVER",
            "NEVER FADES GUARANTEED", "MIRACLE FIT - ANY BODY", "BURNS FAT WHILE WORN",
        ],
        "texts_zh": ["7天见效", "永不褪色", "用了年轻10岁", "100%除菌", "三天瘦10斤", "终身不坏"],
    },
    "qr_code": {
        "desc": "QR code leading off-platform, sometimes with a caption",
        "texts": ["https://wa.me/6591234567", "https://t.me/stylezone", "weixin://dl/chat?fashion_88",
                  "https://cheapwear.shop/p/88213", "https://line.me/ti/p/cloth_deal"],
        "captions": ["SCAN FOR 20% OFF", "SCAN TO CHAT", "扫码加微信", "扫码领券", "扫码下单更优惠", ""],
    },
}

ZH_RATIO = 0.0   # share of text violations drawn in Chinese; overridden by --zh-ratio (English only by default)


def pick_text(vtype):
    v = VIOLATIONS[vtype]
    if v.get("texts_zh") and random.random() < ZH_RATIO:
        return random.choice(v["texts_zh"])
    return random.choice(v["texts"])


def is_cjk(text):
    return any(ord(ch) > 0x2E80 for ch in text)

PALETTE = [
    (230, 40, 40), (255, 210, 0), (20, 20, 20), (255, 255, 255),
    (0, 120, 220), (255, 100, 0), (170, 0, 130), (0, 150, 90),
]

# --------------------------------------------------------------------------
# Fonts: pick up whatever the machine has. macOS paths first, then Linux.
# --------------------------------------------------------------------------

FONT_DIRS = [
    "/System/Library/Fonts", "/System/Library/Fonts/Supplemental", "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
    "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
]
PREFERRED = ["Impact", "Arial Black", "Helvetica", "Arial", "Verdana", "Georgia",
             "Futura", "Avenir", "Trebuchet", "Chalkboard", "Marker", "DejaVuSans",
             "Liberation"]


def collect_fonts():
    found = []
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            found.extend(glob.glob(os.path.join(d, ext)))
    # prefer the bold, display-ish ones sellers actually use
    ranked = [f for f in found if any(p.lower() in os.path.basename(f).lower() for p in PREFERRED)]
    fonts = ranked or found
    usable = []
    for f in fonts:
        try:
            if covers(ImageFont.truetype(f, 40), "Sale 50% OFF @WhatsApp +65"):
                usable.append(f)
        except Exception:
            pass
    return usable[:20]


def covers(font, text):
    """True if every glyph in text renders differently from the font's missing-glyph box."""
    def glyph(ch):
        im = Image.new("L", (64, 64))
        ImageDraw.Draw(im).text((8, 4), ch, font=font, fill=255)
        return im.tobytes()
    missing = glyph("\U0010FFFF")
    return all(glyph(ch) != missing for ch in set(text) if not ch.isspace())


FONTS = collect_fonts()

CJK_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
CJK_FONTS = [f for f in CJK_CANDIDATES
             if os.path.exists(f) and covers(ImageFont.truetype(f, 40), "销量第一见效扫码加微信旗舰店")]


def load_font(size, text=""):
    pool = CJK_FONTS if is_cjk(text) and CJK_FONTS else FONTS
    if pool:
        try:
            return ImageFont.truetype(random.choice(pool), size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


# --------------------------------------------------------------------------
# Image sources
# --------------------------------------------------------------------------

def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr.strip()}")
    return r.stdout


def kaggle_fetch_file(dataset, remote, dest_dir):
    """Download one file from a Kaggle dataset. Kaggle sometimes wraps it in a zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "datasets", "download", "-d", dataset, "-f", remote, "-p", str(dest_dir), "--force"])
    name = os.path.basename(remote)
    out = dest_dir / name
    z = dest_dir / (name + ".zip")
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        z.unlink()
    if not out.exists():
        raise FileNotFoundError(f"Expected {out} after download")
    return out


def source_kaggle(n, work, seed):
    import pandas as pd
    ds = "paramaggarwal/fashion-product-images-dataset"
    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found. pip install kaggle, then put your token at ~/.kaggle/kaggle.json")
    meta_dir = work / "kaggle_meta"
    styles = kaggle_fetch_file(ds, "fashion-dataset/styles.csv", meta_dir)
    df = pd.read_csv(styles, on_bad_lines="skip")
    df = df[df["masterCategory"].isin(["Apparel", "Footwear", "Accessories"])]
    # spread across sub-categories so no single type dominates
    per = max(1, math.ceil(n / df["subCategory"].nunique()))
    picked = df.sample(frac=1, random_state=seed).groupby("subCategory").head(per)
    picked = picked.sample(min(n, len(picked)), random_state=seed)
    img_dir = work / "kaggle_images"
    paths = []
    for i, row in enumerate(picked.itertuples(), 1):
        remote = f"fashion-dataset/images/{row.id}.jpg"
        try:
            p = kaggle_fetch_file(ds, remote, img_dir)
            paths.append((str(row.id), p))
        except Exception as e:
            print(f"  skip {row.id}: {e.__class__.__name__}")
        if i % 25 == 0:
            print(f"  downloaded {i}/{len(picked)}")
    return paths


def source_hf(n, work, seed):
    from datasets import load_dataset
    print("Loading ceyda/fashion-products-small from Hugging Face (max side 512) ...")
    ds = load_dataset("ceyda/fashion-products-small", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    img_dir = work / "hf_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in ds:
        pid = str(row.get("id") or row.get("filename", "").split(".")[0])
        p = img_dir / f"{pid}.jpg"
        row["image"].convert("RGB").save(p, quality=95)
        paths.append((pid, p))
    return paths


ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
# Product types whose main image is an object on a plain background and rarely carries
# printed text of its own (phone cases, groceries, cosmetics etc. are left out).
ABO_TYPES = {
    "SHOES", "BOOT", "SANDAL", "CHAIR", "SOFA", "TABLE", "OTTOMAN", "STOOL_SEATING",
    "LAMP", "LIGHT_FIXTURE", "RUG", "HANDBAG", "HAT", "SUITCASE", "HEADBOARD",
    "FINERING", "RING", "FINEEARRING", "EARRING", "FINENECKLACEBRACELETANKLET", "NECKLACE",
    "HOME_BED_AND_BATH", "HOME_FURNITURE_AND_DECOR", "HARDWARE_HANDLE", "DRINKING_CUP",
    "KITCHEN", "OUTDOOR_LIVING", "ACCESSORY", "SPORTING_GOODS", "FURNITURE_COVER",
}


def _http_get(url, dest):
    import urllib.request
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def source_abo(n, work, seed, shards=2):
    """Main (hero) images from Amazon Berkeley Objects, public S3, no credentials."""
    import gzip
    from concurrent.futures import ThreadPoolExecutor
    meta = work / "abo_meta"
    meta.mkdir(parents=True, exist_ok=True)
    listings = []
    for s in "0123456789abcdef"[:shards]:
        p = meta / f"listings_{s}.json.gz"
        if not p.exists():
            _http_get(f"{ABO}/listings/metadata/listings_{s}.json.gz", p)
        with gzip.open(p, "rt") as f:
            for line in f:
                d = json.loads(line)
                pt = (d.get("product_type") or [{}])[0].get("value")
                if d.get("main_image_id") and pt in ABO_TYPES:
                    listings.append((d["main_image_id"], pt))
    img_csv = meta / "images.csv.gz"
    if not img_csv.exists():
        _http_get(f"{ABO}/images/metadata/images.csv.gz", img_csv)
    import pandas as pd
    im = pd.read_csv(img_csv).set_index("image_id")
    df = pd.DataFrame(listings, columns=["image_id", "ptype"]).drop_duplicates("image_id")
    df = df[df["image_id"].isin(im.index)]
    df = df.join(im, on="image_id")
    df = df[(df[["height", "width"]].min(axis=1) >= 512)]
    # balance across product types
    per = max(1, math.ceil(n / df["ptype"].nunique()))
    picked = df.sample(frac=1, random_state=seed).groupby("ptype").head(per)
    if len(picked) < n:
        rest = df[~df["image_id"].isin(picked["image_id"])]
        picked = pd.concat([picked, rest.sample(min(len(rest), n - len(picked)), random_state=seed)])
    picked = picked.sample(min(n, len(picked)), random_state=seed)
    print(f"  {len(picked)} ABO main images across {picked['ptype'].nunique()} product types")
    (work / "abo_types.json").write_text(json.dumps(dict(zip(picked["image_id"], picked["ptype"]))))
    img_dir = work / "abo_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        dest = img_dir / f"{row.image_id}.jpg"
        if not dest.exists():
            try:
                _http_get(f"{ABO}/images/original/{row.path}", dest)
            except Exception as e:
                print(f"  skip {row.image_id}: {e.__class__.__name__}")
                return None
        return (row.image_id, dest)

    with ThreadPoolExecutor(8) as ex:
        paths = [p for p in ex.map(fetch, picked.itertuples()) if p]
    return paths


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def product_bbox(img, thresh=235):
    """Rough box around the product: everything that is not near-white."""
    g = img.convert("L")
    bw = g.point(lambda v: 0 if v > thresh else 255)
    box = bw.getbbox()
    if not box:
        return (0, 0, img.width, img.height)
    return box


def text_size(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def place(img_w, img_h, w, h, pbox, on_product_prob=0.65):
    """Choose a top-left corner. Most of the time land on the product itself."""
    if random.random() < on_product_prob:
        x0, y0, x1, y1 = pbox
        cx = random.randint(x0, max(x0, x1 - 1))
        cy = random.randint(y0, max(y0, y1 - 1))
        x, y = cx - w // 2, cy - h // 2
    else:
        x = random.randint(0, max(0, img_w - w))
        y = random.randint(0, max(0, img_h - h))
    x = max(4, min(x, img_w - w - 4))
    y = max(4, min(y, img_h - h - 4))
    return x, y


# --------------------------------------------------------------------------
# Overlay drawers. Each returns (RGBA layer, annotation dict).
# --------------------------------------------------------------------------

def draw_text_line(img, text, vtype, pbox, size_range, stroke=True, shadow=True, rotate=True):
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    font = load_font(random.randint(*size_range), text)
    tmp = ImageDraw.Draw(layer)
    tw, th = text_size(tmp, text, font)
    pad = 12
    box = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(box)
    fill = random.choice(PALETTE)
    sw = random.choice([0, 2, 3, 4]) if stroke else 0
    scol = (0, 0, 0, 255) if sum(fill) > 380 else (255, 255, 255, 255)
    if shadow and random.random() < 0.5:
        d.text((pad + 3, pad + 3), text, font=font, fill=(0, 0, 0, 140))
    d.text((pad, pad), text, font=font, fill=fill + (255,), stroke_width=sw, stroke_fill=scol)
    angle = random.uniform(-12, 12) if rotate and random.random() < 0.5 else 0
    if angle:
        box = box.rotate(angle, expand=True, resample=Image.BICUBIC)
    x, y = place(W, H, box.width, box.height, pbox)
    layer.alpha_composite(box, (x, y))
    return layer, {
        "type": vtype, "text": text, "bbox": [x, y, x + box.width, y + box.height],
        "font_px": font.size if hasattr(font, "size") else None,
        "rotation_deg": round(angle, 1), "stroke_px": sw,
    }


def draw_badge(img, text, pbox):
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shape = random.choice(["circle", "ribbon", "pill"])
    font = load_font(random.randint(int(W * 0.035), int(W * 0.06)), text)
    tmp = ImageDraw.Draw(layer)
    tw, th = text_size(tmp, text, font)
    fill = random.choice(PALETTE[:3] + [PALETTE[4], PALETTE[5]])
    tcol = (255, 255, 255, 255) if sum(fill) < 400 else (20, 20, 20, 255)
    if shape == "circle":
        r = int(max(tw, th) * 0.75) + 10
        box = Image.new("RGBA", (r * 2, r * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(box)
        d.ellipse((0, 0, r * 2 - 1, r * 2 - 1), fill=fill + (255,), outline=(255, 255, 255, 255), width=3)
        d.text((r - tw / 2, r - th / 2 - 2), text, font=font, fill=tcol)
    elif shape == "pill":
        pw, ph = tw + 40, th + 24
        box = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(box)
        d.rounded_rectangle((0, 0, pw - 1, ph - 1), radius=ph // 2, fill=fill + (255,))
        d.text((20, 10), text, font=font, fill=tcol)
    else:  # ribbon: rectangle with a notch on the right
        pw, ph = tw + 50, th + 26
        box = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        d = ImageDraw.Draw(box)
        d.polygon([(0, 0), (pw - 1, 0), (pw - 18, ph // 2), (pw - 1, ph - 1), (0, ph - 1)], fill=fill + (255,))
        d.text((16, 11), text, font=font, fill=tcol)
    if random.random() < 0.4:
        box = box.rotate(random.uniform(-15, 15), expand=True, resample=Image.BICUBIC)
    # badges go to a corner most of the time, like real ones
    if random.random() < 0.7:
        corner = random.choice([(8, 8), (W - box.width - 8, 8), (8, H - box.height - 8), (W - box.width - 8, H - box.height - 8)])
        x, y = corner
    else:
        x, y = place(W, H, box.width, box.height, pbox)
    layer.alpha_composite(box, (x, y))
    return layer, {"type": "rank_badge", "text": text, "shape": shape, "bbox": [x, y, x + box.width, y + box.height]}


def draw_watermark(img, text, pbox):
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    font = load_font(random.randint(int(W * 0.04), int(W * 0.07)), text)
    alpha = random.randint(45, 120)
    col = random.choice([(255, 255, 255), (0, 0, 0), (200, 200, 200), (120, 120, 120)])
    angle = random.choice([-30, -20, 0, 20, 30])
    tmp = ImageDraw.Draw(layer)
    tw, th = text_size(tmp, text, font)
    tile = Image.new("RGBA", (tw + 60, th + 60), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((30, 30), text, font=font, fill=col + (alpha,))
    tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
    step_x, step_y = tile.width + random.randint(20, 80), tile.height + random.randint(20, 80)
    mode = random.choice(["tiled", "single"])
    if mode == "tiled":
        for y in range(-tile.height, H, step_y):
            for x in range(-tile.width, W, step_x):
                layer.alpha_composite(tile, (x, y))
        bbox = [0, 0, W, H]
    else:
        x, y = place(W, H, tile.width, tile.height, pbox, on_product_prob=0.5)
        layer.alpha_composite(tile, (x, y))
        bbox = [x, y, x + tile.width, y + tile.height]
    return layer, {"type": "store_watermark", "text": text, "mode": mode, "alpha": alpha,
                   "rotation_deg": angle, "bbox": bbox}


def draw_qr(img, text, pbox):
    import qrcode
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    side = random.randint(int(min(W, H) * 0.14), int(min(W, H) * 0.24))
    q = qrcode.QRCode(border=2, box_size=10)
    q.add_data(text)
    q.make(fit=True)
    code = q.make_image(fill_color="black", back_color="white").convert("RGBA").resize((side, side), Image.NEAREST)
    caption = random.choice([c for c in VIOLATIONS["qr_code"]["captions"] if ZH_RATIO > 0 or not is_cjk(c)])
    if caption:
        font = load_font(max(12, side // 8), caption)
        tw, th = text_size(ImageDraw.Draw(layer), caption, font)
        bw = max(side, tw + 16)
        box = Image.new("RGBA", (bw, side + th + 16), (255, 255, 255, 255))
        box.alpha_composite(code, ((bw - side) // 2, 0))
        ImageDraw.Draw(box).text(((bw - tw) // 2, side + 4), caption, font=font, fill=(230, 40, 40, 255))
    else:
        box = code
    if random.random() < 0.6:
        x, y = random.choice([(8, 8), (W - box.width - 8, 8), (8, H - box.height - 8), (W - box.width - 8, H - box.height - 8)])
    else:
        x, y = place(W, H, box.width, box.height, pbox, on_product_prob=0.5)
    layer.alpha_composite(box, (x, y))
    return layer, {"type": "qr_code", "text": text, "caption": caption, "bbox": [x, y, x + box.width, y + box.height]}


DRAWERS = {
    "promo_claim":        lambda img, t, pb: draw_text_line(img, t, "promo_claim", pb, (int(img.width * 0.045), int(img.width * 0.09))),
    "rank_badge":         lambda img, t, pb: draw_badge(img, t, pb),
    "store_watermark":    lambda img, t, pb: draw_watermark(img, t, pb),
    "contact_info":       lambda img, t, pb: draw_text_line(img, t, "contact_info", pb, (int(img.width * 0.03), int(img.width * 0.05)), rotate=False),
    "exaggerated_effect": lambda img, t, pb: draw_text_line(img, t, "exaggerated_effect", pb, (int(img.width * 0.04), int(img.width * 0.075))),
    "qr_code":            lambda img, t, pb: draw_qr(img, t, pb),
}


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def resize_long_side(img, size):
    w, h = img.size
    s = size / max(w, h)
    if s >= 1:
        return img
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def build(paths, out, size, seed):
    random.seed(seed)
    for sub in ("clean", "violated", "mask"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    ann_path = out / "annotations.jsonl"
    types = list(VIOLATIONS)
    counts = {t: 0 for t in types}
    tfile = out / "_work" / "abo_types.json"
    ptypes = json.loads(tfile.read_text()) if tfile.exists() else {}
    preview = []
    with open(ann_path, "w") as ann:
        for i, (pid, p) in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            img = resize_long_side(img, size)
            pbox = product_bbox(img)
            base = img.convert("RGBA")
            k = 1 if random.random() < 0.65 else 2
            chosen = random.sample(types, k)
            merged = Image.new("RGBA", img.size, (0, 0, 0, 0))
            records = []
            for vt in chosen:
                text = pick_text(vt) if vt != "qr_code" else random.choice(VIOLATIONS[vt]["texts"])
                layer, rec = DRAWERS[vt](img, text, pbox)
                x0, y0, x1, y1 = rec["bbox"]
                rec["bbox"] = [max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)]
                rec["lang"] = "zh" if is_cjk(rec["text"] + rec.get("caption", "")) else "en"
                merged.alpha_composite(layer)
                records.append(rec)
                counts[vt] += 1
            out_img = Image.alpha_composite(base, merged).convert("RGB")
            mask = merged.split()[3].point(lambda a: 255 if a > 8 else 0).convert("L")
            q = random.randint(72, 95)
            img.save(out / "clean" / f"{pid}.jpg", quality=95)
            out_img.save(out / "violated" / f"{pid}.jpg", quality=q)
            mask.save(out / "mask" / f"{pid}.png")
            ann.write(json.dumps({
                "id": pid, "width": img.width, "height": img.height,
                "product_type": ptypes.get(pid), "product_bbox": list(pbox), "jpeg_quality": q,
                "violations": records,
            }) + "\n")
            if len(preview) < 20:
                preview.append((img, out_img, mask))
            if (i + 1) % 50 == 0:
                print(f"  built {i + 1}/{len(paths)}")
    make_preview(preview, out / "preview.jpg")
    print("\nDone.")
    print(f"  images: {len(paths)}")
    for t, c in counts.items():
        print(f"  {t:20s} {c}")
    print(f"  output: {out}")


def make_preview(pairs, path, thumb=256):
    if not pairs:
        return
    cols = 4
    rows = math.ceil(len(pairs) / cols)
    # each cell: clean | violated | mask outline drawn in red over the violated image
    sheet = Image.new("RGB", (cols * thumb * 3 + (cols + 1) * 8, rows * thumb + (rows + 1) * 8), (245, 245, 245))
    for idx, (a, b, m) in enumerate(pairs):
        r, c = divmod(idx, cols)
        x = 8 + c * (thumb * 3 + 8)
        y = 8 + r * (thumb + 8)
        edge = m.filter(ImageFilter.MaxFilter(5)).point(lambda v: 255 if v else 0)
        overlay = Image.composite(Image.new("RGB", b.size, (255, 0, 0)), b.point(lambda v: v // 3), edge)
        overlay = Image.composite(b, overlay, m)
        for j, im in enumerate((a, b, overlay)):
            t = im.copy()
            t.thumbnail((thumb, thumb))
            sheet.paste(t, (x + j * thumb, y))
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder for the dataset")
    ap.add_argument("--n", type=int, default=300, help="number of product images")
    ap.add_argument("--source", choices=["abo", "kaggle", "hf"], default="abo")
    ap.add_argument("--zh-ratio", type=float, default=0.0, help="share of text violations in Chinese")
    ap.add_argument("--size", type=int, default=1024, help="long side of saved images")
    ap.add_argument("--seed", type=int, default=6127)
    args = ap.parse_args()
    global ZH_RATIO
    ZH_RATIO = args.zh_ratio

    out = Path(args.out).expanduser().resolve()
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)

    print(f"Fonts available: {len(FONTS)}" + ("" if FONTS else "  (none found, falling back to PIL default)"))
    print(f"Fetching {args.n} clean images from {args.source} ...")
    fetch = {"abo": source_abo, "kaggle": source_kaggle, "hf": source_hf}[args.source]
    paths = fetch(args.n, work, args.seed)
    if not paths:
        sys.exit("No images downloaded.")
    print(f"Got {len(paths)} images. Drawing violations ...")
    build(paths, out, args.size, args.seed)


if __name__ == "__main__":
    main()
