"""OCR + QR helpers used by the evaluation metrics."""
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def reader():
    import easyocr
    import torch
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)


def read_text(img, min_conf=0.1, mag_ratio=2.0):
    """Run EasyOCR on a PIL image. Returns [(polygon Nx2 int, text, conf)]."""
    arr = np.asarray(img.convert("RGB"))
    out = []
    for box, text, conf in reader().readtext(arr, mag_ratio=mag_ratio, text_threshold=0.6, low_text=0.3):
        if conf >= min_conf and text.strip():
            out.append((np.asarray(box, dtype=np.int32), text, float(conf)))
    return out


_qr = cv2.QRCodeDetector()


def find_qr(img, scale=2):
    """Detect QR codes. Returns [(polygon 4x2 int, decoded_text)]; text is '' if located but unreadable.
    Decoding at 2x raises the decode rate of the planted codes on unedited inputs from 45% to 88%."""
    if scale != 1:
        found = find_qr(img.resize((img.width * scale, img.height * scale), Image.LANCZOS), scale=1)
        return [((p / scale).astype(np.int32), t) for p, t in found]
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    found = []
    try:
        ok, texts, pts, _ = _qr.detectAndDecodeMulti(arr)
        if ok and pts is not None:
            found = [(p.astype(np.int32), t) for p, t in zip(pts, texts)]
    except cv2.error:
        pass
    if not found:
        try:
            t, pts, _ = _qr.detectAndDecode(arr)
            if pts is not None:
                found = [(pts.reshape(-1, 2).astype(np.int32), t)]
        except cv2.error:
            pass
    return found


def poly_center(poly):
    return poly.mean(axis=0)


def poly_bbox(poly):
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    return [int(x0), int(y0), int(x1), int(y1)]


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
