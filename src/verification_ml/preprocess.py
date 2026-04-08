"""
Document image preprocessing before OCR: grayscale, CLAHE contrast, optional deskew.

Deskew uses min-area rectangle on Canny edges (works for many ID scans; not perfect on all docs).
"""

from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


def _decode_base64_to_bgr(data: str) -> NDArray[np.uint8]:
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(BytesIO(raw)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def deskew_and_enhance(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Deskew (estimate angle) + CLAHE on luminance."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _clahe_color(bgr)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < gray.size * 0.01:
        return _clahe_color(bgr)

    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Ignore tiny rotations (noise)
    if abs(angle) < 0.5 or abs(angle) > 45:
        return _clahe_color(bgr)

    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return _clahe_color(rotated)


def _clahe_color(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_for_ocr(image_base64: str) -> NDArray[np.uint8]:
    """Decode image, deskew + enhance, return BGR uint8."""
    bgr = _decode_base64_to_bgr(image_base64)
    return deskew_and_enhance(bgr)


def bgr_to_rgb_uint8(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
