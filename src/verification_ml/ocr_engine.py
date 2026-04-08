"""RapidOCR (ONNX) — CPU-friendly; optional preprocessing applied upstream."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from numpy.typing import NDArray

_lock = threading.Lock()
_engine: Any = None


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
    return _engine


def run_ocr_on_bgr(bgr: NDArray[np.uint8]) -> tuple[str, float | None]:
    """
    Returns (full_text, mean_confidence or None).
    RapidOCR may return OCRResult (.txts, .scores) or a list of [box, text, score].
    """
    engine = _get_engine()
    result, _ = engine(bgr)
    if result is None:
        return "", None

    if hasattr(result, "txts"):
        txts = getattr(result, "txts", None) or []
        scores = getattr(result, "scores", None) or []
        text = "\n".join(str(t) for t in txts).strip()
        if not scores:
            return text, None
        try:
            conf = float(sum(float(s) for s in scores) / len(scores))
        except (TypeError, ValueError):
            conf = None
        return text, conf

    if not isinstance(result, (list, tuple)):
        return "", None

    lines: list[str] = []
    scores: list[float] = []
    for item in result:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            _, text, score = item[0], item[1], item[2]
            lines.append(str(text))
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            lines.append(str(item[1]))
    text = "\n".join(lines).strip()
    conf = float(sum(scores) / len(scores)) if scores else None
    return text, conf
