"""
InsightFace (buffalo_l) embeddings + cosine similarity in [0, 1].

Uses first detected face per image; if none, similarity 0.0.
"""

from __future__ import annotations

import base64
import threading
from io import BytesIO

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from verification_ml.config import settings

_lock = threading.Lock()
_face_app = None


def _decode_bgr(data_b64: str) -> NDArray[np.uint8]:
    raw = base64.b64decode(data_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        pil = Image.open(BytesIO(raw)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def _get_app():
    global _face_app
    with _lock:
        if _face_app is None:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name=settings.insightface_model, root=settings.insightface_root)
            app.prepare(ctx_id=-1, det_size=(settings.face_det_size, settings.face_det_size))
            _face_app = app
    return _face_app


def _largest_embedding(bgr: NDArray[np.uint8]) -> np.ndarray | None:
    app = _get_app()
    faces = app.get(bgr)
    if not faces:
        return None
    # largest face by bbox area
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if hasattr(best, "normed_embedding") and best.normed_embedding is not None:
        emb = np.asarray(best.normed_embedding, dtype=np.float32)
        return emb
    raw = np.asarray(best.embedding, dtype=np.float32)
    n = float(np.linalg.norm(raw) + 1e-8)
    return raw / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def compare_document_selfie(document_b64: str, selfie_b64: str) -> float:
    doc = _decode_bgr(document_b64)
    selfie = _decode_bgr(selfie_b64)
    e1 = _largest_embedding(doc)
    e2 = _largest_embedding(selfie)
    if e1 is None or e2 is None:
        return 0.0
    sim = cosine_similarity(e1, e2)
    # InsightFace cosine for same person often > 0.3; clamp to [0,1]
    return max(0.0, min(1.0, sim))


def analyze_liveness_frame(action: str, image_b64: str) -> dict:
    """
    Realtime single-frame liveness hint.
    - TURN_LEFT / TURN_RIGHT / HOLD_HIGH use InsightFace pose (yaw/pitch/roll).
    - BLINK uses OpenCV eye cascade count as a pragmatic signal (0 eyes likely blink/closed).
    """
    bgr = _decode_bgr(image_b64)
    app = _get_app()
    faces = app.get(bgr)
    if not faces:
        return {
            "matched": False,
            "faceDetected": False,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "eyesDetected": 0,
        }

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    pose = getattr(face, "pose", None)
    pitch = float(pose[0]) if pose is not None and len(pose) > 0 else 0.0
    yaw = float(pose[1]) if pose is not None and len(pose) > 1 else 0.0
    roll = float(pose[2]) if pose is not None and len(pose) > 2 else 0.0

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    roi = bgr[y1:y2, x1:x2]
    eyes_detected = 0
    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(18, 18))
        eyes_detected = int(len(eyes))

    matched = False
    if action == "TURN_RIGHT":
        matched = yaw <= -9.0
    elif action == "TURN_LEFT":
        matched = yaw >= 9.0
    elif action == "HOLD_HIGH":
        # Frontal / "look at camera": neutral yaw and pitch (aligned with on-device ML Kit checks).
        matched = abs(yaw) <= 14.0 and abs(pitch) <= 10.0 and abs(roll) <= 20.0
    elif action == "BLINK":
        matched = eyes_detected == 0

    return {
        "matched": bool(matched),
        "faceDetected": True,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "eyesDetected": eyes_detected,
    }
