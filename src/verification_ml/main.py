"""
FastAPI entry: OCR + face endpoints compatible with `delivery-backend` MlVerificationService.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from verification_ml import __version__
from verification_ml.auth import verify_bearer
from verification_ml.config import settings
from verification_ml.face_engine import analyze_liveness_frame, compare_document_selfie
from verification_ml.ocr_engine import run_ocr_on_bgr
from verification_ml.preprocess import preprocess_for_ocr
from verification_ml.schemas import (
    FaceRequest,
    FaceResponse,
    ForgeryRequest,
    ForgeryResponse,
    HealthResponse,
    OcrRequest,
    OcrResponse,
    LivenessFrameRequest,
    LivenessFrameResponse,
)


FORGERY_NOTE = (
    "Heuristic resaving / recompression signals only. "
    "Does not detect sophisticated forgeries or screen replays."
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm models in background on startup (optional — first request may be slow otherwise)
    yield


app = FastAPI(
    title="Delivery Verification ML",
    description="OCR (RapidONNX + OpenCV preprocess) + face similarity (InsightFace).",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version=__version__)


@app.post("/ocr", response_model=OcrResponse, dependencies=[Depends(verify_bearer)])
async def ocr_endpoint(body: OcrRequest):
    bgr = preprocess_for_ocr(body.imageBase64)
    text, conf = run_ocr_on_bgr(bgr)
    return OcrResponse(text=text, confidence=conf)


@app.post("/face", response_model=FaceResponse, dependencies=[Depends(verify_bearer)])
async def face_endpoint(body: FaceRequest):
    sim = compare_document_selfie(body.documentImageBase64, body.selfieImageBase64)
    return FaceResponse(similarity=sim)


@app.post("/liveness/frame", response_model=LivenessFrameResponse, dependencies=[Depends(verify_bearer)])
async def liveness_frame_endpoint(body: LivenessFrameRequest):
    payload = analyze_liveness_frame(body.action, body.imageBase64)
    return LivenessFrameResponse(**payload)


@app.post("/v1/forgery", response_model=ForgeryResponse, dependencies=[Depends(verify_bearer)])
async def forgery_stub(_body: ForgeryRequest):
    """Placeholder for future JPEG resaving / noise heuristics."""
    return ForgeryResponse(
        score=0.0,
        note="Forgery module not yet implemented; placeholder response.",
        limitations=FORGERY_NOTE,
    )


@app.get("/")
async def root():
    return {
        "service": "delivery-verification-ml",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/ocr", "/face", "/liveness/frame", "/v1/forgery"],
        "auth": "optional Bearer token if API_KEY is set",
    }
