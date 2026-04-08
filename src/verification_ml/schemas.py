from pydantic import BaseModel, Field


class OcrRequest(BaseModel):
    """Matches `delivery-backend` MlVerificationService.extractOcr POST body."""

    imageBase64: str = Field(..., description="Raw document image, base64 (no data: URL prefix required)")


class OcrResponse(BaseModel):
    text: str = ""
    confidence: float | None = None


class FaceRequest(BaseModel):
    """Matches Nest `compareFaces` POST body."""

    documentImageBase64: str
    selfieImageBase64: str


class FaceResponse(BaseModel):
    similarity: float = Field(..., ge=0.0, le=1.0)


class LivenessFrameRequest(BaseModel):
    action: str
    imageBase64: str


class LivenessFrameResponse(BaseModel):
    matched: bool
    faceDetected: bool
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    eyesDetected: int | None = None


class ForgeryRequest(BaseModel):
    imageBase64: str


class ForgeryResponse(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    note: str
    limitations: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
