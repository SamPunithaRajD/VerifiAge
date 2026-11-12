from typing import List, Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    status: str  # "success" | "no_face" | "multiple_faces" | "low_quality"
    predicted_age: Optional[float] = None
    age_range: Optional[str] = None
    face_status: Optional[str] = None
    quality_status: Optional[str] = None
    quality_message: Optional[str] = None
    model: Optional[str] = None
    calibration_applied: bool = False
    confidence_pct: Optional[float] = None
    message: Optional[str] = None
    gradcam_base64: Optional[str] = None
    face_box: Optional[List[int]] = None  # [x, y, w, h] in the submitted image's pixel coords


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
