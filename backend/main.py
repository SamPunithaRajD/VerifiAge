import json
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import inference
from schemas import HealthResponse, PredictResponse

app = FastAPI(title="VerifiAge API")

# Needed for the live-camera component: it's a self-contained HTML/JS block embedded via
# Streamlit's components API, which renders in its own origin and calls this API directly
# via fetch(). No cookies/credentials are used, so an open CORS policy is fine here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
CONFIDENCE_PATH = os.path.join(RESULTS_DIR, "confidence_by_band.json")
CALIBRATION_PATH = os.path.join(RESULTS_DIR, "calibration.json")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": inference.model_available()}


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="No metrics available yet. Run evaluation first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/confidence")
def confidence_info():
    if not os.path.exists(CONFIDENCE_PATH):
        raise HTTPException(status_code=404, detail="Confidence table not available yet.")
    with open(CONFIDENCE_PATH) as f:
        return json.load(f)


@app.get("/calibration")
def calibration_info():
    if not os.path.exists(CALIBRATION_PATH):
        raise HTTPException(status_code=404, detail="Calibration not available yet.")
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), mock: bool = False, live: bool = False):
    if mock:
        return inference.mock_predict()

    if not inference.model_available():
        raise HTTPException(status_code=503, detail="Model checkpoint not found. Train the model first.")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        # live=true (periodic calls from the live-camera loop): skip Grad-CAM, which is
        # explanation-only and never part of the live inference path (also keeps each
        # periodic call fast - it would otherwise cost an extra forward+backward pass).
        result = inference.predict_from_bytes(file_bytes, include_gradcam=not live)
    except inference.InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error")

    return result


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
