"""Wraps ml/predict.py so the backend and the CLI share the exact same model + preprocessing code."""

import io
import os
import sys

from PIL import Image, UnidentifiedImageError

ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

import predict as ml_predict  # noqa: E402

MOCK_RESPONSE = {
    "status": "success",
    "predicted_age": 28.4,
    "age_range": "25-31",
    "face_status": "valid",
    "quality_status": "PASS",
    "model": "resnet18",
    "calibration_applied": False,
}


class InvalidImageError(Exception):
    pass


def model_available():
    return os.path.exists(ml_predict.CHECKPOINT_PATH)


def predict_from_bytes(file_bytes, include_gradcam=True):
    if not file_bytes:
        raise InvalidImageError("Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise InvalidImageError("File is not a valid image")

    return ml_predict.predict_from_image(image, include_gradcam=include_gradcam)


def mock_predict():
    return dict(MOCK_RESPONSE)
