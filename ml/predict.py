"""
Single-image inference. The model NEVER sees an image that fails the face gate
(no_face / multiple_faces / low_quality all short-circuit before ResNet18 runs).

Usage: python ml/predict.py <image_path>

No age-confidence is invented: "confidence_pct" (when available) is a genuine
empirical reliability rate measured on the held-out TEST set for the predicted
age's band (see ml/calibrate.py) - not a per-image calibrated probability. If
calibration/confidence artifacts haven't been generated yet, those fields are
simply omitted rather than faked.
"""

import base64
import io
import json
import os
import sys

import torch
from PIL import Image

import face_gate
from metrics import age_range_label
from model import create_model
from preprocessing import eval_transform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_resnet18.pth")
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "results", "calibration.json")
CONFIDENCE_TABLE_PATH = os.path.join(PROJECT_ROOT, "results", "confidence_by_band.json")

_model = None
_device = None
_calibration = "unloaded"
_confidence_table = "unloaded"


def _load_json_once(path, cache_attr_name):
    value = globals()[cache_attr_name]
    if value != "unloaded":
        return value
    if os.path.exists(path):
        with open(path) as f:
            value = json.load(f)
    else:
        value = None
    globals()[cache_attr_name] = value
    return value


def get_calibration():
    return _load_json_once(CALIBRATION_PATH, "_calibration")


def get_confidence_table():
    return _load_json_once(CONFIDENCE_TABLE_PATH, "_confidence_table")


def lookup_confidence(predicted_age, table):
    band_width = 10
    lo = int(predicted_age // band_width) * band_width
    key = f"{lo}-{lo + band_width}"
    band = table["bands"].get(key)
    if band is None:
        return table["overall_rate_pct"]
    return band["confidence_pct"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    _device = get_device()
    model = create_model(pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()

    _model = model
    return _model, _device


def predict_from_image(image, include_gradcam=True):
    """image: a PIL.Image already loaded/converted to RGB. Used by both CLI and API.
    include_gradcam=False skips the explanation pass entirely (used by the live-camera
    loop: Grad-CAM is explanation-only, never part of the live inference path, and
    skipping its extra forward+backward pass also keeps periodic live calls fast).

    Runs the full gate first: IMAGE -> FACE DETECTION -> FACE VALIDATION -> QUALITY
    GATE -> SINGLE FACE -> CROP. The model only runs if status ends up "valid".

    IMPORTANT: the crop is calibrated, not a tight box-only crop and not the whole
    unprocessed image. Measured directly from 148 training images: the detected face
    covers ~46% of frame width / ~61% of frame height on average. face_gate.crop_face
    reproduces that same framing around the detected face for ANY input photo. Two
    failure modes were found and fixed here:
      1. An earlier version cropped tightly to just the face box (no margin
         calibration) - zoomed in far more than training data, which measurably hurt
         accuracy.
      2. A naive "just feed the whole image" fix (which worked on this dataset's own
         already-tight test images) catastrophically failed on realistic photos where
         the face is a small fraction of a larger frame - verified reproducible: a
         1-year-old's face, small in a larger frame, was predicted ~20 years old
         instead of ~1-2. The model was never trained on that framing and effectively
         guesses close to the dataset's overall mean age when given it.
    The calibrated crop fixes both: it matches training's framing regardless of the
    input photo's own composition.
    """
    gate_result = face_gate.run_face_gate(image)

    if gate_result["status"] != face_gate.FACE_STATUS_VALID:
        return {
            "status": gate_result["status"],
            "predicted_age": None,
            "message": gate_result["message"],
        }

    model, device = load_model()
    crop = gate_result["crop"]

    tensor = eval_transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor).squeeze().item()

    calibration_applied = False
    calibration = get_calibration()
    if calibration and calibration.get("apply_calibration"):
        prediction = calibration["a"] * prediction + calibration["b"]
        calibration_applied = True

    prediction = max(0.0, prediction)

    result = {
        "status": "success",
        "predicted_age": round(prediction, 1),
        "age_range": age_range_label(prediction),
        "face_status": "valid",
        "quality_status": gate_result["quality"],
        "quality_message": gate_result["message"],
        "model": "resnet18",
        "calibration_applied": calibration_applied,
        "face_box": list(gate_result["face"]["box"]),  # [x, y, w, h] in the submitted image's own pixel coords
    }

    confidence_table = get_confidence_table()
    if confidence_table:
        result["confidence_pct"] = lookup_confidence(prediction, confidence_table)

    if include_gradcam:
        try:
            import gradcam
            overlay = gradcam.generate_gradcam(model, device, crop)
            if overlay is not None:
                buf = io.BytesIO()
                overlay.save(buf, format="PNG")
                result["gradcam_base64"] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass  # explanation feature only - never block the core prediction

    return result


def predict_age(image_path):
    image = Image.open(image_path).convert("RGB")
    return predict_from_image(image)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ml/predict.py <image_path>")
        sys.exit(1)

    result = predict_age(sys.argv[1])
    print(result)
