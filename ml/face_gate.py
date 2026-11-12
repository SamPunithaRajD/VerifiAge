"""
Shared face detection + validation gate. Used by the backend (/predict) and the
frontend (upload + live camera) so the same rules decide what reaches the model
everywhere: IMAGE -> DETECT -> VALIDATE -> QUALITY GATE -> SINGLE FACE -> CROP.

The ResNet18 model must never see an image that failed this gate.
"""

import os

import cv2
import numpy as np
from PIL import Image

YUNET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_detection_yunet.onnx")

# ---- configurable thresholds (tested against the 41-image sample in test_detector_sample.py) ----
SCORE_THRESHOLD = 0.75          # YuNet detection confidence
MIN_FACE_SIZE_PX = 40           # reject tiny/far-away faces
MIN_ASPECT_RATIO = 0.6          # face box width/height - real faces are roughly square-ish
MAX_ASPECT_RATIO = 1.7
BLUR_THRESHOLD = 30.0           # Laplacian variance below this = too blurry
MIN_BRIGHTNESS = 25             # mean pixel brightness (0-255) below this = too dark
MAX_BRIGHTNESS = 235            # above this = blown out / overexposed

_detector = None


def _get_detector(w, h):
    global _detector
    if _detector is None:
        _detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (w, h), score_threshold=SCORE_THRESHOLD)
    _detector.setInputSize((w, h))
    return _detector


def detect_faces(pil_image):
    """Returns a list of dicts: {box: (x,y,w,h), score, landmarks}. Raw detections,
    not yet validated."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    detector = _get_detector(w, h)
    _, faces = detector.detect(bgr)
    if faces is None:
        return []

    results = []
    for f in faces:
        x, y, fw, fh = f[0:4].astype(int)
        landmarks = f[4:14].reshape(5, 2).astype(int).tolist()
        score = float(f[14])
        results.append({"box": (int(x), int(y), int(fw), int(fh)), "score": score, "landmarks": landmarks})
    return results


def _blur_score(pil_crop):
    gray = cv2.cvtColor(np.array(pil_crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _clamp_box(box, image_size):
    """YuNet boxes routinely overshoot the frame by a few px for faces near the edge
    (normal detector jitter, not a bad detection) - clamp instead of rejecting the
    whole face outright."""
    x, y, w, h = box
    img_w, img_h = image_size
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return (x, y, max(1, x2 - x), max(1, y2 - y))


def validate_face(face, image_size):
    """Multi-signal validation of a single raw detection. Returns (is_valid, reason)."""
    x, y, w, h = _clamp_box(face["box"], image_size)

    if face["score"] < SCORE_THRESHOLD:
        return False, "detector confidence too low"

    if w < MIN_FACE_SIZE_PX or h < MIN_FACE_SIZE_PX:
        return False, "face too small"

    aspect = w / h if h > 0 else 0
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False, "implausible face proportions"

    return True, None


# Calibrated from the ACTUAL training data (measured across 148 training images with
# this same YuNet detector, see diagnostic in the child-misprediction investigation):
# detected face boxes cover width=46.2% / height=61.4% of the frame on average. These
# margins reproduce that same framing around a detected face in ANY input photo, so a
# real-world upload (where the face might be tiny in a big photo, or fill the whole
# frame) gets resized to look like what the model actually saw in training. Without
# this, faces occupying a very different fraction of the frame than training data
# caused wildly wrong predictions (verified: a child's face small in a large frame
# was predicted ~20 years old instead of ~1-2).
MARGIN_RATIO_W = 0.582
MARGIN_RATIO_H = 0.314


def crop_face(pil_image, box, margin_ratio_w=MARGIN_RATIO_W, margin_ratio_h=MARGIN_RATIO_H):
    width, height = pil_image.size
    x, y, w, h = _clamp_box(box, pil_image.size)
    mx, my = int(w * margin_ratio_w), int(h * margin_ratio_h)
    left, top = max(0, x - mx), max(0, y - my)
    right, bottom = min(width, x + w + mx), min(height, y + h + my)
    return pil_image.crop((left, top, right, bottom))


def quality_gate(face_crop):
    """PASS / WARNING / REJECT based on blur and brightness of the cropped face."""
    gray = np.array(face_crop.convert("L"))
    brightness = gray.mean()
    blur = _blur_score(face_crop)

    # brightness checked first: darkening an image also lowers its Laplacian-variance blur
    # score, so checking blur first would misdiagnose a dark-but-sharp face as "blurry"
    if brightness < MIN_BRIGHTNESS:
        return "REJECT", "Image is too dark. Try better lighting or move closer."
    if blur < BLUR_THRESHOLD:
        return "REJECT", "Image quality is too low (blurry). Try better lighting or hold still."
    if brightness > MAX_BRIGHTNESS:
        return "REJECT", "Image is overexposed. Reduce lighting/glare."
    if blur < BLUR_THRESHOLD * 1.8:
        return "WARNING", "Image is a bit blurry, prediction may be less reliable."
    return "PASS", None


FACE_STATUS_NO_FACE = "no_face"
FACE_STATUS_MULTIPLE = "multiple_faces"
FACE_STATUS_LOW_QUALITY = "low_quality"
FACE_STATUS_VALID = "valid"


def run_face_gate(pil_image):
    """Full pipeline: detect -> validate -> single-face policy -> quality gate.

    Returns a dict: {status, message, face (dict or None), crop (PIL.Image or None)}
    status is one of: no_face, multiple_faces, low_quality, valid.
    """
    raw_faces = detect_faces(pil_image)
    valid_faces = []
    for f in raw_faces:
        ok, _ = validate_face(f, pil_image.size)
        if ok:
            valid_faces.append(f)

    if len(valid_faces) == 0:
        return {"status": FACE_STATUS_NO_FACE, "message": "No reliable human face detected.",
                "face": None, "crop": None}

    if len(valid_faces) > 1:
        return {"status": FACE_STATUS_MULTIPLE,
                "message": "Multiple faces detected. Please provide an image containing one person.",
                "face": None, "crop": None}

    face = dict(valid_faces[0])
    face["box"] = _clamp_box(face["box"], pil_image.size)  # for accurate box display/overlay use
    crop = crop_face(pil_image, face["box"])
    quality, quality_message = quality_gate(crop)

    if quality == "REJECT":
        return {"status": FACE_STATUS_LOW_QUALITY, "message": quality_message,
                "face": face, "crop": None}

    return {"status": FACE_STATUS_VALID, "message": quality_message, "face": face,
            "crop": crop, "quality": quality}
