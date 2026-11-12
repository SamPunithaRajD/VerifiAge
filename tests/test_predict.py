import glob
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ml"))

import face_gate
import predict

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_sample_face_paths = glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", "age_prediction_up",
                                             "age_prediction", "test", "030", "*.jpg"))
_child_face_paths = glob.glob(os.path.join(PROJECT_ROOT, "data", "raw", "age_prediction_up",
                                            "age_prediction", "test", "001", "*.jpg"))


def _sample_face_image():
    if not _sample_face_paths:
        pytest.skip("no sample dataset image available")
    return Image.open(_sample_face_paths[0]).convert("RGB")


def test_predict_from_image_valid_face():
    if not os.path.exists(predict.CHECKPOINT_PATH):
        pytest.skip("no trained checkpoint available")

    image = _sample_face_image()
    result = predict.predict_from_image(image)

    assert result["status"] == "success"
    assert result["predicted_age"] >= 0
    assert result["model"] == "resnet18"


def test_predict_from_image_no_face():
    blank = Image.new("RGB", (200, 200), color=(128, 128, 128))
    result = predict.predict_from_image(blank)

    assert result["status"] == "no_face"
    assert result["predicted_age"] is None


def test_face_gate_rejects_non_face_pattern():
    from PIL import ImageDraw
    checker = Image.new("RGB", (200, 200), (255, 255, 255))
    d = ImageDraw.Draw(checker)
    for i in range(0, 200, 25):
        for j in range(0, 200, 25):
            if (i + j) // 25 % 2 == 0:
                d.rectangle([i, j, i + 25, j + 25], fill=(0, 0, 0))

    gate_result = face_gate.run_face_gate(checker)
    assert gate_result["status"] == face_gate.FACE_STATUS_NO_FACE


def test_small_face_in_large_frame_does_not_produce_wild_prediction():
    """Regression test for a real bug: a child's face occupying only a small fraction
    of a much larger frame (a realistic photo composition, unlike this dataset's own
    tightly-framed images) was predicted ~20 years old instead of ~1-2, because
    face_gate wasn't cropping to match the framing the model was trained on. The
    calibrated crop (face_gate.MARGIN_RATIO_W/H) fixes this - this test guards
    against it silently regressing."""
    if not os.path.exists(predict.CHECKPOINT_PATH):
        pytest.skip("no trained checkpoint available")
    if not _child_face_paths:
        pytest.skip("no sample child image available")

    child = Image.open(_child_face_paths[0]).convert("RGB")
    canvas = Image.new("RGB", (600, 700), (190, 185, 180))
    canvas.paste(child.resize((170, 170)), (215, 265))

    result = predict.predict_from_image(canvas)
    if result["status"] != "success":
        pytest.skip(f"gate rejected this synthetic composition (status={result['status']}), "
                     "not the failure mode under test")

    # Threshold set from measured evidence, not an accuracy target: without the
    # calibrated crop this composition produced 19-25 years; with it, 15-18. This
    # catches a regression back toward "no crop"/wrong-margin behavior, but small
    # faces in large frames are a disclosed, only-partially-solved limitation - this
    # is not a claim the fix is perfect.
    assert result["predicted_age"] < 20, (
        f"predicted {result['predicted_age']} for a young child with a small face in a "
        "large frame - framing/crop calibration may have regressed toward the original bug"
    )
