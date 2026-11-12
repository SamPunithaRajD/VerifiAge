import io
import os
import sys

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import inference  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)


def _sample_jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(120, 90, 60)).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_mock_mode():
    files = {"file": ("test.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/predict?mock=true", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert "predicted_age" in body


def test_predict_no_face_returns_typed_status(monkeypatch):
    monkeypatch.setattr(inference, "model_available", lambda: True)
    files = {"file": ("blank.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_face"
    assert body["predicted_age"] is None


def test_predict_invalid_file():
    files = {"file": ("not_an_image.txt", b"this is not an image", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_empty_upload():
    files = {"file": ("empty.jpg", b"", "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_missing_model_checkpoint(monkeypatch):
    monkeypatch.setattr(inference, "model_available", lambda: False)
    files = {"file": ("test.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 503
