"""Thin wrapper around the VerifiAge FastAPI backend. The UI never loads the model itself,
and never sees raw thresholds/URLs - those stay internal."""

import io

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8010"
TIMEOUT = 30


class BackendError(Exception):
    pass


def check_health(base_url=DEFAULT_BASE_URL):
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise BackendError(str(e))


def _get_optional(path, base_url):
    try:
        r = requests.get(f"{base_url}{path}", timeout=5)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def get_metrics(base_url=DEFAULT_BASE_URL):
    return _get_optional("/metrics", base_url)


def get_confidence_info(base_url=DEFAULT_BASE_URL):
    return _get_optional("/confidence", base_url)


def get_calibration_info(base_url=DEFAULT_BASE_URL):
    return _get_optional("/calibration", base_url)


def predict_image(pil_image, base_url=DEFAULT_BASE_URL, mock=False):
    """Returns the backend's response dict as-is: status is one of
    success / no_face / multiple_faces / low_quality."""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("face.jpg", buf, "image/jpeg")}
    params = {"mock": "true"} if mock else {}

    try:
        r = requests.post(f"{base_url}/predict", files=files, params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise BackendError(str(e))

    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise BackendError(detail)

    return r.json()
