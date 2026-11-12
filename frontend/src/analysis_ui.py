"""Shared result-rendering helpers for the Face Analysis and Upload Analysis pages."""

import base64
import io

from PIL import Image

from src.api_client import get_confidence_info
from src.theme import status_pill


def get_confidence_threshold(backend_ok):
    """The model's own overall empirical reliability rate - used as the High/Low split
    point, never an invented number."""
    if not backend_ok:
        return None
    info = get_confidence_info()
    return info["overall_rate_pct"] if info else None


def confidence_label(confidence_pct, threshold):
    if confidence_pct is None or threshold is None:
        return None
    return "High" if confidence_pct >= threshold else "Low"


def render_error_state(st, container, status, message):
    titles = {
        "no_face": ("No reliable human face detected.", None),
        "multiple_faces": ("Multiple faces detected.", "Please keep one person in frame."),
        "low_quality": ("Image quality is insufficient.", "Try better lighting or move closer."),
    }
    title, sub = titles.get(status, ("Prediction unavailable.", message))
    with container:
        st.markdown(f'''
        <div class="card fade-in">
            <div class="section-heading">Analysis</div>
            {status_pill(title, "bad")}
            {f'<div class="result-message" style="margin-top:0.9rem;">{sub}</div>' if sub else ""}
        </div>
        ''', unsafe_allow_html=True)


def render_success(st, container, result, threshold, show_model_row=False):
    quality_kind = "ok" if result.get("quality_status") == "PASS" else "warn"
    quality_label = "Good" if result.get("quality_status") == "PASS" else "Low Quality"
    conf_label = confidence_label(result.get("confidence_pct"), threshold)

    with container:
        rows = f'''
        <div class="card fade-in">
            <div class="section-heading">Analysis</div>
            <div class="metric-row"><span class="metric-label">Face</span>
                {status_pill("Verified", "ok")}</div>
            <div class="metric-row"><span class="metric-label">Quality</span>
                {status_pill(quality_label, quality_kind)}</div>
        '''
        if conf_label:
            rows += f'''<div class="metric-row"><span class="metric-label">Confidence</span>
                {status_pill(conf_label, "ok" if conf_label == "High" else "warn")}</div>'''
        if show_model_row:
            rows += '''<div class="metric-row"><span class="metric-label">Model</span>
                <span class="metric-value">ResNet18</span></div>'''
        rows += '</div>'

        rows += f'''
        <div class="card fade-in">
            <div class="section-heading">Estimated Age</div>
            <div class="metric-value big"><span class="accent-num">{result["predicted_age"]:.0f}</span></div>
            <div class="result-message">years</div>
        </div>
        <div class="card fade-in">
            <div class="metric-row"><span class="metric-label">Age Range</span>
                <span class="metric-value">{result["age_range"]}</span></div>
        </div>
        '''
        st.markdown(rows, unsafe_allow_html=True)

        if quality_kind == "warn" and result.get("quality_message"):
            st.caption(result["quality_message"])


def render_gradcam(st, container, result):
    """Explanation only, shown after the result, collapsed by default. Never part of
    live inference (the backend skips it entirely there via ?live=true)."""
    b64 = result.get("gradcam_base64")
    if not b64:
        return
    with container:
        with st.expander("Why this prediction?"):
            st.image(Image.open(io.BytesIO(base64.b64decode(b64))), use_container_width=True,
                     caption="Regions the model weighted most for this estimate")


def render_waiting(st, container, message="Position your face inside the frame."):
    with container:
        st.markdown(f'''
        <div class="card">
            <div class="section-heading">Analysis</div>
            <div class="result-message">{message}</div>
        </div>
        ''', unsafe_allow_html=True)


def draw_face_box(pil_image, face_box):
    """Draws the backend's actual detected face box on a copy of the image - not a
    re-detection, just a visualization of the same box the model was validated on."""
    from PIL import ImageDraw
    img = pil_image.convert("RGB").copy()
    if not face_box:
        return img
    x, y, w, h = face_box
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + w, y + h], outline=(57, 255, 20), width=max(2, img.width // 200))
    return img
