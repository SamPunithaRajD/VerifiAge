import os
import sys

import streamlit as st

FRONTEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FRONTEND_ROOT)
from src.api_client import DEFAULT_BASE_URL, check_health
from src.theme import inject

inject(st)

ML_DIR = os.path.join(os.path.dirname(FRONTEND_ROOT), "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

st.markdown('<div class="section-eyebrow">Diagnostics</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Developer Mode</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Technical details, hidden from the main product experience.</div>',
            unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Backend</div>', unsafe_allow_html=True)
st.code(f"Backend URL: {DEFAULT_BASE_URL}", language=None)
try:
    health = check_health()
    st.json(health)
except Exception as e:
    st.error(f"Health check failed: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Face Gate Thresholds</div>', unsafe_allow_html=True)
try:
    import face_gate
    st.json({
        "SCORE_THRESHOLD": face_gate.SCORE_THRESHOLD,
        "MIN_FACE_SIZE_PX": face_gate.MIN_FACE_SIZE_PX,
        "MIN_ASPECT_RATIO": face_gate.MIN_ASPECT_RATIO,
        "MAX_ASPECT_RATIO": face_gate.MAX_ASPECT_RATIO,
        "BLUR_THRESHOLD": face_gate.BLUR_THRESHOLD,
        "MIN_BRIGHTNESS": face_gate.MIN_BRIGHTNESS,
        "MAX_BRIGHTNESS": face_gate.MAX_BRIGHTNESS,
        "MARGIN_RATIO_W": face_gate.MARGIN_RATIO_W,
        "MARGIN_RATIO_H": face_gate.MARGIN_RATIO_H,
    })
except Exception as e:
    st.warning(f"Could not load face_gate thresholds: {e}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Live Camera Config</div>', unsafe_allow_html=True)
try:
    from src.live_camera import INFER_INTERVAL_MS, SMOOTHING_WINDOW, STABILITY_SPREAD_YEARS
    st.json({
        "infer_interval_ms": INFER_INTERVAL_MS,
        "smoothing_window": SMOOTHING_WINDOW,
        "stability_spread_years": STABILITY_SPREAD_YEARS,
    })
except Exception as e:
    st.warning(f"Could not load live camera config: {e}")
st.markdown('</div>', unsafe_allow_html=True)
