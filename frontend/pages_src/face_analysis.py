import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api_client import BackendError, DEFAULT_BASE_URL, check_health
from src.live_camera import render as render_live_camera
from src.theme import inject

inject(st)

try:
    health = check_health()
    backend_ok = health.get("model_loaded", False)
except BackendError:
    backend_ok = False

st.markdown('<div class="section-eyebrow">Live</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Face Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Continuous camera tracking with periodic ResNet18 inference. '
            'No photos are captured or stored.</div>', unsafe_allow_html=True)

if not backend_ok:
    st.markdown('<div class="state-banner">The analysis service is currently unavailable. '
                'Please try again shortly.</div>', unsafe_allow_html=True)
    st.stop()

render_live_camera(st, backend_url=DEFAULT_BASE_URL)
