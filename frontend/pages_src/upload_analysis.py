import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis_ui import (draw_face_box, get_confidence_threshold, render_error_state,
                              render_gradcam, render_success, render_waiting)
from src.api_client import BackendError, check_health, predict_image
from src.theme import inject
from src.utils import validate_uploaded_file

inject(st)

try:
    health = check_health()
    backend_ok = health.get("model_loaded", False)
except BackendError:
    backend_ok = False

st.markdown('<div class="section-eyebrow">Upload</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Upload Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload a photo. A prediction is only produced when a single, '
            'valid face passes every check.</div>', unsafe_allow_html=True)

if not backend_ok:
    st.markdown('<div class="state-banner">The analysis service is currently unavailable. '
                'Please try again shortly.</div>', unsafe_allow_html=True)
    st.stop()

threshold = get_confidence_threshold(backend_ok)

if "upload_result" not in st.session_state:
    st.session_state.upload_result = None
if "upload_image" not in st.session_state:
    st.session_state.upload_image = None

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-heading">Image Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image, error = validate_uploaded_file(uploaded_file)
    if error:
        with left:
            st.error(error)
    else:
        with st.spinner("Analyzing..."):
            try:
                result = predict_image(image)
            except BackendError as e:
                result = {"status": "error", "message": str(e)}

        with left:
            if result.get("status") == "success" and result.get("face_box"):
                st.image(draw_face_box(image, result["face_box"]), use_container_width=True)
            else:
                st.image(image, use_container_width=True)

        if result.get("status") == "success":
            render_success(st, right, result, threshold, show_model_row=True)
            render_gradcam(st, right, result)
        elif result.get("status") == "error":
            with right:
                st.error(f"Analysis failed: {result['message']}")
        else:
            render_error_state(st, right, result["status"], result.get("message"))

        with right:
            if st.button("Analyze Again"):
                st.rerun()
else:
    render_waiting(st, right, "Upload a photo to begin.")
