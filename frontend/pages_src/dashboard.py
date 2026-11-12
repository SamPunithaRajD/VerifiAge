import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api_client import BackendError, check_health, get_metrics
from src.theme import inject

inject(st)

try:
    health = check_health()
    backend_ok = True
    model_loaded = health.get("model_loaded", False)
except BackendError:
    backend_ok = False
    model_loaded = False

metrics = get_metrics() if backend_ok else None

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-wrap" style="padding-top: 1.5rem; padding-bottom: 1.5rem;">
    <div class="hero-eyebrow"><span class="dot"></span>DASHBOARD</div>
    <div class="hero-title" style="font-size: 3.2rem;">VERIFIAGE</div>
    <div class="hero-sub" style="font-size: 1.1rem; max-width: 640px;">
        AI age estimation that knows when not to guess.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
test_mae = metrics["overall"]["mae"] if metrics else None
r2 = metrics["overall"]["r2"] if metrics else None
cs5 = metrics["overall"]["cs5"] if metrics else None

kpis = [
    ("Model", "ResNet18"),
    ("Test MAE", f"{test_mae:.2f} years" if test_mae is not None else "Not evaluated"),
    ("R²", f"{r2:.2f}" if r2 is not None else "Not evaluated"),
    ("CS(5)", f"{cs5:.0f}%" if cs5 is not None else "Not evaluated"),
    ("Dataset", "~190K clean images"),
]
cols = st.columns(len(kpis))
for col, (label, value) in zip(cols, kpis):
    with col:
        st.markdown(f'''
        <div class="card" style="text-align:center; padding: 1.3rem 0.8rem;">
            <div class="metric-label" style="display:block; margin-bottom:0.5rem;">{label}</div>
            <div class="metric-value" style="font-size:1.3rem;">{value}</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div style="height: 1.2rem;"></div>', unsafe_allow_html=True)

status_col, flow_col = st.columns([1, 1.4], gap="large")

with status_col:
    st.markdown('<div class="section-heading">Recent System Status</div>', unsafe_allow_html=True)

    def status_row(label, ok):
        cls = "ok" if ok else "bad"
        text = label
        st.markdown(f'<div class="metric-row"><span class="status-pill {cls}">'
                    f'<span class="status-dot"></span>{text}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    status_row("Backend Online" if backend_ok else "Backend Offline", backend_ok)
    status_row("Model Loaded" if model_loaded else "Model Not Loaded", model_loaded)
    status_row("Face Detector Ready", backend_ok)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="margin-bottom:0.4rem;">Live Camera</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-message" style="margin-bottom:1rem;">Continuous face tracking with periodic '
                'ResNet18 inference and smoothed age readings.</div>', unsafe_allow_html=True)
    if st.button("Open Live Analysis", type="primary", use_container_width=True):
        st.switch_page("pages_src/face_analysis.py")
    st.markdown('</div>', unsafe_allow_html=True)

with flow_col:
    st.markdown('<div class="section-heading">How It Works</div>', unsafe_allow_html=True)
    steps = ["Input", "Face Validation", "Quality Check", "ResNet18", "Age Prediction"]
    flow_html = '<div class="card" style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.4rem;">'
    for i, step in enumerate(steps):
        flow_html += f'<div style="text-align:center; flex:1; min-width:90px;">' \
                      f'<div style="font-size:0.8rem; font-weight:600; color:#F5F5F7;">{step}</div></div>'
        if i < len(steps) - 1:
            flow_html += '<div style="color:#39FF14; font-size:1.1rem;">&rarr;</div>'
    flow_html += '</div>'
    st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown('<div style="height: 0.8rem;"></div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Try Live Camera", use_container_width=True):
            st.switch_page("pages_src/face_analysis.py")
    with b2:
        if st.button("Upload a Photo", use_container_width=True):
            st.switch_page("pages_src/upload_analysis.py")
