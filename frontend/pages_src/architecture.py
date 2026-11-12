import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.theme import inject

inject(st)

st.markdown('<div class="section-eyebrow">Technical</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Architecture</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">How an image becomes a validated age estimate.</div>', unsafe_allow_html=True)

STEPS = [
    ("User", None),
    ("Upload / Live Camera", "Streamlit"),
    ("Input Validation", None),
    ("Face Detection", "OpenCV YuNet"),
    ("Face Validation", None),
    ("Quality Gate", None),
    ("Preprocessing", "PyTorch"),
    ("Age Regression", "ResNet18"),
    ("Prediction", "Grad-CAM (explanation only)"),
    ("API Response", "FastAPI"),
    ("VerifiAge UI", "Streamlit"),
]

html = '<div class="card" style="padding: 2.4rem 2rem;"><div style="display:flex; flex-direction:column; align-items:center;">'
for i, (title, label) in enumerate(STEPS):
    label_html = f'<div style="font-size:0.72rem; color:#86868B; letter-spacing:0.08em; text-transform:uppercase; margin-top:0.3rem;">{label}</div>' if label else ''
    html += (
        '<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1); '
        'border-radius:12px; padding:0.9rem 1.6rem; text-align:center; min-width:260px;">'
        f'<div style="font-size:0.95rem; font-weight:600; color:#F5F5F7; letter-spacing:0.02em;">{title}</div>'
        f'{label_html}'
        '</div>'
    )
    if i < len(STEPS) - 1:
        html += '<div style="width:1.5px; height:28px; background:linear-gradient(180deg, #39FF14, rgba(57,255,20,0.15)); margin: 2px 0;"></div>'
html += '</div></div>'

st.markdown(html, unsafe_allow_html=True)

st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

st.markdown('<div class="section-heading">Stack</div>', unsafe_allow_html=True)
stack_cols = st.columns(6)
stack_items = ["OpenCV YuNet", "PyTorch", "ResNet18", "FastAPI", "Streamlit", "Grad-CAM"]
for col, item in zip(stack_cols, stack_items):
    with col:
        st.markdown(f'<div class="card" style="text-align:center; padding:1rem 0.5rem;">'
                    f'<div class="metric-value" style="font-size:0.9rem;">{item}</div></div>',
                    unsafe_allow_html=True)
