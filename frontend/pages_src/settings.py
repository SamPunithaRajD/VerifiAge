import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.theme import inject

inject(st)

st.markdown('<div class="section-eyebrow">Preferences</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Settings</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">This build is configured for a fixed demo setup — no editable '
            'options are needed here yet.</div>', unsafe_allow_html=True)

st.markdown('''
<div class="card">
    <div class="section-heading">About</div>
    <div class="metric-row"><span class="metric-label">Product</span><span class="metric-value">VerifiAge</span></div>
    <div class="metric-row"><span class="metric-label">Model</span><span class="metric-value">ResNet18</span></div>
    <div class="metric-row"><span class="metric-label">Gender detection</span><span class="metric-value">Not supported (out of scope)</span></div>
    <div class="metric-row"><span class="metric-label">Identity recognition</span><span class="metric-value">Not implemented</span></div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="result-message">Technical thresholds, raw backend details and diagnostics '
            'live under <b>Developer Mode</b> in the sidebar.</div>', unsafe_allow_html=True)
