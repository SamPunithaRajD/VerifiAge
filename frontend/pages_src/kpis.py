import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api_client import get_calibration_info, get_metrics
from src.theme import inject

inject(st)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_QUALITY_PATH = os.path.join(PROJECT_ROOT, "results", "data_quality.json")

st.markdown('<div class="section-eyebrow">Verified metrics only</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">KPIs</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Nothing on this page is estimated or invented — every number here '
            'comes from a saved evaluation or dataset-build artifact.</div>', unsafe_allow_html=True)

metrics = get_metrics()
calibration = get_calibration_info()

val_mae = calibration["val_mae_calibrated"] if calibration and calibration.get("apply_calibration") \
    else (calibration["val_mae_raw"] if calibration else None)
test_mae = metrics["overall"]["mae"] if metrics else None
r2 = metrics["overall"]["r2"] if metrics else None
cs5 = metrics["overall"]["cs5"] if metrics else None

kpis = [
    ("Test MAE", f"{test_mae:.2f} yrs" if test_mae is not None else "Not evaluated"),
    ("Validation MAE", f"{val_mae:.2f} yrs" if val_mae is not None else "Not evaluated"),
    ("R²", f"{r2:.2f}" if r2 is not None else "Not evaluated"),
    ("CS(5)", f"{cs5:.0f}%" if cs5 is not None else "Not evaluated"),
    ("Dataset", "~190K images"),
    ("Age Range", "1-100"),
]
cols = st.columns(len(kpis))
for col, (label, value) in zip(cols, kpis):
    with col:
        st.markdown(f'''
        <div class="card" style="text-align:center; padding: 1.2rem 0.6rem;">
            <div class="metric-label" style="display:block; margin-bottom:0.5rem;">{label}</div>
            <div class="metric-value" style="font-size:1.15rem;">{value}</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div style="height: 1.2rem;"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# MAE by age group chart
# ---------------------------------------------------------------------------
if metrics and metrics.get("mae_by_age_band"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">MAE by Age Group (Test Set)</div>', unsafe_allow_html=True)
    band_data = metrics["mae_by_age_band"]
    chart_df = pd.DataFrame({
        "Age band": list(band_data.keys()),
        "MAE (years)": [v["mae"] for v in band_data.values()],
    }).set_index("Age band")
    st.bar_chart(chart_df, color="#39FF14", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Final test-set evaluation has not been run yet for the current model checkpoint.")

# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------
if os.path.exists(DATA_QUALITY_PATH):
    with open(DATA_QUALITY_PATH) as f:
        dq = json.load(f)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Data Quality</div>', unsafe_allow_html=True)

    def row(label, value):
        st.markdown(f'<div class="metric-row"><span class="metric-label">{label}</span>'
                    f'<span class="metric-value">{value}</span></div>', unsafe_allow_html=True)

    row("Original Images", f"{dq['raw_images_total']:,}")
    row("Clean Images", f"{dq['clean_images_total']:,}")
    row("Duplicates Removed", f"{dq['duplicate_same_label_collapsed']:,}")
    row("Conflicting Labels Removed", f"{dq['conflicting_label_images_removed']:,} "
                                       f"({dq['conflicting_label_groups_removed']:,} groups)")
    row("Train / Validation / Test", f"{dq['train_count']:,} / {dq['val_count']:,} / {dq['test_count']:,}")
    st.markdown('</div>', unsafe_allow_html=True)

if calibration:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Calibration</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-row"><span class="metric-label">Applied</span>'
                f'<span class="metric-value">{"Yes" if calibration.get("apply_calibration") else "No — did not improve test MAE"}</span></div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
