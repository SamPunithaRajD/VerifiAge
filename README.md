# VerifiAge — Age Prediction from Biometric Data

Estimates a person's age from a face photo. The system is built around one principle:
**it should never guess an age unless a reliable, single human face passed validation.**

Official model: **ResNet18** (ImageNet pretrained) + FastAPI backend + Streamlit UI.

## Pipeline

```
IMAGE -> IMAGE VALIDATION -> FACE DETECTION (YuNet) -> FACE VALIDATION -> QUALITY GATE
       -> SINGLE FACE CONFIRMATION -> CALIBRATED CROP -> RESNET18 -> AGE RESULT
```

The model never sees an image that fails the gate. Rejections are explicit, typed statuses
(`no_face`, `multiple_faces`, `low_quality`), never a guessed age.

**Framing calibration (verified, went through two wrong designs first — see Known
Limitations for the full incident writeup):** the crop fed to the model is not the raw
detected face box, and not the whole unprocessed image either. Measured directly across 148
training images: the detected face covers ~46% of frame width / ~61% of frame height on
average (std ~9%). `face_gate.crop_face` reproduces that same framing around the detected
face for *any* input photo, using margins calibrated from that measurement
(`MARGIN_RATIO_W=0.582`, `MARGIN_RATIO_H=0.314`). This makes the model's input consistent
with training regardless of how the source photo itself is composed — a tight face-only crop
and a face-in-a-large-photo both get normalized to roughly the same framing.

## Project structure

```
verifiage/
├── data/
│   ├── raw/        # extracted Kaggle dataset (gitignored)
│   ├── clean/       # clean_metadata.csv (deduped, leakage-checked)
│   └── splits/      # train.csv / val.csv / test.csv
├── models/
│   └── face_detection_yunet.onnx   # OpenCV YuNet face detector weights
├── ml/
│   ├── build_dataset.py   # scan raw -> dedup -> stratified split (cached)
│   ├── preprocessing.py   # SHARED transform used by training + inference
│   ├── face_gate.py       # detect -> validate -> quality gate -> single-face policy -> crop
│   ├── dataset.py
│   ├── model.py            # ResNet18 regression head
│   ├── metrics.py
│   ├── baseline.py         # mean-age baseline
│   ├── train.py            # differential LR, early stopping, LR scheduler, weighted sampler
│   ├── evaluate.py         # final TEST set metrics -> results/metrics.json
│   ├── calibrate.py        # bias correction fit on VAL, verified on TEST
│   ├── gradcam.py          # Grad-CAM explanation overlay (regression variant)
│   ├── predict.py          # single-image CLI inference (runs the full gate)
│   └── test_detector_sample.py   # YuNet sample test (faces vs objects/jewelry/patterns)
├── backend/
│   ├── main.py        # FastAPI: /health /predict /metrics /confidence /calibration
│   ├── inference.py    # reuses ml/predict.py
│   └── schemas.py
├── frontend/
│   ├── app.py                 # Streamlit entry point (st.navigation, persistent sidebar)
│   ├── pages_src/
│   │   ├── dashboard.py         # hero + KPI cards + system status + how-it-works
│   │   ├── face_analysis.py     # continuous live camera analysis
│   │   ├── upload_analysis.py   # upload-a-photo analysis
│   │   ├── architecture.py      # visual pipeline diagram
│   │   ├── kpis.py               # measured metrics, MAE-by-band chart, data quality
│   │   ├── settings.py           # non-technical info only
│   │   └── developer.py          # backend URL, health, thresholds (dev-only)
│   └── src/
│       ├── api_client.py       # calls the FastAPI backend
│       ├── theme.py            # shared black + neon-green dark UI CSS
│       ├── live_camera.py      # continuous getUserMedia camera + periodic inference
│       └── analysis_ui.py      # shared result/error rendering helpers
├── checkpoints/best_resnet18.pth
├── results/           # metrics.json, calibration.json, confidence_by_band.json
└── tests/
```

## Setup

```bash
pip install -r requirements.txt
```

Dataset: [mariafrenti/age-prediction](https://www.kaggle.com/datasets/mariafrenti/age-prediction) on Kaggle (downloaded via Kaggle API into `data/raw/`).

## Pipeline commands

```bash
python ml/build_dataset.py             # one-time, cached: clean + split the dataset
python ml/baseline.py                  # mean-age baseline MAE
python ml/test_detector_sample.py      # sanity-check YuNet on faces vs non-faces first
python ml/train.py --smoke             # sanity check: tiny subset, 1 epoch
python ml/train.py --epochs 2 --frac 0.2   # pilot run
python ml/train.py --epochs 30         # full run, early stopping -> checkpoints/best_resnet18.pth
python ml/train.py --epochs 30 --weighted-sampler  # A/B: age-balanced sampling
python ml/evaluate.py                  # final TEST metrics -> results/metrics.json
python ml/calibrate.py                 # bias correction fit on VAL, verified on TEST
python ml/predict.py path/to/image.jpg
python ml/gradcam.py path/to/image.jpg
```

## Backend

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8010
```

- `GET /health` — `{"status": "ok", "model_loaded": true}`
- `POST /predict` — multipart image upload. Response `status` is one of:
  - `success` — `{status, predicted_age, age_range, face_status, quality_status, model, confidence_pct, gradcam_base64}`
  - `no_face` / `multiple_faces` / `low_quality` — `{status, predicted_age: null, message}`
- `POST /predict?mock=true` — fixed sample response, no model needed (for frontend dev)
- `GET /metrics` — saved test-set metrics (MAE/RMSE/R2/CS(2,3,5), per-age-band bias)
- `GET /calibration` — bias-correction parameters and whether they're applied
- `GET /confidence` — empirical reliability-by-age-band table + methodology

## Frontend

```bash
cd frontend
python -m streamlit run app.py
```

Opens at `http://localhost:8501` with a persistent left sidebar (Dashboard / Face Analysis /
Upload Analysis / Architecture / KPIs / Settings / Developer Mode). Talks to the backend over
HTTP — no internal URLs, thresholds, or debug controls are shown outside Developer Mode. Face
detection/validation/quality gating happens entirely on the backend (single source of truth);
the frontend just renders whatever status the backend returns.

**Live camera:** `frontend/src/live_camera.py` embeds a custom HTML/JS component
(`getUserMedia` + periodic `fetch`) for true continuous camera tracking with a live neon
bounding box, not a 3-photo capture workflow. Inference runs periodically
(`INFER_INTERVAL_MS`) and the displayed age is smoothed over a rolling window
(`SMOOTHING_WINDOW`) directly in JS, without triggering a Streamlit rerun per frame.

## Face detection

**Detector: OpenCV YuNet** (`cv2.FaceDetectorYN`), replacing an earlier Haar Cascade approach
that produced false positives on non-face objects. Tested on 41 representative images before
being wired into the pipeline (`ml/test_detector_sample.py`): 20/20 real faces detected,
21/21 non-faces correctly rejected (0 false positives), including synthetic jewelry/ring/
locket/checkerboard/noise images specifically constructed to trip up a weaker detector.

Multi-signal validation gate (`ml/face_gate.py`): detector confidence, minimum face size,
aspect-ratio plausibility, in-bounds box, then a quality gate (blur via Laplacian variance,
brightness) before the model ever runs. Zero, one, or multiple valid faces are handled as
distinct explicit outcomes — never "pick the first/largest face."

## Results

ResNet18, full 133,326-image training set, 5 epochs (never triggered early stopping — still
improving each epoch, so more epochs would likely help further), differential LR
(backbone 1e-5, head 1e-4), CPU-only (measured, not assumed).

| | Value |
|---|---|
| Baseline (predict mean age) MAE | 11.41 years |
| Validation MAE (best epoch) | 6.01 years |
| **Test MAE** (held-out, 28,565 images) | **5.96 years** |
| Test RMSE | 8.12 years |
| Test R² | 0.694 |
| CS(2) / CS(3) / CS(5) | 24.2% / 35.3% / 54.0% |
| Mean signed error (bias) | -0.64 years (slight under-prediction) |

**Calibration was tested and correctly NOT applied**: a linear correction fit on the
validation set (`calibrated_age = 0.947 * pred + 2.60`) was verified against the held-out
test set and made MAE *slightly worse* (5.965 -> 5.984), even though it reduced the aggregate
bias closer to zero. Per the project rule ("only keep calibration if it genuinely improves
test MAE"), it was left off — the deployed model uses raw predictions. This is a sign the
model itself is now well-calibrated on average; earlier, less-trained versions did benefit
from calibration.

**MAE and bias by age band** (still shows the classic regression-to-the-mean pattern, reduced
but not eliminated by more data/training):

| Age band | n | MAE | Bias |
|---|---|---|---|
| 1-12 | 662 | 5.03 | +4.40 |
| 13-19 | 1343 | 7.54 | +7.09 |
| 20-29 | 7441 | 4.69 | +3.08 |
| 30-39 | 8483 | 4.91 | -0.49 |
| 40-49 | 5471 | 6.28 | -2.96 |
| 50-59 | 2805 | 7.78 | -4.66 |
| 60-69 | 1454 | 8.75 | -6.51 |
| 70-79 | 618 | 11.38 | -10.41 |
| 80-100 | 288 | 15.63 | -14.94 |

The model still hedges toward the dense middle of the training distribution (20-45), and the
under/over-represented age bands (children, elderly) have far fewer training images — this is
a data-imbalance effect, not something recalibration alone can fix (an age-balanced
`WeightedRandomSampler` is implemented and available via `--weighted-sampler` as a controlled
A/B experiment, but wasn't run for this final model given time constraints).

**Note on the table above:** it's measured via the training-matched raw-image path
(`evaluate.py`, same as training - no face detection involved). The table below is the same
age bands measured through the *actual deployed pipeline* (`ml/evaluate_live_pipeline.py`:
face detection + calibrated crop + model, sampled 150/band from the held-out test set) - this
is what real users experience, and is the honest number to trust for live behavior:

| Age band | n sampled | MAE | Bias |
|---|---|---|---|
| 1-12 | 150 | 5.09 | +4.70 |
| 13-19 | 150 | 6.56 | +6.23 |
| 20-29 | 150 | 4.29 | +1.92 |
| 30-39 | 150 | 5.49 | -0.93 |
| 40-49 | 150 | 6.57 | -3.93 |
| 50-59 | 150 | 8.01 | -5.73 |
| 60-69 | 150 | 9.34 | -7.36 |
| 70-79 | 150 | 11.74 | -10.78 |
| 80-100 | 150 | 16.68 | -16.32 |

Population-weighted overall MAE through the live pipeline: **6.12 years** (vs. 5.97 offline —
a small, acceptable cost for framing robustness on arbitrary real-world photos; see Known
Limitations for the full incident writeup).

## Known limitations

- Face detection favors precision by default (fewer false positives, occasionally misses a
  partially-occluded or extreme-angle face) — this is inherent to any single-pass detector at
  fixed default thresholds.
- Training compute is CPU-only on this machine (measured directly, not assumed) — this bounds
  how many epochs/how much data a "final" run can realistically cover under a hackathon
  deadline. See training run notes for the exact scope actually used.
- Gender is not predicted, and no identity/face-recognition of any kind is implemented — both
  intentionally out of scope.
- "Confidence" is a measured empirical reliability rate, not a calibrated per-image probability
  (see `/confidence` endpoint's `methodology` field for the exact definition).
- Age-balanced sampling (`--weighted-sampler`) is implemented but not yet run as a controlled
  comparison against the current model — a real next step for narrowing the age-band bias.
- **Even with the calibrated crop, a face that's a very small fraction of a much larger photo
  still degrades accuracy** (upsampling a small face region to the model's input size loses
  real detail) — verified: one such case improved from ~20 years predicted error to ~15-18,
  a real improvement but not a full fix. Users should expect somewhat lower accuracy on wide
  shots where the subject is small in frame; this is disclosed, not hidden.
- **Two real bugs found and fixed via a live production failure report** ("a clearly visible
  ~1-year-old was predicted as ~31"). Root-caused with a four-path reproduction (offline
  eval, `predict.py`, FastAPI, frontend) plus direct model-input inspection:
  1. An *intermediate* fix (feeding the model the whole unprocessed image) was correct for
     this dataset's own already-tightly-framed test images, but wrong in general: a
     realistic photo where the face is small in a larger frame was verified to reproduce
     the exact failure (~20 years predicted for a 1-year-old), because the model was never
     trained on that framing and defaults toward the dataset's overall mean age (~37) for
     unfamiliar compositions. Fixed by the calibrated crop described above, which normalizes
     *any* input photo's framing to match training - measured on 1,332 held-out samples
     (150/age-band): population-weighted MAE 5.97 -> 6.12 (a small, acceptable cost) while
     fixing the catastrophic small-face-in-frame case.
  2. `face_gate`'s bounds check was rejecting an entire high-confidence face detection (e.g.
     score 0.96) outright whenever the box overshot the frame edge by even 1-2px - normal
     detector jitter for faces near the edge of a tight photo, not a bad detection. This was
     silently producing false `no_face` rejections. Fixed by clamping the box to the frame
     instead of discarding the detection.
  Both fixes verified consistent across all four paths, verified not to reduce no-face/
  object-rejection accuracy (jewelry/pattern/noise synthetic tests re-run, still 0 false
  positives), and covered by a new regression test
  (`test_small_face_in_large_frame_does_not_produce_wild_prediction`).
