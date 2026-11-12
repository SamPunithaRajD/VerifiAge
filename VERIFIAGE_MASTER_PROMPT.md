# VerifiAge — Master Context Prompt

Paste everything below as the first message in a new Claude chat. It gives Claude full context
on the project so you can ask questions, get explanations for your team, or ask it to build a
PDF/slide deck — without re-explaining the project from scratch.

---

## PROMPT STARTS HERE

You are helping me (a solo builder) prepare to explain a project called **VerifiAge** to my
team of 7 people, and to answer my own follow-up questions about it. Read this context fully
before responding. Do not assume anything about the project beyond what's written here — if I
ask something not covered, say so rather than guessing.

### What VerifiAge is

VerifiAge estimates a person's **age from a face photo**. The single guiding principle behind
every design decision: **the system should never guess an age unless a reliable, single human
face has passed validation.** It explicitly refuses (with a clear reason) rather than silently
producing a bad guess for a missing face, multiple faces, or a low-quality image.

- **Official model: ResNet18** (ImageNet-pretrained, torchvision), single regression head
  outputting one number (age in years). This was a deliberate choice over a heavier model
  (EfficientNet was tried earlier and explicitly replaced) — ResNet18 is the permanent,
  final model for this project.
- **Face detector: OpenCV YuNet** (`cv2.FaceDetectorYN`), a CNN-based detector. It replaced
  an earlier Haar Cascade approach that produced false positives on non-face objects (jewelry,
  patterns, noise). YuNet was validated on 41 test images: 20/20 real faces detected, 21/21
  non-faces correctly rejected, 0 false positives.
- **Backend: FastAPI** — a stateless HTTP API (`/health`, `/predict`, `/metrics`,
  `/calibration`, `/confidence`).
- **Frontend: Streamlit** — a multi-page dashboard UI (black background, fluorescent-green
  accent, Apple-inspired restrained styling) with a persistent left sidebar.

### The full pipeline (every prediction goes through all of these gates)

```
IMAGE
  -> IMAGE VALIDATION
  -> FACE DETECTION (YuNet)
  -> FACE VALIDATION (detector confidence, min size, aspect ratio, in-bounds box)
  -> SINGLE-FACE POLICY (zero faces -> reject; 2+ faces -> reject; exactly 1 -> continue)
  -> QUALITY GATE (blur via Laplacian variance, brightness)
  -> CALIBRATED CROP (see below)
  -> RESNET18 REGRESSION
  -> AGE RESULT (+ optional Grad-CAM explanation, never part of the actual prediction path)
```

The model **never** sees an image that failed any gate. Rejections are explicit, typed
statuses (`no_face`, `multiple_faces`, `low_quality`) — never a silently guessed age.

### Why the "calibrated crop" exists (this is the single most important engineering story)

**The incident:** in a live test, a clearly visible ~1-year-old child was predicted as
~31 years old. This was treated as a critical production bug, not a training issue, and was
root-caused before any fix was attempted (per my own explicit rule: diagnose first, don't
just retrain or lower thresholds to make the symptom go away).

**Root cause:** an earlier fix had fed the model the *whole unprocessed photo*, believing a
too-tight crop was the problem. This worked fine on the training dataset's own photos (which
are already tightly framed around the face) but failed catastrophically on realistic photos
where the face is a small part of a larger frame — the model had never seen that framing
during training, so it defaulted toward the dataset's overall mean age (~37) instead of the
child's true age.

**The fix:** measured the *actual* face-to-frame ratio across 148 real training images
(the face covers ~46% of frame width / ~61% of frame height on average). The crop fed to the
model at inference time is calibrated to reproduce that same framing around whatever face
YuNet detects — not a tight box-only crop, and not the raw unprocessed photo. This makes a
tight selfie and a face-in-a-wide-shot both get normalized to the framing the model was
actually trained on.

**Verified cost/benefit:** population-weighted MAE went from 5.97 (offline, training-matched
path) to 6.12 (through the full real-world pipeline: detection + crop + model) — a small,
accepted cost for fixing the catastrophic small-face-in-frame failure mode.

**A second bug found during the same investigation:** the face-validation bounds check was
rejecting high-confidence detections (e.g. 0.96 confidence) outright whenever the detected box
overshot the frame edge by 1-2 pixels — normal detector jitter, not a bad detection. Fixed by
clamping the box to the frame instead of discarding the detection. Both fixes are covered by a
regression test (`test_small_face_in_large_frame_does_not_produce_wild_prediction`).

### Measured results (ResNet18, full 133,326-image training set, 5 epochs, CPU-only)

| Metric | Value |
|---|---|
| Baseline (predict mean age) MAE | 11.41 years |
| Validation MAE (best epoch) | 6.01 years |
| **Test MAE** (28,565 held-out images) | **5.96 years** |
| Test RMSE | 8.12 years |
| Test R² | 0.694 |
| CS(2) / CS(3) / CS(5) — % of predictions within N years | 24.2% / 35.3% / 54.0% |
| Mean signed bias | -0.64 years (slight under-prediction) |
| MAE through the actual live pipeline (detection + crop + model) | 6.12 years |

Accuracy is **not uniform across ages** — it degrades toward the extremes (children, elderly)
because those age bands are under-represented in training data. This is disclosed, not hidden,
in the UI (KPIs page shows MAE by age band).

**Calibration was tested and correctly NOT applied**: a linear bias-correction fit on
validation data made test MAE slightly *worse* (5.965 → 5.984) even though it reduced
aggregate bias — so per the project rule ("only keep it if it measurably improves test MAE"),
raw predictions are used in production.

### What "confidence" means (and does not mean)

The UI shows a High/Low confidence label. This is **not** a per-image probability from the
model — it's an empirical reliability rate: for the predicted age's 10-year band, what
percentage of test-set predictions in that band landed within ±3 years of the true age. This
was a deliberate choice to never fabricate a confidence number the model doesn't actually
produce.

### Explicitly out of scope (intentional, not oversights)

- No gender detection
- No identity / face recognition of any kind
- No "heavy" deepfake-detection model
- No unnecessary extra dashboards beyond what's needed to show the pipeline honestly

### Current product structure (Streamlit dashboard)

A persistent left sidebar (`VERIFIAGE` logo + fluorescent-green accent) with 7 pages:

1. **Dashboard** — hero, 5 KPI cards (model / test MAE / R² / CS(5) / dataset size), recent
   system status (backend online, model loaded), "How It Works" flow diagram, live-camera
   preview card.
2. **Face Analysis** — true continuous live camera (getUserMedia + periodic fetch to the
   backend, not a 3-photo capture flow), neon tracking box drawn from the backend's actual
   returned face coordinates, periodic ResNet18 inference with smoothed age display.
3. **Upload Analysis** — upload a photo, see the detected face box drawn on the image, get
   the same full result (age, confidence, error states) as live camera.
4. **Architecture** — a visual pipeline diagram (User → Upload/Camera → Validation → Face
   Detection → Face Validation → Quality Gate → Preprocessing → ResNet18 → Prediction →
   API Response → UI) plus the tech-stack row.
5. **KPIs** — the measured metrics table above, MAE-by-age-band bar chart, data-quality
   numbers (dataset cleaning stats), calibration status.
6. **Settings** — minimal, non-technical info only (product name, model name, explicit
   confirmation that gender/identity detection are not implemented).
7. **Developer Mode** — backend URL, health check JSON, face-gate thresholds, live-camera
   config constants. Everything technical/debug-facing is deliberately hidden from the other
   6 pages and only lives here.

**Explanation-only Grad-CAM**: a heatmap overlay showing which part of the face influenced the
prediction most. Strictly a separate forward/backward pass after the real prediction — it
never modifies or is part of the actual inference path.

### Data

- Source: [mariafrenti/age-prediction](https://www.kaggle.com/datasets/mariafrenti/age-prediction)
  on Kaggle.
- Raw: 273,640 images → after removing 0 corrupt images, collapsing 34,276 same-label
  duplicates, and removing 48,908 images from 18,429 conflicting-label groups → **190,456
  clean images**.
- Split: 133,326 train / 28,565 validation / 28,565 test.
- Age range: 1–100.

### Known limitations (say these plainly if asked — they were never hidden)

- Accuracy degrades on wide shots where the subject's face is small in the frame (upsampling
  a small face region loses real detail) — improved by the calibrated crop, not fully solved.
- Face detection favors precision by default — occasionally misses a partially-occluded or
  extreme-angle face rather than risk a false positive.
- Training was CPU-only on the available hardware — this bounded how many epochs a "final"
  run could realistically cover.
- Age-balanced sampling (`WeightedRandomSampler`) is implemented and was tested in a
  controlled pilot experiment but not run as the final production model, given time
  constraints (documented trade-off, not an oversight).
- MAE by age band shows the classic regression-to-the-mean pattern: children and elderly
  subjects (under-represented in training data) have measurably higher error than the
  20-45 range.

### Codebase layout (for reference, not needed for a team explanation)

```
frontend/   Streamlit dashboard (app.py, pages_src/*, src/*)
backend/    FastAPI (main.py, inference.py, schemas.py)
ml/         training, evaluation, face_gate.py, gradcam.py, calibrate.py, predict.py
checkpoints/best_resnet18.pth   production model weights
models/face_detection_yunet.onnx   YuNet face detector weights
results/    metrics.json, calibration.json, confidence_by_band.json, data_quality.json
tests/      pytest suite
```

### What I need from you in this chat

1. Answer my questions about any part of this project accurately, using only what's stated
   above (or ask me if something isn't covered here).
2. When I ask, help me build a **PDF or slide-style document for my team of 7** that explains:
   what VerifiAge does, why it's designed the way it is (especially the calibrated-crop bug
   story — it's the best example of rigorous debugging in this project), the measured results,
   what it deliberately doesn't do, and the current dashboard structure. Keep it honest — real
   measured numbers only, no invented metrics, and don't overstate accuracy (5.96 years MAE
   overall, worse at the extremes, is the true number to present).
3. Match the tone to a technical team audience — assume they can read a metrics table and
   understand terms like MAE/R², but don't assume they've seen the codebase.

## PROMPT ENDS HERE
