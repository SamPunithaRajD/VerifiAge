"""
Continuous live-camera analysis. Streamlit's built-in camera_input only takes single
snapshots on click, so this is a self-contained HTML/JS component (getUserMedia +
periodic fetch() calls to the FastAPI backend) instead - the whole point being real
continuous tracking, not a "take N photos" workflow.

Both the camera feed AND the results panel live inside this one component (not split
across Streamlit reruns), so updates are smooth/flicker-free and the result panel sits
directly beside the camera with no page reruns needed.
"""

INFER_INTERVAL_MS = 400   # how often a frame is sent for inference (not every frame)
SMOOTHING_WINDOW = 5      # rolling average window for the displayed age
STABILITY_SPREAD_YEARS = 4  # window spread below this counts as "Stable"


def render(st, backend_url, height=560):
    html = f"""
<div id="verifiage-live-root" style="font-family: 'Inter', -apple-system, sans-serif; color: #F5F5F7;">
  <style>
    #verifiage-live-root {{ display: flex; gap: 16px; align-items: stretch; }}
    .vg-left {{ flex: 1.25; position: relative; background: #0B0B0B; border: 1px solid rgba(255,255,255,0.08);
                border-radius: 20px; overflow: hidden; min-height: 460px; }}
    .vg-right {{ flex: 0.85; background: rgba(11,11,13,0.6); backdrop-filter: blur(20px);
                 border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
                 padding: 26px 24px; display: flex; flex-direction: column; gap: 4px; }}
    .vg-video-wrap {{ position: relative; width: 100%; height: 100%; }}
    .vg-video-wrap video {{ width: 100%; height: 100%; object-fit: cover; display: block;
                              filter: saturate(1.02); }}
    .vg-box {{
      position: absolute; border: 1.5px solid rgba(57,255,20,0.85); border-radius: 12px;
      display: none; pointer-events: none;
      transition: left 0.25s cubic-bezier(.4,0,.2,1), top 0.25s cubic-bezier(.4,0,.2,1),
                  width 0.25s cubic-bezier(.4,0,.2,1), height 0.25s cubic-bezier(.4,0,.2,1),
                  opacity 0.3s ease;
    }}
    .vg-heading {{ font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: #86868B;
                   font-weight: 600; margin-bottom: 8px; }}
    .vg-row {{ display: flex; justify-content: space-between; align-items: center; padding: 9px 0;
               border-bottom: 1px solid rgba(255,255,255,0.07); }}
    .vg-row:last-child {{ border-bottom: none; }}
    .vg-label {{ color: #86868B; font-size: 13px; font-weight: 400; }}
    .vg-value {{ font-weight: 500; font-size: 13px; padding: 3px 10px; border-radius: 100px;
                 background: rgba(255,255,255,0.05); transition: all 0.3s ease; }}
    .vg-value.ok {{ color: #39FF14; background: rgba(57,255,20,0.1); }}
    .vg-value.warn {{ color: #FF9F0A; background: rgba(255,159,10,0.1); }}
    .vg-value.bad {{ color: #FF453A; background: rgba(255,69,58,0.1); }}
    .vg-value.plain {{ color: #F5F5F7; background: rgba(255,255,255,0.05); }}
    .vg-age {{ font-size: 3.6rem; font-weight: 700; color: #F5F5F7; letter-spacing: -0.03em;
               text-align: center; margin: 8px 0 0 0; font-variant-numeric: tabular-nums;
               transition: opacity 0.25s ease; }}
    .vg-age-label {{ text-align: center; color: #86868B; font-size: 11px; letter-spacing: 0.12em;
                      text-transform: uppercase; margin-bottom: 2px; margin-top: 10px; }}
    .vg-range {{ text-align: center; color: #86868B; font-size: 13px; margin-bottom: 6px; }}
    .vg-msg {{ color: #86868B; font-size: 12px; line-height: 1.5; text-align: center; padding: 6px 0 0 0; min-height: 1.2em; }}
    .vg-overlay-msg {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
                        text-align: center; color: #86868B; font-size: 14px; padding: 24px;
                        background: rgba(5,5,5,0.55); backdrop-filter: blur(6px); }}
  </style>

  <div class="vg-left">
    <div class="vg-video-wrap">
      <video id="vg-video" autoplay playsinline muted></video>
      <div id="vg-box" class="vg-box"></div>
      <div id="vg-overlay-msg" class="vg-overlay-msg">Requesting camera access...</div>
    </div>
  </div>

  <div class="vg-right">
    <div class="vg-heading">Live Analysis</div>
    <div class="vg-row"><span class="vg-label">Face Status</span><span class="vg-value" id="vg-face">Idle</span></div>
    <div class="vg-row"><span class="vg-label">Image Quality</span><span class="vg-value" id="vg-quality">-</span></div>
    <div class="vg-row"><span class="vg-label">Stability</span><span class="vg-value" id="vg-stability">-</span></div>
    <div class="vg-row"><span class="vg-label">Model</span><span class="vg-value plain">ResNet18</span></div>
    <div style="flex:1; display:flex; flex-direction:column; justify-content:center;">
      <div class="vg-age-label">Live Age</div>
      <div class="vg-age" id="vg-age">--</div>
      <div class="vg-range" id="vg-range">&nbsp;</div>
      <div class="vg-msg" id="vg-msg"></div>
    </div>
  </div>
</div>

<script>
(function() {{
  const BACKEND_URL = "{backend_url}";
  const INFER_INTERVAL_MS = {INFER_INTERVAL_MS};
  const SMOOTHING_WINDOW = {SMOOTHING_WINDOW};
  const STABILITY_SPREAD = {STABILITY_SPREAD_YEARS};

  const video = document.getElementById('vg-video');
  const box = document.getElementById('vg-box');
  const overlayMsg = document.getElementById('vg-overlay-msg');
  const faceEl = document.getElementById('vg-face');
  const qualityEl = document.getElementById('vg-quality');
  const stabilityEl = document.getElementById('vg-stability');
  const ageEl = document.getElementById('vg-age');
  const rangeEl = document.getElementById('vg-range');
  const msgEl = document.getElementById('vg-msg');

  const captureCanvas = document.createElement('canvas');
  const ctx = captureCanvas.getContext('2d');

  let ageBuffer = [];
  let busy = false;

  function setValueClass(el, cls) {{ el.className = 'vg-value ' + cls; }}

  function resetPanel(faceText, faceCls, msg) {{
    faceEl.textContent = faceText; setValueClass(faceEl, faceCls);
    qualityEl.textContent = '-'; setValueClass(qualityEl, '');
    stabilityEl.textContent = '-'; setValueClass(stabilityEl, '');
    ageEl.style.opacity = 0.3; ageEl.textContent = '--';
    rangeEl.textContent = '\\u00A0';
    msgEl.textContent = msg || '';
    box.style.display = 'none';
    ageBuffer = [];
  }}

  async function startCamera() {{
    try {{
      const stream = await navigator.mediaDevices.getUserMedia({{
        video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }} }}, audio: false
      }});
      video.srcObject = stream;
      overlayMsg.style.display = 'none';
      resetPanel('Waiting', '', 'Position your face inside the frame.');
      setInterval(captureAndInfer, INFER_INTERVAL_MS);
    }} catch (err) {{
      overlayMsg.textContent = 'Camera unavailable - access may be blocked by your browser or this environment.';
    }}
  }}

  function drawBox(faceBox) {{
    if (!faceBox || !video.videoWidth) {{ box.style.display = 'none'; return; }}
    const scaleX = video.clientWidth / video.videoWidth;
    const scaleY = video.clientHeight / video.videoHeight;
    const [x, y, w, h] = faceBox;
    box.style.left = (x * scaleX) + 'px';
    box.style.top = (y * scaleY) + 'px';
    box.style.width = (w * scaleX) + 'px';
    box.style.height = (h * scaleY) + 'px';
    box.style.display = 'block';
  }}

  function smoothedAge() {{
    return ageBuffer.reduce((a, b) => a + b, 0) / ageBuffer.length;
  }}

  async function captureAndInfer() {{
    if (busy || !video.videoWidth) return;
    busy = true;
    try {{
      captureCanvas.width = video.videoWidth;
      captureCanvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

      const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, 'image/jpeg', 0.85));
      if (!blob) return;

      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      const resp = await fetch(BACKEND_URL + '/predict?live=true', {{ method: 'POST', body: formData }});
      const data = await resp.json();

      if (data.status === 'success') {{
        faceEl.textContent = 'Tracking'; setValueClass(faceEl, 'ok');
        const qOk = data.quality_status === 'PASS';
        qualityEl.textContent = qOk ? 'Good' : 'Low';
        setValueClass(qualityEl, qOk ? 'ok' : 'warn');
        msgEl.textContent = qOk ? '' : (data.quality_message || '');

        ageBuffer.push(data.predicted_age);
        if (ageBuffer.length > SMOOTHING_WINDOW) ageBuffer.shift();

        if (ageBuffer.length >= 3) {{
          const spread = Math.max(...ageBuffer) - Math.min(...ageBuffer);
          const stable = spread <= STABILITY_SPREAD;
          stabilityEl.textContent = stable ? 'Stable' : 'Uncertain';
          setValueClass(stabilityEl, stable ? 'ok' : 'warn');
        }} else {{
          stabilityEl.textContent = 'Reading...'; setValueClass(stabilityEl, '');
        }}

        ageEl.style.opacity = 1;
        ageEl.textContent = smoothedAge().toFixed(0);
        rangeEl.textContent = data.age_range || '\\u00A0';
        drawBox(data.face_box);
      }} else if (data.status === 'no_face') {{
        resetPanel('No Face', 'bad', 'No reliable human face detected.');
      }} else if (data.status === 'multiple_faces') {{
        resetPanel('Multiple', 'bad', 'Multiple faces detected.');
      }} else if (data.status === 'low_quality') {{
        faceEl.textContent = 'Detected'; setValueClass(faceEl, 'warn');
        qualityEl.textContent = 'Low'; setValueClass(qualityEl, 'bad');
        stabilityEl.textContent = '-'; setValueClass(stabilityEl, '');
        ageEl.style.opacity = 0.3; ageEl.textContent = '--';
        rangeEl.textContent = '\\u00A0';
        msgEl.textContent = 'Image quality is insufficient.';
        box.style.display = 'none';
        ageBuffer = [];
      }}
    }} catch (err) {{
      msgEl.textContent = 'Reconnecting...';
    }} finally {{
      busy = false;
    }}
  }}

  startCamera();
}})();
</script>
"""
    st.components.v1.html(html, height=height, scrolling=False)
