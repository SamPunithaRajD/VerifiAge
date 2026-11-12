"""Apple-inspired dark UI: minimal, cinematic, restrained. Fluorescent green is used
sparingly as the signature accent, not as a neon-everywhere effect."""

ACCENT = "#39FF14"
BG = "#050505"
SURFACE = "#0B0B0D"
TEXT = "#F5F5F7"
MUTED = "#86868B"

CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {{
        --bg: {BG};
        --surface: {SURFACE};
        --surface-glass: rgba(11,11,13,0.6);
        --border: rgba(255,255,255,0.08);
        --border-strong: rgba(255,255,255,0.14);
        --text: {TEXT};
        --muted: {MUTED};
        --accent: {ACCENT};
        --accent-dim: rgba(57,255,20,0.12);
        --danger: #FF453A;
        --danger-dim: rgba(255,69,58,0.1);
        --warning: #FF9F0A;
        --warning-dim: rgba(255,159,10,0.1);
    }}
    html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
    .stApp {{ background: radial-gradient(ellipse at top, #0a0a0c 0%, var(--bg) 55%); color: var(--text); }}
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    header[data-testid="stHeader"] [data-testid="stToolbar"] {{ visibility: hidden; }}
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {{ visibility: visible !important; }}
    .block-container {{
        padding-top: 2rem; padding-bottom: 2rem;
        padding-left: 2.5rem; padding-right: 2.5rem;
        max-width: 100%;
    }}
    section[data-testid="stSidebar"] {{ position: sticky; top: 0; height: 100vh; }}

    * {{ transition: color 0.2s ease, background-color 0.2s ease, border-color 0.2s ease, opacity 0.25s ease; }}

    /* ---------- Hero / landing ---------- */
    .hero-wrap {{ padding: 3.5rem 0 2rem 0; }}
    .hero-eyebrow {{
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.18em; color: var(--accent);
        text-transform: uppercase; margin-bottom: 1.1rem; display: flex; align-items: center; gap: 0.5rem;
    }}
    .hero-eyebrow .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
        box-shadow: 0 0 8px var(--accent); }}
    .hero-title {{
        font-size: 4.4rem; font-weight: 700; letter-spacing: -0.03em; color: var(--text);
        margin-bottom: 0.6rem; line-height: 1.02;
    }}
    .hero-headline {{
        font-size: 2.1rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text);
        line-height: 1.25; margin-bottom: 1.1rem; max-width: 640px;
    }}
    .hero-sub {{
        font-size: 1.08rem; color: var(--muted); max-width: 560px; line-height: 1.6;
        font-weight: 400; margin-bottom: 2.2rem;
    }}
    .hero-disclaimer {{
        font-size: 0.8rem; color: var(--muted); border-top: 1px solid var(--border);
        padding-top: 1.2rem; margin-top: 2.5rem;
    }}

    /* subtle animated face-scan visual */
    .scan-frame {{
        position: relative; width: 100%; max-width: 340px; aspect-ratio: 1/1; margin: 0 auto;
        border-radius: 24px; background: linear-gradient(145deg, #0d0d10, #08080a);
        border: 1px solid var(--border); overflow: hidden;
    }}
    .scan-corner {{
        position: absolute; width: 26px; height: 26px; border: 2px solid var(--accent);
        opacity: 0.85; filter: drop-shadow(0 0 6px rgba(57,255,20,0.35));
    }}
    .scan-corner.tl {{ top: 22px; left: 22px; border-right: none; border-bottom: none; border-radius: 6px 0 0 0; }}
    .scan-corner.tr {{ top: 22px; right: 22px; border-left: none; border-bottom: none; border-radius: 0 6px 0 0; }}
    .scan-corner.bl {{ bottom: 22px; left: 22px; border-right: none; border-top: none; border-radius: 0 0 0 6px; }}
    .scan-corner.br {{ bottom: 22px; right: 22px; border-left: none; border-top: none; border-radius: 0 0 6px 0; }}
    .scan-line {{
        position: absolute; left: 8%; right: 8%; height: 1.5px;
        background: linear-gradient(90deg, transparent, var(--accent) 20%, var(--accent) 80%, transparent);
        box-shadow: 0 0 12px 1px rgba(57,255,20,0.45);
        animation: scan-move 3.2s ease-in-out infinite;
    }}
    @keyframes scan-move {{
        0%   {{ top: 18%; opacity: 0; }}
        10%  {{ opacity: 1; }}
        50%  {{ top: 82%; opacity: 1; }}
        90%  {{ opacity: 1; }}
        100% {{ top: 18%; opacity: 0; }}
    }}
    .scan-face-icon {{ position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
        opacity: 0.15; font-size: 5.5rem; }}

    /* ---------- Section headings ---------- */
    .section-eyebrow {{
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 0.6rem;
    }}
    .section-title {{ font-size: 1.9rem; font-weight: 650; letter-spacing: -0.02em; color: var(--text); margin-bottom: 0.3rem; }}
    .section-sub {{ font-size: 0.98rem; color: var(--muted); margin-bottom: 1.6rem; max-width: 560px; }}

    .page-title {{ font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); margin-bottom: 0.2rem; }}
    .page-sub {{ font-size: 0.95rem; color: var(--muted); margin-bottom: 1.2rem; }}

    /* ---------- Glass cards ---------- */
    .card {{
        background: var(--surface-glass); backdrop-filter: blur(20px) saturate(140%);
        -webkit-backdrop-filter: blur(20px) saturate(140%);
        border: 1px solid var(--border); border-radius: 18px;
        padding: 1.5rem 1.7rem; margin-bottom: 1rem;
    }}
    .card:hover {{ border-color: var(--border-strong); }}

    .section-heading {{
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--muted); margin-bottom: 0.9rem;
    }}

    /* ---------- Status pills ---------- */
    .status-pill {{
        display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0.85rem;
        border-radius: 100px; font-size: 0.82rem; font-weight: 500;
        background: rgba(255,255,255,0.05); border: 1px solid var(--border);
    }}
    .status-pill.ok {{ background: var(--accent-dim); color: var(--accent); border-color: rgba(57,255,20,0.25); }}
    .status-pill.warn {{ background: var(--warning-dim); color: var(--warning); border-color: rgba(255,159,10,0.25); }}
    .status-pill.bad {{ background: var(--danger-dim); color: var(--danger); border-color: rgba(255,69,58,0.25); }}
    .status-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

    /* ---------- Metrics ---------- */
    .metric-row {{ display: flex; justify-content: space-between; align-items: baseline; padding: 0.65rem 0; border-bottom: 1px solid var(--border); }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-label {{ font-size: 0.84rem; color: var(--muted); font-weight: 400; letter-spacing: 0.01em; }}
    .metric-value {{ font-size: 1rem; color: var(--text); font-weight: 600; }}
    .metric-value.big {{
        font-size: 4rem; font-weight: 700; color: var(--text); letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
    }}
    .metric-value.big .accent-num {{ color: var(--accent); }}

    .result-message {{ font-size: 0.94rem; color: var(--muted); line-height: 1.55; }}

    .state-banner {{
        border-radius: 14px; padding: 1.1rem 1.4rem; font-size: 0.92rem; font-weight: 400;
        border: 1px solid var(--border); background: var(--surface-glass); color: var(--muted);
        margin-bottom: 1rem;
    }}

    /* ---------- Product story blocks ---------- */
    .story-block {{
        background: var(--surface-glass); border: 1px solid var(--border); border-radius: 18px;
        padding: 1.8rem 1.6rem; height: 100%;
    }}
    .story-block:hover {{ border-color: rgba(57,255,20,0.3); }}
    .story-index {{ font-size: 0.75rem; color: var(--accent); font-weight: 600; letter-spacing: 0.1em; margin-bottom: 0.8rem; }}
    .story-title {{ font-size: 1.15rem; font-weight: 650; color: var(--text); margin-bottom: 0.5rem; }}
    .story-desc {{ font-size: 0.92rem; color: var(--muted); line-height: 1.5; }}

    /* ---------- Buttons / inputs ---------- */
    div[data-testid="stFileUploader"] section {{
        background: var(--surface-glass); border: 1px dashed var(--border-strong); border-radius: 16px;
    }}
    div[data-testid="stSidebar"] {{ background-color: var(--surface); border-right: 1px solid var(--border); }}
    .stButton > button {{
        border-radius: 980px; font-weight: 500; border: 1px solid var(--border-strong); color: var(--text);
        background: rgba(255,255,255,0.04); padding: 0.55rem 1.3rem;
    }}
    .stButton > button:hover {{ border-color: var(--accent); background: var(--accent-dim); color: var(--accent); }}
    .stButton > button[kind="primary"] {{ background: var(--accent); border-color: var(--accent); color: #050505; font-weight: 600; }}
    .stButton > button[kind="primary"]:hover {{ background: #4dff33; color: #050505; }}
    .stTabs [data-baseweb="tab"] {{ color: var(--muted); font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}

    /* fade-in reveal for results */
    .fade-in {{ animation: fadeIn 0.45s ease; }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: translateY(0); }} }}
</style>
"""


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)


def status_pill(label, kind="ok"):
    """kind: ok | warn | bad"""
    return f'<span class="status-pill {kind}"><span class="status-dot"></span>{label}</span>'
