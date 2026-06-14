"""
╔══════════════════════════════════════════════════════════════╗
║   Sentiment Analysis — Streamlit App                         ║
║   Powered by sentiment_model_v2.py  (3-Layer MLP, NumPy)    ║
║                                                              ║
║   HOW TO RUN:                                                ║
║     pip install streamlit numpy                              ║
║     streamlit run app.py                                     ║
║                                                              ║
║   Make sure app.py and sentiment_model_v2.py are in the      ║
║   SAME folder before running.                                ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import time
import sys
import os

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="SentimentAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Train once, cache forever for the session."""
    # Import from the same directory as this script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import sentiment_model_v2 as sm
        sm._init()          # triggers training
        return sm
    except ImportError:
        st.error("❌  Could not find `sentiment_model_v2.py`.\n\n"
                 "Place both files in the same folder and re-run.")
        st.stop()

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin-bottom: 0.4rem;
}
.hero-title span { color: #a78bfa; }
.hero-subtitle {
    color: #8888aa;
    font-size: 1rem;
    margin-bottom: 0.8rem;
}
.arch-pills {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 12px;
}
.arch-pill {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 12px;
    color: #a78bfa;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 11px;
    color: #666688;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
}

/* ── Verdict badge ── */
.verdict-pos { background: rgba(34,212,126,0.12); border: 1px solid rgba(34,212,126,0.3); border-radius: 16px; padding: 1.2rem 1.6rem; display: flex; align-items: center; gap: 14px; }
.verdict-neg { background: rgba(245,90,90,0.12);  border: 1px solid rgba(245,90,90,0.3);  border-radius: 16px; padding: 1.2rem 1.6rem; display: flex; align-items: center; gap: 14px; }
.verdict-neu { background: rgba(108,143,255,0.12);border: 1px solid rgba(108,143,255,0.3);border-radius: 16px; padding: 1.2rem 1.6rem; display: flex; align-items: center; gap: 14px; }
.v-emoji  { font-size: 3rem; line-height: 1; }
.v-label  { flex: 1; }
.v-label h2 { font-size: 1.6rem; font-weight: 700; color: #fff; margin: 0; text-transform: capitalize; }
.v-label p  { color: #9999bb; font-size: 0.875rem; margin: 4px 0 0; }
.v-conf-pos { background: rgba(34,212,126,0.15);  color: #22d47e; border-radius: 99px; padding: 6px 16px; font-size: 1.1rem; font-weight: 700; }
.v-conf-neg { background: rgba(245,90,90,0.15);   color: #f55a5a; border-radius: 99px; padding: 6px 16px; font-size: 1.1rem; font-weight: 700; }
.v-conf-neu { background: rgba(108,143,255,0.15); color: #6c8fff; border-radius: 99px; padding: 6px 16px; font-size: 1.1rem; font-weight: 700; }

/* ── Metric mini cards ── */
.mini-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
}
.mini-label { font-size: 11px; color: #666688; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
.mini-val   { font-size: 1.5rem; font-weight: 700; color: #e8e8f5; }
.mini-sub   { font-size: 11px; color: #555577; margin-top: 3px; }

/* ── Progress bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-label { font-size: 13px; color: #aaaacc; width: 72px; }
.prob-bg    { flex: 1; height: 8px; background: rgba(255,255,255,0.07); border-radius: 99px; overflow: hidden; }
.prob-fill-pos { height: 100%; background: #22d47e; border-radius: 99px; transition: width 0.8s; }
.prob-fill-neg { height: 100%; background: #f55a5a; border-radius: 99px; transition: width 0.8s; }
.prob-fill-neu { height: 100%; background: #6c8fff; border-radius: 99px; transition: width 0.8s; }
.prob-pct-pos { font-size: 13px; font-weight: 600; color: #22d47e; width: 46px; text-align: right; }
.prob-pct-neg { font-size: 13px; font-weight: 600; color: #f55a5a; width: 46px; text-align: right; }
.prob-pct-neu { font-size: 13px; font-weight: 600; color: #6c8fff; width: 46px; text-align: right; }

/* ── Token chips ── */
.token-cloud { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.tok-pos { background: rgba(34,212,126,0.15);  border: 1px solid rgba(34,212,126,0.3);  color: #22d47e; border-radius: 8px; padding: 5px 12px; font-size: 13px; font-weight: 500; }
.tok-neg { background: rgba(245,90,90,0.15);   border: 1px solid rgba(245,90,90,0.3);   color: #f55a5a; border-radius: 8px; padding: 5px 12px; font-size: 13px; font-weight: 500; }
.tok-neu { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12);color: #9999bb; border-radius: 8px; padding: 5px 12px; font-size: 13px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,10,25,0.85) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
.sidebar-title {
    font-size: 13px;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}
.example-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #ccccee;
    cursor: pointer;
    line-height: 1.4;
}

/* ── History row ── */
.hist-row {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.hist-emoji { font-size: 1.2rem; flex-shrink: 0; }
.hist-body  { flex: 1; overflow: hidden; }
.hist-text  { font-size: 13px; color: #ccccee; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.hist-meta  { font-size: 11px; color: #666688; margin-top: 2px; }
.hist-badge-pos { background: rgba(34,212,126,0.12);  color: #22d47e; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 99px; flex-shrink: 0; }
.hist-badge-neg { background: rgba(245,90,90,0.12);   color: #f55a5a; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 99px; flex-shrink: 0; }
.hist-badge-neu { background: rgba(108,143,255,0.12); color: #6c8fff; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 99px; flex-shrink: 0; }

/* ── Divider ── */
.section-divider { height: 1px; background: rgba(255,255,255,0.06); margin: 1.2rem 0; }

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8e8f5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 2px rgba(167,139,250,0.15) !important;
}
.stButton > button {
    background: #a78bfa !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.6rem 1.6rem !important;
    width: 100% !important;
    transition: filter 0.2s !important;
}
.stButton > button:hover { filter: brightness(1.15) !important; }
div[data-testid="stSelectbox"] > div { background: rgba(255,255,255,0.05) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────
if "history"       not in st.session_state: st.session_state.history = []
if "input_text"    not in st.session_state: st.session_state.input_text = ""
if "last_result"   not in st.session_state: st.session_state.last_result = None
if "model_trained" not in st.session_state: st.session_state.model_trained = False

# ── Load/train model ────────────────────────────────────────────────────────
with st.spinner("🧠  Training model on 400 samples — takes ~5 seconds…"):
    sm = load_model()
    st.session_state.model_trained = True

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Understand <span>how text feels</span></div>
    <div class="hero-subtitle">
        Deep NLP model — trained on 400 labeled samples. Paste any text to decode its emotional tone.
    </div>
    <div class="arch-pills">
        <span class="arch-pill">Feature Extraction</span>
        <span class="arch-pill">3-Layer MLP</span>
        <span class="arch-pill">ReLU Activations</span>
        <span class="arch-pill">Softmax Output</span>
        <span class="arch-pill">60 → 128 → 64 → 3</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚡ Quick Examples</div>', unsafe_allow_html=True)

    EXAMPLES = {
        "😊 Very positive":   "I absolutely love this product, it changed my life! Incredible quality and fantastic service.",
        "😞 Very negative":   "Worst purchase ever. Completely broken on arrival and customer service was absolutely terrible.",
        "😐 Neutral":         "The product arrived on time. It works as described. Nothing special but no complaints either.",
        "🎭 Mixed signals":   "The design is beautiful but the performance is quite disappointing. I have mixed feelings overall.",
        "💼 Professional":    "The quarterly results were in line with expectations. Revenue remained stable with moderate growth.",
        "🌟 Enthusiastic":    "This is utterly phenomenal! I have never been so impressed. Absolutely brilliant in every way!",
        "😤 Frustrated":      "I cannot believe how bad this is. Completely useless, total waste of money. Never again!",
        "📦 Product review":  "Decent build quality for the price. Shipping was a bit slow but packaging was fine. Average overall.",
    }

    for label, text in EXAMPLES.items():
        if st.button(label, key=f"ex_{label}", use_container_width=True):
            st.session_state.input_text = text
            st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Options</div>', unsafe_allow_html=True)
    show_tokens   = st.toggle("Show attention tokens",    value=True)
    show_probs    = st.toggle("Show class probabilities", value=True)
    show_history  = st.toggle("Show analysis history",   value=True)
    show_rawjson  = st.toggle("Show raw JSON output",    value=False)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size:12px;color:#666688;line-height:1.8">
    <b style="color:#9999bb">Architecture</b><br>60 → 128 → 64 → 3<br><br>
    <b style="color:#9999bb">Training Data</b><br>400 labeled samples<br><br>
    <b style="color:#9999bb">Classes</b><br>Positive · Negative · Neutral<br><br>
    <b style="color:#9999bb">Optimizer</b><br>Mini-batch SGD<br><br>
    <b style="color:#9999bb">Epochs</b><br>80 · LR 0.008<br><br>
    <b style="color:#9999bb">Backend</b><br>Pure NumPy (no TF/PyTorch)
</div>
""", unsafe_allow_html=True)

# ── Main layout ─────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

# ── LEFT: Input ──────────────────────────────────────────────────────────────
with col_input:
    st.markdown('<div class="card-title">✍️  INPUT TEXT</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        label="",
        value=st.session_state.input_text,
        placeholder="Type or paste any text here — a product review, tweet, email, feedback…\n\nPress Ctrl+Enter or click Analyse to see results.",
        height=180,
        max_chars=1000,
        key="text_area_main",
        label_visibility="collapsed",
    )

    wc = len(text_input.split()) if text_input.strip() else 0
    st.markdown(
        f'<div style="font-size:12px;color:#555577;text-align:right;margin-top:-8px;margin-bottom:8px">'
        f'{len(text_input)} chars · {wc} words</div>',
        unsafe_allow_html=True
    )

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        analyse_btn = st.button("🔍  Analyse Sentiment", use_container_width=True)
    with btn_col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.input_text = ""
        st.session_state.last_result = None
        st.rerun()

    # ── Bulk / batch mode ──────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋  BATCH MODE — analyse multiple lines at once</div>',
                unsafe_allow_html=True)
    batch_input = st.text_area(
        label="",
        placeholder="Paste multiple texts, one per line:\n\nI love this product!\nTerrible experience.\nIt was okay I guess.",
        height=130,
        key="batch_area",
        label_visibility="collapsed",
    )
    batch_btn = st.button("📊  Analyse All Lines", use_container_width=True)

# ── RIGHT: Results ───────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="card-title">📈  RESULTS</div>', unsafe_allow_html=True)

    result_placeholder = st.empty()

    def render_result(res):
        label = res["sentiment"]
        conf  = res["confidence"]
        emoji = res["emoji"]
        probs = res["probabilities"]
        cls   = {"positive":"pos","negative":"neg","neutral":"neu"}[label]

        sub_map = {
            "positive": "Strong positive emotional signal detected",
            "negative": "Critical negative sentiment identified",
            "neutral":  "Balanced, low-polarity language detected",
        }

        # ── Verdict ──────────────────────────────────────────────────────
        result_placeholder.markdown(f"""
<div class="verdict-{cls}">
    <div class="v-emoji">{emoji}</div>
    <div class="v-label">
        <h2>{label}</h2>
        <p>{sub_map[label]}</p>
    </div>
    <div class="v-conf-{cls}">{conf}%</div>
</div>
""", unsafe_allow_html=True)

        # ── Metrics ───────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="mini-card"><div class="mini-label">Confidence</div>'
                        f'<div class="mini-val">{conf}%</div>'
                        f'<div class="mini-sub">model certainty</div></div>',
                        unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="mini-card"><div class="mini-label">Intensity</div>'
                        f'<div class="mini-val">{res["intensity"]}</div>'
                        f'<div class="mini-sub">emotional weight</div></div>',
                        unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="mini-card"><div class="mini-label">Tokens</div>'
                        f'<div class="mini-val">{res["token_count"]}</div>'
                        f'<div class="mini-sub">after filtering</div></div>',
                        unsafe_allow_html=True)

        # ── Probabilities ─────────────────────────────────────────────────
        if show_probs:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">CLASS PROBABILITIES</div>',
                        unsafe_allow_html=True)
            for lbl, fill_cls, pct in [
                ("Positive", "pos", probs["positive"]),
                ("Negative", "neg", probs["negative"]),
                ("Neutral",  "neu", probs["neutral"]),
            ]:
                st.markdown(f"""
<div class="prob-row">
    <span class="prob-label">{lbl}</span>
    <div class="prob-bg"><div class="prob-fill-{fill_cls}" style="width:{pct}%"></div></div>
    <span class="prob-pct-{fill_cls}">{pct}%</span>
</div>""", unsafe_allow_html=True)

        # ── Attention tokens ──────────────────────────────────────────────
        if show_tokens and res["attention"]:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">ATTENTION HEATMAP — influential tokens</div>',
                        unsafe_allow_html=True)
            POS_W = sm.POS_WORDS
            NEG_W = sm.NEG_WORDS
            chips = ""
            for item in res["attention"]:
                w = item["word"]
                tok_cls = "tok-pos" if w in POS_W else "tok-neg" if w in NEG_W else "tok-neu"
                chips += f'<span class="{tok_cls}">{w}</span>'
            st.markdown(f'<div class="token-cloud">{chips}</div>',
                        unsafe_allow_html=True)

        # ── Raw JSON ──────────────────────────────────────────────────────
        if show_rawjson:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="card-title">RAW JSON OUTPUT</div>',
                        unsafe_allow_html=True)
            import json
            st.code(json.dumps(res, indent=2), language="json")

    # ── Run single analysis ───────────────────────────────────────────────
    if analyse_btn and text_input.strip():
        with st.spinner("Analysing…"):
            time.sleep(0.3)  # small UX delay for feel
            result = sm.analyse(text_input.strip())
            st.session_state.last_result = result
            st.session_state.history.insert(0, {
                "text": text_input.strip(),
                "result": result,
            })
            if len(st.session_state.history) > 10:
                st.session_state.history.pop()

    if st.session_state.last_result:
        render_result(st.session_state.last_result)
    else:
        result_placeholder.markdown("""
<div style="text-align:center;padding:3rem 1rem;color:#444466">
    <div style="font-size:3rem;margin-bottom:12px">🧠</div>
    <div style="font-size:15px;font-weight:500;color:#6666aa">Ready to analyse</div>
    <div style="font-size:13px;margin-top:6px">Type text on the left or pick a quick example →</div>
</div>
""", unsafe_allow_html=True)

# ── Batch results ────────────────────────────────────────────────────────────
if batch_btn and batch_input.strip():
    lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]
    if lines:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-title">📊 BATCH RESULTS — {len(lines)} texts analysed</div>',
            unsafe_allow_html=True
        )

        summary = {"positive": 0, "negative": 0, "neutral": 0}
        rows = []
        with st.spinner(f"Analysing {len(lines)} texts…"):
            for line in lines:
                r = sm.analyse(line)
                summary[r["sentiment"]] += 1
                rows.append(r | {"text": line})

        # Summary bar
        total = len(rows)
        s1, s2, s3 = st.columns(3)
        with s1:
            pct = round(summary["positive"]/total*100)
            st.markdown(f'<div class="mini-card" style="border-color:rgba(34,212,126,0.3)">'
                        f'<div class="mini-label">Positive</div>'
                        f'<div class="mini-val" style="color:#22d47e">{summary["positive"]}</div>'
                        f'<div class="mini-sub">{pct}% of total</div></div>',
                        unsafe_allow_html=True)
        with s2:
            pct = round(summary["negative"]/total*100)
            st.markdown(f'<div class="mini-card" style="border-color:rgba(245,90,90,0.3)">'
                        f'<div class="mini-label">Negative</div>'
                        f'<div class="mini-val" style="color:#f55a5a">{summary["negative"]}</div>'
                        f'<div class="mini-sub">{pct}% of total</div></div>',
                        unsafe_allow_html=True)
        with s3:
            pct = round(summary["neutral"]/total*100)
            st.markdown(f'<div class="mini-card" style="border-color:rgba(108,143,255,0.3)">'
                        f'<div class="mini-label">Neutral</div>'
                        f'<div class="mini-val" style="color:#6c8fff">{summary["neutral"]}</div>'
                        f'<div class="mini-sub">{pct}% of total</div></div>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Per-row results
        for i, row in enumerate(rows, 1):
            lbl = row["sentiment"]
            cls = {"positive":"pos","negative":"neg","neutral":"neu"}[lbl]
            badge_html = (f'<span class="hist-badge-{cls}">{lbl}</span>')
            st.markdown(f"""
<div class="hist-row">
    <div class="hist-emoji">{row['emoji']}</div>
    <div class="hist-body">
        <div class="hist-text">{i}. {row['text']}</div>
        <div class="hist-meta">Confidence: {row['confidence']}% · Intensity: {row['intensity']} · Tokens: {row['token_count']}</div>
    </div>
    {badge_html}
</div>""", unsafe_allow_html=True)

# ── History ──────────────────────────────────────────────────────────────────
if show_history and st.session_state.history:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.markdown('<div class="card-title">🕒  RECENT ANALYSES</div>',
                    unsafe_allow_html=True)
    with hcol2:
        if st.button("Clear", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

    for i, entry in enumerate(st.session_state.history[:8]):
        res  = entry["result"]
        lbl  = res["sentiment"]
        cls  = {"positive":"pos","negative":"neg","neutral":"neu"}[lbl]
        txt  = entry["text"]
        disp = txt[:80] + "…" if len(txt) > 80 else txt
        badge = f'<span class="hist-badge-{cls}">{lbl}</span>'
        if st.button(f"↩ Load", key=f"hist_{i}", help="Click to load this text"):
            st.session_state.input_text = entry["text"]
            st.session_state.last_result = res
            st.rerun()
        st.markdown(f"""
<div class="hist-row" style="margin-top:-8px">
    <div class="hist-emoji">{res['emoji']}</div>
    <div class="hist-body">
        <div class="hist-text">{disp}</div>
        <div class="hist-meta">{res['confidence']}% confidence · {res['token_count']} tokens</div>
    </div>
    {badge}
</div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;color:#333355;font-size:12px">
    Deep NLP Sentiment Analysis · Pure NumPy · 3-Layer MLP · 400 training samples<br>
    <span style="color:#555577">sentiment_model_v2.py</span> · Run with <code style="color:#a78bfa">streamlit run app.py</code>
</div>
""", unsafe_allow_html=True)
