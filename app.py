import time
import streamlit as st
from rag_engine import load_base, ask_rag


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Top 10 Robbery Cases",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# 2. CUSTOM CSS
st.markdown("""
<style>
    :root {
        --primary: #6C5CE7;
        --primary-dark: #4B3FCC;
        --accent: #00B894;
        --bg-soft: #F5F3FF;
        --text-main: #2D3436;
        --text-muted: #636E72;
        --border-soft: #E8E4F8;
        --card-bg: #FFFFFF;
    }

    .stApp {
        background: linear-gradient(180deg, #FAF9FF 0%, #F5F3FF 100%);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- WELCOME PAGE ---------- */
    .welcome-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding-top: 6vh;
        padding-bottom: 2rem;
    }
    .welcome-icon {
        font-size: 80px;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .welcome-title {
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #6C5CE7, #00B894);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .welcome-subtitle {
        font-size: 18px;
        color: var(--text-muted);
        max-width: 620px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    .welcome-badge {
        display: inline-block;
        background: #EDE9FE;
        color: var(--primary);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .feature-row {
        display: flex;
        gap: 16px;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        border: 1px solid var(--border-soft);
        border-radius: 14px;
        padding: 18px 22px;
        width: 200px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(108, 92, 231, 0.06);
    }
    .feature-card .icon { font-size: 28px; }
    .feature-card .label {
        font-size: 14px;
        font-weight: 600;
        color: var(--text-main);
        margin-top: 8px;
    }
    .feature-card .desc {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* ---------- HEADER (main app) ---------- */
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #6C5CE7, #00B894);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: var(--text-muted);
        font-size: 16px;
        margin-bottom: 1.5rem;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F3FF 100%);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] h1 {
        color: var(--primary);
        font-size: 22px;
    }

    /* ---------- CHAT BUBBLES ---------- */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #EDE9FE;
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #FFFFFF;
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        box-shadow: 0 2px 8px rgba(108, 92, 231, 0.08);
    }

    /* ---------- CHAT INPUT ---------- */
    div[data-testid="stChatInput"] {
        border-radius: 14px;
        border: 2px solid var(--border-soft);
        background: white;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(108, 92, 231, 0.15);
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #6C5CE7, #00B894);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2.2rem;
        font-weight: 700;
        font-size: 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(108, 92, 231, 0.35);
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 6px 14px;
        background: #F5F3FF;
    }
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }

    /* ---------- FLOWCHART ---------- */
    .flow-step {
        background: white;
        border: 2px solid var(--border-soft);
        border-radius: 10px;
        padding: 10px 12px;
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-main);
        transition: all 0.2s ease;
    }
    .flow-step:hover {
        border-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.15);
    }
    .flow-arrow {
        text-align: center;
        color: var(--primary);
        font-size: 20px;
        line-height: 1;
        margin: 2px 0;
    }
    .team-card {
        background: white;
        border: 1px solid var(--border-soft);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .team-name {
        font-weight: 600;
        color: var(--text-main);
        font-size: 14px;
    }
    .team-role {
        color: var(--text-muted);
        font-size: 12px;
    }

    /* ---------- LOADING ---------- */
    .loading-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-top: 20vh;
        text-align: center;
    }
    .loading-title {
        font-size: 24px;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    .loading-sub {
        font-size: 14px;
        color: var(--text-muted);
    }

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #F5F3FF; }
    ::-webkit-scrollbar-thumb {
        background: #C7BFEF;
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
</style>
""", unsafe_allow_html=True)


# 3. STAGE STATE
if "stage" not in st.session_state:
    st.session_state.stage = "welcome"   # welcome → loading → app


# =============================================================
# STAGE 1 — WELCOME PAGE
# =============================================================
if st.session_state.stage == "welcome":

    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🔎</div>
        <div class="welcome-badge">RAG-POWERED RESEARCH ASSISTANT</div>
        <div class="welcome-title">Top 10 Robbery Cases</div>
        <div class="welcome-subtitle">
            Ask questions about ten of the world's most significant
            robberies — values, methods, investigations, and outcomes —
            answered only from the verified case documents.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown("""
    <div class="feature-row">
        <div class="feature-card">
            <div class="icon">📚</div>
            <div class="label">10 Case Studies</div>
            <div class="desc">Verified robbery documents</div>
        </div>
        <div class="feature-card">
            <div class="icon">🧠</div>
            <div class="label">Grounded Answers</div>
            <div class="desc">No outside knowledge used</div>
        </div>
        <div class="feature-card">
            <div class="icon">⚡</div>
            <div class="label">Fast Retrieval</div>
            <div class="desc">FAISS + Groq LLM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Start button (centered)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Start", use_container_width=True):
            st.session_state.stage = "loading"
            st.rerun()

    st.markdown(
        "<div style='text-align:center; color:#9b9b9b; "
        "font-size:12px; margin-top:2rem;'>"
        "Group-05 · Gnanesh N · Akshaya K N · Bhavyashree · "
        "Hanamanthraya D G · Ashok · Deepti B D</div>",
        unsafe_allow_html=True,
    )

    st.stop()


# =============================================================
# STAGE 2 — LOADING SCREEN
# =============================================================
if st.session_state.stage == "loading":

    st.markdown("""
    <div class="loading-wrap">
        <div class="loading-title">⚙️ Preparing your assistant…</div>
        <div class="loading-sub">Loading case documents and building the search index</div>
    </div>
    """, unsafe_allow_html=True)

    progress = st.progress(0)
    status = st.empty()

    steps = [
        ("📂 Loading PDF documents…", 20),
        ("📄 Extracting text…", 40),
        ("✂️ Creating chunks…", 60),
        ("🔢 Generating embeddings…", 80),
        ("🗄 Building FAISS index…", 100),
    ]

    for msg, pct in steps:
        status.info(msg)
        time.sleep(0.4)
        progress.progress(pct)

    st.session_state.stage = "app"
    st.rerun()


# =============================================================
# STAGE 3 — MAIN APP
# =============================================================
# Re-expand sidebar for main app
st.markdown(
    "<style>section[data-testid='stSidebar'] "
    "{display:block !important;}</style>",
    unsafe_allow_html=True,
)


# inside `with st.sidebar:` — replace the tabs line with:
info, cases, pipeline, team = st.tabs(
    ["📘 Info", "📚 Cases", "⚙️ Pipeline", "👥 Team"]
)

# then add this block after the `with info:` block:
with cases:
    st.markdown("### 📚 Indexed Cases")
    st.caption("These are the ten robberies in the knowledge base.")

    case_list = [
        ("CF-01", "Central Bank of Iraq", "2003", "Iraq", "disputed"),
        ("CF-02", "Isabella Stewart Gardner Museum", "1990", "USA", "unsolved"),
        ("CF-03", "City Bonds Robbery", "1990", "UK", "solved"),
        ("CF-04", "Dar Es Salaam Bank", "2007", "Iraq", "unsolved"),
        ("CF-05", "Dresden Green Vault", "2019", "Germany", "mostly solved"),
        ("CF-06", "Carlton Cannes Hotel", "2013", "France", "unresolved"),
        ("CF-07", "Antwerp Diamond Center", "2003", "Belgium", "solved"),
        ("CF-08", "Louvre Crown Jewels", "2025", "France", "ongoing"),
        ("CF-09", "Harry Winston Jewelry Store", "2008", "France", "mostly solved"),
        ("CF-10", "Schiphol Airport", "2005", "Netherlands", "partially solved"),
    ]

    status_colors = {
        "solved": ("#00B894", "#E8F8F5"),
        "mostly solved": ("#00B894", "#E8F8F5"),
        "partially solved": ("#F39C12", "#FEF5E7"),
        "unsolved": ("#E17055", "#FDEDEC"),
        "unresolved": ("#E17055", "#FDEDEC"),
        "disputed": ("#F39C12", "#FEF5E7"),
        "ongoing": ("#6C5CE7", "#EDE9FE"),
    }

    for code, name, year, country, status in case_list:
        fg, bg = status_colors.get(status, ("#636E72", "#F5F3FF"))
        st.markdown(
            f'<div class="team-card">'
            f'<div class="team-name">{code} · {name}</div>'
            f'<div class="team-role">{year} · {country} · '
            f'<span style="background:{bg}; color:{fg}; '
            f'padding:2px 8px; border-radius:8px; font-weight:600;">'
            f'{status}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ---------- MAIN HEADER ----------
st.markdown('<div class="main-title">Top 10 Robbery Cases</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">RAG-Based Research Assistant</div>',
            unsafe_allow_html=True)
st.caption(
    "🔎 Ask about the ten major robberies — values, methods, "
    "investigations, and outcomes — answered only from the case documents."
)
st.write("")


# ---------- GROQ KEY ----------
try:
    mykey = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    mykey = None

if not mykey:
    st.warning(
        "⚠️ Groq API key not configured. "
        "Add `GROQ_API_KEY` to Streamlit Secrets."
    )


# ---------- RAG ENGINE ----------
@st.cache_resource
def initiate_rag():
    return load_base("datafiles")


try:
    index, embed_model, chunks = initiate_rag()
except Exception as error:
    st.error(f"Unable to load RAG engine: {error}")
    st.stop()


# ---------- CHAT ----------
if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])

question = st.chat_input("💬 Ask anything about the top 10 robbery cases...")

if question and question.strip():
    st.session_state.chat_hist.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching robbery case documents..."):
            try:
                if not mykey:
                    raise ValueError(
                        "Groq API key is missing. "
                        "Configure GROQ_API_KEY in Streamlit Secrets."
                    )

                answer = ask_rag(
                    question, embed_model, chunks, index,
                    api_key=mykey, num_chunks=5,
                )
                st.write(answer)

            except Exception as error:
                answer = f"Error: {error}"
                st.error(answer)

            st.session_state.chat_hist.append(
                {"role": "assistant", "content": answer}
            )

elif question is not None:
    st.warning("Please enter a question.")


# ---------- CLEAR / BACK ----------
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.chat_hist = []
        st.toast("Chat history cleared ✨")
        st.rerun()

with st.sidebar:
    st.divider()
    if st.button("⬅ Back to Welcome", use_container_width=True):
        st.session_state.stage = "welcome"
        st.session_state.chat_hist = []
        st.rerun()
