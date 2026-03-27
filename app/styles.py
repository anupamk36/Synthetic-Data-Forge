"""Custom CSS theme for ForgeFlow AI — modern, demo-ready UI."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ===== Global ===== */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0A2E 0%, #1A1145 40%, #251B5A 100%);
}
section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}
section[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
}
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 0.65rem 1rem !important;
    border-radius: 8px !important;
    transition: background 0.15s ease !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.08) !important;
}
section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {
    background: rgba(99,102,241,0.35) !important;
}
section[data-testid="stSidebar"] .stDivider {
    border-color: rgba(255,255,255,0.1) !important;
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
    color: rgba(255,255,255,0.45) !important;
}

/* ===== Page Background ===== */
.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1200px;
}

/* ===== Hero Banner ===== */
.hero-banner {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    color: white;
    margin-bottom: 1.75rem;
    box-shadow: 0 4px 24px rgba(102,126,234,0.25);
}
.hero-banner h2 {
    margin: 0 0 0.35rem;
    font-size: 1.55rem;
    font-weight: 700;
    color: white !important;
}
.hero-banner p {
    margin: 0;
    opacity: 0.92;
    font-size: 0.95rem;
    font-weight: 400;
    color: white !important;
}

/* ===== Metric Row ===== */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-tile {
    flex: 1;
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-tile .val {
    font-size: 1.65rem;
    font-weight: 700;
    color: #1F2937;
    line-height: 1.2;
}
.metric-tile .lbl {
    font-size: 0.8rem;
    color: #6B7280;
    font-weight: 500;
    margin-top: 0.2rem;
}

/* ===== Status Pill ===== */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.pill-green { background: #D1FAE5; color: #065F46; }
.pill-red   { background: #FEE2E2; color: #991B1B; }
.pill-blue  { background: #DBEAFE; color: #1E40AF; }
.pill-amber { background: #FEF3C7; color: #92400E; }

/* ===== Primary Button ===== */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    box-shadow: 0 3px 10px rgba(102,126,234,0.3) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.35) !important;
}

/* ===== Secondary / Stop Button ===== */
div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="stBaseButton-secondary"] {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.12s ease !important;
}

/* ===== Progress Bar ===== */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667EEA 0%, #764BA2 100%) !important;
    border-radius: 8px !important;
}
.stProgress > div > div > div {
    border-radius: 8px !important;
}

/* ===== File Uploader ===== */
[data-testid="stFileUploader"] section {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 2rem !important;
    background: #F9FAFB !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #667EEA !important;
    background: #EEF2FF !important;
}

/* ===== Container Borders (Card Style) ===== */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] > div.stMarkdown) {
    border-radius: 12px !important;
    border-color: #E5E7EB !important;
}

/* ===== Expanders ===== */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ===== Dataframe ===== */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ===== Tab Bar ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
}

/* ===== Alerts ===== */
.stAlert {
    border-radius: 10px !important;
}

/* ===== Download Button ===== */
.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
}

/* ===== Section Header Helper ===== */
.section-hdr {
    font-size: 1.05rem;
    font-weight: 600;
    color: #374151;
    margin: 1.25rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ===== Empty State ===== */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 0.75rem; }
.empty-state .msg  { font-size: 1.05rem; font-weight: 500; color: #6B7280; }
.empty-state .hint { font-size: 0.87rem; color: #9CA3AF; margin-top: 0.35rem; }

/* ===== Scrollbar ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ===== Divider Subtle ===== */
hr {
    border-color: #E5E7EB !important;
    margin: 1.25rem 0 !important;
}

/* ===== Number input / selectbox rounding ===== */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 10px !important;
}
</style>
"""


def inject_css():
    """Inject the custom CSS into the Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
