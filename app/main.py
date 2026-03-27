"""ForgeFlow AI -- Synthetic Data Platform.

Slim entry-point: sets up logging, injects CSS, renders sidebar navigation,
and routes to the selected page module.
"""

import logging
import os
import sys

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from core.logging_config import setup_logging
setup_logging()

from app.styles import inject_css
from app.ui_single import render_single_table_page
from app.ui_relational import render_relational_page
from app.ui_time_travel import render_time_travel_page
from app.ui_privacy import render_privacy_page
from app.ui_quality import render_quality_page
from app.ui_schema_library import render_schema_library_page
from app.ui_history import render_history_page
from core.llm_logic import LLMLogicEngine

logger = logging.getLogger(__name__)

# Page Config
st.set_page_config(
    page_title="ForgeFlow AI",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# Sidebar
PAGES = {
    "\U0001f4ca  Single Table": render_single_table_page,
    "\U0001f517  Multi-Table (Hydra)": render_relational_page,
    "\u23f0  Time Travel": render_time_travel_page,
    "\U0001f6e1\ufe0f  Privacy Audit": render_privacy_page,
    "\U0001f4c8  Data Quality": render_quality_page,
    "\U0001f4da  Schema Library": render_schema_library_page,
    "\U0001f4dc  History": render_history_page,
}

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 0.6rem">
            <span style="font-size:2rem">\u26a1</span><br>
            <span style="font-size:1.15rem;font-weight:700;letter-spacing:0.3px">ForgeFlow AI</span><br>
            <span style="font-size:0.78rem;opacity:0.6">Synthetic Data Platform</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()

    # System Status
    st.caption("SYSTEM STATUS")
    llm = LLMLogicEngine()
    if llm.is_available():
        models = llm.get_available_models()
        st.markdown(
            '<span class="status-pill pill-green">\U0001f7e2 Ollama Online</span>',
            unsafe_allow_html=True,
        )
        if models:
            st.caption(f"Models: {', '.join(models[:3])}")
    else:
        st.markdown(
            '<span class="status-pill pill-red">\U0001f534 Ollama Offline</span>',
            unsafe_allow_html=True,
        )

    # Generation Stats
    gen_count = 0
    for key in ("generated_df", "multi_results", "tt_generated_df"):
        df = st.session_state.get(key)
        if df is not None:
            gen_count += len(df) if hasattr(df, "__len__") else 0
    if gen_count:
        st.caption(f"Records generated this session: **{gen_count:,}**")

    st.divider()
    st.caption("v2.0.0  \u00b7  Built with Streamlit")

# Page Router
PAGES[page]()
