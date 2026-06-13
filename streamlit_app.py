"""Punto de entrada principal — define la navegación con iconos Material.

Ejecutar con:  streamlit run streamlit_app.py
"""
from __future__ import annotations

from PIL import Image

import streamlit as st

st.set_page_config(
    page_title="RAG Académico",
    page_icon=Image.open("assets/favicon.png"),
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

    /* ── Títulos ─────────────────────────────────────────────────────────── */
    [data-testid="stMain"] h1 {
        background: linear-gradient(120deg, #38bdf8 0%, #22d3ee 55%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        letter-spacing: -0.04em;
        padding-bottom: 4px;
    }
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 { color: #7dd3fc; font-weight: 700; }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid rgba(56,189,248,0.12) !important;
    }

    /* ── Métrica cards ───────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.18);
        border-radius: 14px;
        padding: 1.2rem 1.4rem !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.35);
        transition: all 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(56,189,248,0.35);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 20px rgba(56,189,248,0.08);
        transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #e2e8f0 !important;
    }

    /* ── Botones ─────────────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.18s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(14,165,233,0.3) !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        box-shadow: 0 6px 22px rgba(14,165,233,0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Expanders ───────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border: 1px solid rgba(56,189,248,0.18) !important;
        border-radius: 12px !important;
        background: rgba(56,189,248,0.04) !important;
    }

    /* ── Dataframes ──────────────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid rgba(56,189,248,0.15) !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.4) !important;
    }

    /* ── File uploader ───────────────────────────────────────────────────── */
    [data-testid="stFileDropzone"] {
        border: 2px dashed rgba(56,189,248,0.22) !important;
        border-radius: 12px !important;
        background: rgba(56,189,248,0.04) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileDropzone"]:hover {
        border-color: #38bdf8 !important;
        background: rgba(56,189,248,0.08) !important;
    }

    /* ── Progress bar ────────────────────────────────────────────────────── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #0ea5e9, #22d3ee) !important;
        border-radius: 10px !important;
    }

    /* ── Divider ─────────────────────────────────────────────────────────── */
    [data-testid="stMarkdownContainer"] hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.25), transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Chat input ──────────────────────────────────────────────────────── */
    [data-testid="stChatInput"] textarea {
        border-radius: 16px !important;
    }

    /* ── Toast ───────────────────────────────────────────────────────────── */
    [data-testid="stToast"] {
        border-radius: 12px !important;
        border-left: 4px solid #38bdf8 !important;
    }

    /* ── Modales / Dialogs ───────────────────────────────────────────────── */
    @keyframes modal-in {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0);   }
    }
    /* centrado vertical del dialog dentro del scroll-container */
    [data-baseweb="modal"] > div:last-child {
        display: flex !important;
        align-items: center !important;
        min-height: 100% !important;
    }
    [data-baseweb="dialog"] {
        animation: modal-in 0.18s ease-out both;
    }
    /* botón primario dentro de cualquier dialog → azul del tema */
    [data-testid="stDialog"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(14,165,233,0.3) !important;
    }
    [data-testid="stDialog"] .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        box-shadow: 0 6px 20px rgba(14,165,233,0.45) !important;
    }

    /* ── Multiselect tags (chips) ─────────────────────────────────────────── */
    [data-baseweb="tag"] {
        background: #082f49 !important;
        border: 1px solid #0ea5e9 !important;
        border-radius: 6px !important;
    }
    [data-baseweb="tag"] span {
        color: #7dd3fc !important;
        font-weight: 500 !important;
    }
    /* × button inside the tag */
    [data-baseweb="tag"] [role="presentation"] span {
        color: #38bdf8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pg = st.navigation(
    [
        st.Page("pages/inicio.py",   title="Inicio",               icon=":material/terminal:"),
        st.Page("pages/ingesta.py",  title="Ingesta de Documentos", icon=":material/cloud_upload:"),
        st.Page("pages/chatbot.py",  title="Chatbot RAG",           icon=":material/smart_toy:"),
        st.Page("pages/busqueda.py", title="Búsqueda Semántica",    icon=":material/manage_search:"),
        st.Page("pages/dashboard.py",title="Dashboard",             icon=":material/monitoring:"),
    ]
)
pg.run()
