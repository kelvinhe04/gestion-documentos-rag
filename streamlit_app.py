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

pg = st.navigation(
    [
        st.Page("pages/inicio.py", title="Inicio", icon=":material/home:"),
        st.Page(
            "pages/ingesta.py",
            title="Ingesta de Documentos",
            icon=":material/upload_file:",
        ),
        st.Page("pages/chatbot.py", title="Chatbot RAG", icon=":material/chat:"),
        st.Page(
            "pages/busqueda.py",
            title="Búsqueda Semántica",
            icon=":material/search:",
        ),
        st.Page("pages/dashboard.py", title="Dashboard", icon=":material/bar_chart:"),
    ]
)
pg.run()
