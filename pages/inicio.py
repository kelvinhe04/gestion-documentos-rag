"""Página de inicio: bienvenida, métricas rápidas y guía de uso."""
from __future__ import annotations

import streamlit as st

import app_helpers as ah
from src import config

ah.sidebar_status()

st.title("Sistema de Gestión de Documentos Académicos con RAG")
st.markdown(
    "**Grupo 7 — Proyecto Integrador.** Carga documentos académicos (PDFs), "
    "indéxalos con embeddings en **ChromaDB** y haz preguntas sobre su contenido "
    "mediante **RAG** (Retrieval-Augmented Generation)."
)

version = ah.data_version()
stats = ah.cached_statistics(version)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documentos", stats["n_documents"])
c2.metric("Fragmentos", stats["n_chunks"])
c3.metric("Páginas", stats["n_pages"])
c4.metric("Fuentes", len(stats["by_source"]) if stats["by_source"] else 0)

st.divider()

col_izq, col_der = st.columns([1.3, 1])

with col_izq:
    st.subheader("¿Cómo usar el sistema?")
    st.markdown(
        """
        1. **Ingesta de Documentos** — Sube tus propios PDFs. El pipeline
           limpia el texto, lo divide en fragmentos y los indexa con embeddings.
           También puedes eliminar documentos ya indexados.
        2. **Chatbot RAG** — Crea chats independientes, cada uno con su propio
           contexto de documentos. Haz preguntas en lenguaje natural.
        3. **Búsqueda Semántica** — Busca por significado (no por palabras
           exactas) dentro de todos los documentos.
        4. **Dashboard** — Estadísticas de los documentos y **clustering
           (KMeans)** que los agrupa por temática.
        """
    )

with col_der:
    st.subheader("Arquitectura")
    st.markdown(
        """
        - **Pipeline de datos:** ingesta de PDFs subidos por el usuario
        - **Preprocesamiento:** limpieza + *chunking* con solapamiento
        - **Indexación:** ChromaDB + embeddings *all-MiniLM-L6-v2*
        - **ML:** clustering KMeans (silueta + TF-IDF + PCA)
        - **RAG:** búsqueda semántica + LLM open source (Llama 3)
        - **Dashboard:** Streamlit + Plotly
        """
    )
    if config.LLM_PROVIDER == "groq" and not config.GROQ_API_KEY:
        st.info(
            "No hay `GROQ_API_KEY` configurada: el chatbot funcionará en "
            "**modo extractivo**. Añádela en `.env` para respuestas generativas.",
            icon=":material/info:",
        )

if stats["n_documents"] == 0:
    st.warning(
        "Todavía no hay documentos indexados. Ve a **Ingesta de Documentos** "
        "y sube tus PDFs para empezar.",
        icon=":material/warning:",
    )
