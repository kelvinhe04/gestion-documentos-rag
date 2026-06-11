"""Utilidades compartidas por las páginas de Streamlit.

Centraliza el acceso cacheado al pipeline, las estadísticas y el clustering
para que las distintas páginas del dashboard reutilicen los mismos recursos.
"""
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from src import config
from src.indexing import vector_store
from src.ml import clustering
from src.pipeline import get_statistics


def page_config(title: str, icon: str = "📚") -> None:
    """Configuración común de página."""
    st.set_page_config(page_title=f"{title} · RAG Académico", page_icon=icon, layout="wide")


def sidebar_status() -> None:
    """Muestra el estado del sistema en la barra lateral (todas las páginas)."""
    with st.sidebar:
        st.markdown("### ⚙️ Estado del sistema")
        st.caption(f"**LLM:** {config.llm_status()}")
        try:
            n_chunks = vector_store.count_chunks()
        except Exception:  # noqa: BLE001
            n_chunks = 0
        st.caption(f"**Fragmentos indexados:** {n_chunks}")
        st.caption(f"**Colección:** `{config.COLLECTION_NAME}`")
        st.divider()
        st.caption("Grupo 7 · Gestión de Documentos Académicos con RAG")


@st.cache_data(show_spinner=False)
def cached_statistics(version: int) -> Dict[str, Any]:
    """Estadísticas de los documentos. `version` invalida la caché al cambiar."""
    return get_statistics()


@st.cache_data(show_spinner="Ejecutando clustering (KMeans)...")
def cached_clustering(version: int, n_clusters: int | None) -> Dict[str, Any]:
    """Resultado del clustering de documentos (cacheado por versión + k)."""
    return clustering.cluster_documents(n_clusters=n_clusters)


def data_version() -> int:
    """Versión del estado de datos (nº de fragmentos) para invalidar cachés."""
    try:
        return vector_store.count_chunks()
    except Exception:  # noqa: BLE001
        return 0
