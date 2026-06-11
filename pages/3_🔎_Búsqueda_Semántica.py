"""Página de búsqueda semántica: recupera fragmentos por significado."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import app_helpers as ah
from src.indexing import vector_store

ah.page_config("Búsqueda Semántica", "🔎")
ah.sidebar_status()

st.title("🔎 Búsqueda Semántica")
st.caption(
    "Encuentra fragmentos por **significado**, no por coincidencia exacta de "
    "palabras. Usa los embeddings y la distancia coseno de ChromaDB."
)

col1, col2 = st.columns([4, 1])
consulta = col1.text_input("¿Qué quieres buscar?", placeholder="ej. métodos de evaluación de modelos")
k = col2.number_input("Resultados", min_value=1, max_value=20, value=5)

if ah.data_version() == 0:
    st.warning("No hay documentos indexados. Ve a **📥 Ingesta de Documentos** primero.", icon="⚠️")

if consulta:
    with st.spinner("Buscando..."):
        hits = vector_store.search(consulta, k=int(k))

    if not hits:
        st.info("Sin resultados relevantes.")
    else:
        st.success(f"{len(hits)} resultado(s) encontrados.")

        # Tabla resumen.
        tabla = pd.DataFrame(
            [
                {
                    "Relevancia": h["score"],
                    "Documento": h["metadata"].get("title", "?"),
                    "Fuente": h["metadata"].get("source", "?"),
                    "Fragmento #": h["metadata"].get("chunk_index", "?"),
                }
                for h in hits
            ]
        )
        st.dataframe(tabla, width="stretch", hide_index=True)

        st.divider()
        for i, h in enumerate(hits, start=1):
            titulo = h["metadata"].get("title", "Documento")
            with st.expander(f"#{i} · {titulo} · relevancia {h['score']}"):
                st.write(h["text"])
                meta = h["metadata"]
                st.caption(
                    f"Fuente: {meta.get('source', '?')} · "
                    f"Páginas: {meta.get('num_pages', '?')} · "
                    f"Fragmento: {meta.get('chunk_index', '?')}"
                )
