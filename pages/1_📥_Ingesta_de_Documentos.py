"""Página de ingesta: carga PDFs locales subidos por el usuario."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import app_helpers as ah
from src import config
from src import pipeline

ah.page_config("Ingesta", "📥")
ah.sidebar_status()

st.title("📥 Ingesta de Documentos")
st.caption("Sube PDFs académicos para indexarlos en ChromaDB.")

st.subheader("Subir documentos PDF")
st.write("Sube uno o varios PDFs académicos. Se procesarán e indexarán en ChromaDB.")

uploaded = st.file_uploader(
    "Selecciona archivos PDF", type=["pdf"], accept_multiple_files=True
)

if uploaded and st.button("⚙️ Procesar e indexar", type="primary", key="btn_pdf"):
    progress = st.progress(0.0)
    resultados = []
    for i, file in enumerate(uploaded, start=1):
        dest = config.RAW_DIR / file.name
        dest.write_bytes(file.getbuffer())
        with st.spinner(f"Procesando {file.name}..."):
            try:
                res = pipeline.ingest_pdf_file(dest)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "title": file.name, "reason": str(exc), "chunks": 0}
        resultados.append(res)
        progress.progress(i / len(uploaded))

    st.cache_data.clear()

    ok = [r for r in resultados if r.get("ok")]
    st.success(f"✅ {len(ok)}/{len(resultados)} documentos indexados.")
    for r in resultados:
        if r.get("ok"):
            st.write(f"- **{r['title']}** — {r['chunks']} fragmentos, {r.get('num_pages', '?')} págs.")
        else:
            st.write(f"- ❌ **{r.get('title')}** — {r.get('reason')}")

# También permite indexar PDFs ya presentes en data/raw.
st.divider()
pdfs_en_disco = sorted(Path(config.RAW_DIR).glob("*.pdf"))
st.caption(f"Hay {len(pdfs_en_disco)} PDF(s) en `data/raw/`.")
if pdfs_en_disco and st.button("📂 Indexar todos los PDFs de data/raw", key="btn_dir"):
    with st.spinner("Indexando carpeta data/raw..."):
        resultados = pipeline.ingest_pdf_directory(config.RAW_DIR)
    st.cache_data.clear()
    ok = [r for r in resultados if r.get("ok")]
    st.success(f"✅ {len(ok)}/{len(resultados)} documentos indexados desde disco.")

st.divider()
stats = ah.cached_statistics(ah.data_version())
st.metric("Total de documentos indexados", stats["n_documents"])
