"""Página de ingesta: carga PDFs y gestión de documentos indexados."""
from __future__ import annotations

import streamlit as st

import app_helpers as ah
from src import chat_sessions, config, pipeline
from src.indexing import vector_store

ah.sidebar_status()

st.title("Ingesta de Documentos")
st.caption("Sube PDFs académicos para indexarlos en ChromaDB.")


# ── Diálogos de confirmación ───────────────────────────────────────────────
@st.dialog("Eliminar documento")
def _dialog_eliminar_uno(doc: dict) -> None:
    st.markdown(
        f"¿Seguro que quieres eliminar **{doc['title']}**?  \n"
        f"Se borrarán **{doc['chunk_count']} fragmentos** del índice y el archivo del disco."
    )
    st.caption("Esta acción no se puede deshacer.")
    st.write("")
    ca, cb = st.columns(2)
    if ca.button("Eliminar", type="primary", use_container_width=True):
        n = vector_store.delete_document(doc["doc_id"])
        if doc["filename"]:
            pdf_path = config.RAW_DIR / doc["filename"]
            if pdf_path.exists():
                pdf_path.unlink()
        chat_sessions.remove_doc_from_all_sessions(doc["doc_id"])
        st.cache_data.clear()
        st.session_state.pop("_del_doc", None)
        st.session_state["_del_toast"] = f"Eliminado: **{doc['title']}** ({n} fragmentos)."
        st.rerun()
    if cb.button("Cancelar", use_container_width=True):
        st.session_state.pop("_del_doc", None)
        st.rerun()


@st.dialog("Eliminar documentos seleccionados")
def _dialog_eliminar_bulk(ids: list[str], docs_map: dict) -> None:
    n = len(ids)
    st.markdown(f"¿Eliminar los siguientes **{n} documento(s)**?")
    for doc_id in ids:
        doc = docs_map.get(doc_id)
        if doc:
            frags = doc["chunk_count"]
            st.markdown(f"- {doc['title']} *({frags} frags.)*")
    st.caption("Esta acción no se puede deshacer.")
    st.write("")
    ba, bb = st.columns(2)
    if ba.button(f"Eliminar {n}", type="primary", use_container_width=True):
        total_frags = 0
        for doc_id in ids:
            total_frags += vector_store.delete_document(doc_id)
            doc = docs_map.get(doc_id)
            if doc and doc["filename"]:
                pdf_path = config.RAW_DIR / doc["filename"]
                if pdf_path.exists():
                    pdf_path.unlink()
            chat_sessions.remove_doc_from_all_sessions(doc_id)
        st.cache_data.clear()
        st.session_state.pop("_del_bulk", None)
        st.session_state["_del_toast"] = (
            f"Eliminados {n} documentos ({total_frags} fragmentos)."
        )
        st.rerun()
    if bb.button("Cancelar", use_container_width=True):
        st.session_state.pop("_del_bulk", None)
        st.rerun()


# ── Subir documentos ───────────────────────────────────────────────────────
st.subheader("Subir documentos PDF")

# La key cambia para resetear el widget tras indexar
uploader_key = f"uploader_{st.session_state.get('_uploader_gen', 0)}"
uploaded = st.file_uploader(
    "Selecciona archivos PDF", type=["pdf"],
    accept_multiple_files=True, key=uploader_key,
)

# Detectar cambio en selección de archivos para limpiar resultados previos
uploaded_names = sorted(f.name for f in uploaded) if uploaded else []
if st.session_state.get("_ingesta_uploaded") != uploaded_names:
    st.session_state.pop("ingesta_resultados", None)
    st.session_state["_ingesta_uploaded"] = uploaded_names

if "ingesta_resultados" in st.session_state:
    resultados = st.session_state["ingesta_resultados"]
    ok  = [r for r in resultados if r.get("ok")]
    err = [r for r in resultados if not r.get("ok")]

    if ok:
        st.success(f"✅ {len(ok)}/{len(resultados)} documento(s) indexado(s).")
        for r in ok:
            st.write(f"- **{r['title']}** — {r['chunks']} fragmentos, {r.get('num_pages', '?')} págs.")

    if err:
        st.error(f"❌ {len(err)} documento(s) no se pudieron indexar:")
        for r in err:
            st.write(f"- **{r.get('title', r.get('filename', '?'))}** — {r.get('reason', 'error desconocido')}")
        st.caption(
            "Los PDFs escaneados (sin capa de texto) no se pueden indexar sin OCR. "
            "Los PDFs con texto muy corto tampoco generan fragmentos."
        )

    if st.button("Cerrar y subir más", key="btn_cerrar_resultados"):
        st.session_state.pop("ingesta_resultados", None)
        st.session_state.pop("_ingesta_uploaded", None)
        st.session_state["_uploader_gen"] = st.session_state.get("_uploader_gen", 0) + 1
        st.rerun()
elif uploaded and st.button("Procesar e indexar", type="primary", key="btn_pdf"):
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
    st.session_state["ingesta_resultados"] = resultados
    st.rerun()

# ── Documentos indexados ───────────────────────────────────────────────────
st.divider()
st.subheader("Documentos indexados")

# Toast de confirmación tras eliminar
if toast_msg := st.session_state.pop("_del_toast", None):
    st.toast(toast_msg, icon=":material/check_circle:")

docs = vector_store.list_documents()

if not docs:
    stats = ah.cached_statistics(ah.data_version())
    st.metric("Total de documentos indexados", stats["n_documents"])
    st.info("No hay documentos indexados aún.")
else:
    st.caption(f"{len(docs)} documento(s) en la colección.")

    # ── Cabecera con selector global ──────────────────────────────────────
    h0, h1, h2, h3, h4, h5 = st.columns([0.5, 3.5, 1, 1, 2, 1])
    _prev_sel_todos = st.session_state.get("_prev_sel_todos", False)
    seleccionar_todos = h0.checkbox("", key="sel_todos", label_visibility="collapsed")
    h1.markdown("**Título**")
    h2.markdown("**Págs.**")
    h3.markdown("**Frags.**")
    h4.markdown("**Indexado**")
    h5.markdown("**Acción**")
    st.divider()

    seleccionados: list[str] = []

    for doc in docs:
        doc_id = doc["doc_id"]
        if seleccionar_todos:
            st.session_state[f"chk_{doc_id}"] = True
        elif _prev_sel_todos and not seleccionar_todos:
            # acaba de deseleccionarse → limpiar checkboxes individuales
            st.session_state[f"chk_{doc_id}"] = False

        c0, c1, c2, c3, c4, c5 = st.columns([0.5, 3.5, 1, 1, 2, 1])
        checked = c0.checkbox("", key=f"chk_{doc_id}", label_visibility="collapsed")
        if checked:
            seleccionados.append(doc_id)
        c1.write(doc["title"])
        c2.caption(str(doc["num_pages"]))
        c3.caption(str(doc["chunk_count"]))
        c4.caption(doc["ingested_at"][:10] if doc["ingested_at"] else "—")

        if c5.button("Eliminar", key=f"del_{doc_id}", type="secondary"):
            st.session_state["_del_doc"] = doc

    st.session_state["_prev_sel_todos"] = seleccionar_todos

    # ── Barra de acciones masivas ──────────────────────────────────────────
    if seleccionados:
        st.divider()
        n_sel = len(seleccionados)
        col_info, col_btn = st.columns([3, 1])
        col_info.markdown(f"**{n_sel}** documento(s) seleccionado(s)")
        if col_btn.button(
            f"Eliminar {n_sel} seleccionado(s)",
            type="primary",
            key="btn_eliminar_sel",
        ):
            st.session_state["_del_bulk"] = seleccionados

    # ── Abrir modales si corresponde ──────────────────────────────────────
    if "_del_doc" in st.session_state:
        _dialog_eliminar_uno(st.session_state["_del_doc"])

    if "_del_bulk" in st.session_state:
        docs_map = {d["doc_id"]: d for d in docs}
        _dialog_eliminar_bulk(st.session_state["_del_bulk"], docs_map)
