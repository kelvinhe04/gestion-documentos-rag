"""Página del chatbot RAG: pregunta y respuesta sobre los documentos indexados."""
from __future__ import annotations

import streamlit as st

import app_helpers as ah
from src import config
from src.rag import rag_pipeline

ah.page_config("Chatbot RAG", "💬")
ah.sidebar_status()

st.title("💬 Chatbot RAG")
st.caption("Pregunta en lenguaje natural; respondo usando solo el contenido de los documentos indexados.")

# Parámetro de recuperación.
with st.sidebar:
    st.markdown("### 🎚️ Parámetros RAG")
    top_k = st.slider("Fragmentos a recuperar (k)", 1, 10, config.TOP_K)

# Estado de la conversación.
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if ah.data_version() == 0:
    st.warning("No hay documentos indexados. Ve a **📥 Ingesta de Documentos** primero.", icon="⚠️")

# Historial.
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 Fuentes ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"], start=1):
                    titulo = s["metadata"].get("title", "Documento")
                    st.markdown(f"**{i}. {titulo}** · relevancia `{s['score']}`")
                    st.caption(s["text"][:350] + ("…" if len(s["text"]) > 350 else ""))

# Entrada del usuario.
if pregunta := st.chat_input("Escribe tu pregunta..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en los documentos y generando respuesta..."):
            resultado = rag_pipeline.answer(pregunta, k=top_k)
        st.markdown(resultado["answer"])
        provider = resultado.get("provider", "?")
        st.caption(f"🧠 Generado con: **{provider}**")
        if resultado.get("sources"):
            with st.expander(f"📎 Fuentes ({len(resultado['sources'])})"):
                for i, s in enumerate(resultado["sources"], start=1):
                    titulo = s["metadata"].get("title", "Documento")
                    st.markdown(f"**{i}. {titulo}** · relevancia `{s['score']}`")
                    st.caption(s["text"][:350] + ("…" if len(s["text"]) > 350 else ""))

    st.session_state.mensajes.append(
        {"role": "assistant", "content": resultado["answer"], "sources": resultado.get("sources", [])}
    )

if st.session_state.mensajes and st.sidebar.button("🗑️ Limpiar conversación"):
    st.session_state.mensajes = []
    st.rerun()
