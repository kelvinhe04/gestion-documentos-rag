"""Chatbot RAG con múltiples sesiones, cada una con su contexto de PDFs."""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

import app_helpers as ah
from src import chat_sessions, config
from src.rag import rag_pipeline


def _chat_scroll_ui(auto_scroll: bool = False) -> None:
    """Inyecta el botón flotante ↓ y (opcionalmente) auto-scroll al fondo."""
    components.html(
        f"""
        <script>
        (function() {{
            const doc   = window.parent.document;
            const AUTO  = {'true' if auto_scroll else 'false'};
            const BTN_ID = 'rag-scroll-btn';

            // ── Detectar contenedor de scroll ────────────────────────────
            function getMain() {{
                return doc.querySelector('[data-testid="stMain"]')
                    || doc.querySelector('.main')
                    || doc.documentElement;
            }}

            // ── Auto-scroll con delay para esperar render ─────────────────
            if (AUTO) {{
                setTimeout(() => {{
                    const m = getMain();
                    if (m) m.scrollTo({{ top: m.scrollHeight, behavior: 'smooth' }});
                }}, 120);
            }}

            // ── Botón flotante ────────────────────────────────────────────
            let btn = doc.getElementById(BTN_ID);
            if (!btn) {{
                btn = doc.createElement('button');
                btn.id = BTN_ID;
                btn.title = 'Ir al mensaje más reciente';
                btn.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                         stroke="currentColor" stroke-width="2.5"
                         stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>`;
                btn.style.cssText = `
                    position: fixed;
                    bottom: 76px;
                    left: 50%;
                    transform: translateX(-50%) scale(1);
                    width: 36px; height: 36px;
                    border-radius: 50%;
                    border: 1px solid rgba(255,255,255,0.18);
                    background: rgba(35,35,35,0.92);
                    color: rgba(255,255,255,0.9);
                    cursor: pointer;
                    z-index: 9998;
                    display: none;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 3px 14px rgba(0,0,0,0.55);
                    backdrop-filter: blur(8px);
                    transition: transform 0.15s ease, background 0.15s ease;
                `;
                btn.onmouseenter = () => {{
                    btn.style.transform = 'translateX(-50%) scale(1.12)';
                    btn.style.background = 'rgba(60,60,60,0.95)';
                }};
                btn.onmouseleave = () => {{
                    btn.style.transform = 'translateX(-50%) scale(1)';
                    btn.style.background = 'rgba(35,35,35,0.92)';
                }};
                btn.onclick = () => {{
                    const m = getMain();
                    if (m) m.scrollTo({{ top: m.scrollHeight, behavior: 'smooth' }});
                }};
                doc.body.appendChild(btn);
            }}

            // ── Mostrar/ocultar según posición de scroll ──────────────────
            function updateBtn() {{
                const m = getMain();
                if (!m) return;
                const dist = m.scrollHeight - m.scrollTop - m.clientHeight;
                if (dist > 160) {{
                    btn.style.display = 'flex';
                }} else {{
                    btn.style.display = 'none';
                }}
            }}

            const m = getMain();
            if (m) {{
                if (m._ragScrollHandler) m.removeEventListener('scroll', m._ragScrollHandler);
                m._ragScrollHandler = updateBtn;
                m.addEventListener('scroll', updateBtn);
                updateBtn();
            }}
        }})();
        </script>
        """,
        height=0,
    )

st.markdown(
    """
    <style>
    /* ── Burbujas de chat ─────────────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(56,189,248,0.1);
        border-radius: 14px !important;
        padding: 0.85rem 1rem !important;
        margin-bottom: 0.35rem;
        background: rgba(22,27,39,0.6);
        backdrop-filter: blur(6px);
        transition: border-color 0.2s ease;
    }
    [data-testid="stChatMessage"]:hover {
        border-color: rgba(56,189,248,0.22);
    }
    /* Avatar icon color → electric blue */
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] {
        background: rgba(14,165,233,0.12) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        border-radius: 9px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

ah.sidebar_status()


@st.dialog("Eliminar chat")
def _dialog_eliminar_chat(session_id: str, session_name: str) -> None:
    st.markdown(
        f"¿Seguro que quieres eliminar **{session_name}**?  \n"
        "Se perderá todo el historial de mensajes de este chat."
    )
    st.caption("Esta acción no se puede deshacer.")
    st.write("")
    ca, cb = st.columns(2)
    if ca.button("Eliminar", type="primary", use_container_width=True):
        chat_sessions.delete_session(session_id)
        remaining = chat_sessions.list_sessions()
        st.session_state.current_session = remaining[0]["id"]
        st.session_state.pop("_del_chat", None)
        st.session_state.pop("_loaded_session", None)
        st.rerun()
    if cb.button("Cancelar", use_container_width=True):
        st.session_state.pop("_del_chat", None)
        st.rerun()


# ── Cargar y validar sesiones ───────────────────────────────────────────────
sessions = chat_sessions.list_sessions()
if not sessions:
    chat_sessions.create_session("Chat 1")
    sessions = chat_sessions.list_sessions()

session_ids = [s["id"] for s in sessions]
session_name_map = {s["id"]: s["name"] for s in sessions}

# Asegurar que current_session apunta a una sesión válida
if (
    "current_session" not in st.session_state
    or st.session_state.current_session not in session_ids
):
    st.session_state.current_session = session_ids[0]

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Mis chats")

    # Sin key: el selectbox se controla por index= en cada render.
    # Nunca modificamos session_state del widget después de instanciarlo.
    current_idx = session_ids.index(st.session_state.current_session)
    selected_id = st.selectbox(
        "Sesión activa",
        options=session_ids,
        format_func=lambda x: session_name_map.get(x, x),
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state.current_session = selected_id
    sid = selected_id

    col_n, col_d = st.columns(2)

    if col_n.button("+ Nuevo", use_container_width=True, key="btn_new"):
        new_id = chat_sessions.create_session()
        st.session_state.current_session = new_id
        st.rerun()

    if col_d.button(
        "Eliminar",
        use_container_width=True,
        key="btn_del",
        disabled=len(sessions) <= 1,
    ):
        st.session_state["_del_chat"] = (sid, session_name_map.get(sid, sid))

    st.divider()

    # ── Renombrar sesión ────────────────────────────────────────────────────
    st.markdown("**Renombrar chat**")
    session_data = chat_sessions.get_session(sid) or {}

    # Inicializar el input con el nombre actual cuando cambia la sesión
    rename_key = f"_rnm_{sid}"
    if rename_key not in st.session_state:
        st.session_state[rename_key] = session_data.get("name", "")

    new_name = st.text_input(
        "Nombre",
        key=rename_key,
        label_visibility="collapsed",
        placeholder="Escribe el nombre del chat...",
    )
    if st.button("Guardar nombre", use_container_width=True, key=f"save_name_{sid}"):
        if new_name.strip():
            chat_sessions.rename_session(sid, new_name.strip())
            st.rerun()

    st.divider()

    # ── Documentos del chat ─────────────────────────────────────────────────
    st.markdown("### Documentos del chat")
    docs = ah.list_documents_cached(ah.data_version())
    active_docs: list[str] = []

    if not docs:
        st.caption("No hay documentos indexados.")
    else:
        doc_map = {d["doc_id"]: d["title"] for d in docs}
        saved_docs = [d for d in session_data.get("doc_ids", []) if d in doc_map]

        new_selection = st.multiselect(
            "Filtrar documentos",
            options=list(doc_map.keys()),
            default=saved_docs,
            format_func=lambda x: doc_map.get(x, x),
            help="Selecciona los documentos que usará este chat.",
            label_visibility="collapsed",
        )
        if set(new_selection) != set(saved_docs):
            chat_sessions.set_session_docs(sid, new_selection)
        active_docs = new_selection

    st.divider()

    # ── Parámetros RAG ──────────────────────────────────────────────────────
    st.markdown("### Parámetros RAG")
    top_k = st.slider("Fragmentos a recuperar (k)", 1, 20, config.TOP_K)

    if session_data.get("messages") and st.button(
        "Limpiar conversación", use_container_width=True
    ):
        chat_sessions.clear_messages(sid)
        st.session_state.pop("_loaded_session", None)
        st.rerun()

# ── Modal de confirmación de eliminación ────────────────────────────────────
if "_del_chat" in st.session_state:
    _del_id, _del_name = st.session_state["_del_chat"]
    _dialog_eliminar_chat(_del_id, _del_name)

# ── Sincronizar historial desde JSON al cambiar de sesión ───────────────────
if st.session_state.get("_loaded_session") != sid:
    fresh = chat_sessions.get_session(sid) or {}
    st.session_state.mensajes = list(fresh.get("messages", []))
    st.session_state._loaded_session = sid

# ── Área principal ──────────────────────────────────────────────────────────
st.title("Chatbot RAG")
st.caption(f"**{session_name_map.get(sid, sid)}**")

if ah.data_version() == 0:
    st.warning(
        "No hay documentos indexados. Ve a **Ingesta de Documentos** primero.",
        icon=":material/warning:",
    )
    st.stop()

# Bloquear chat si no hay documentos seleccionados
if not active_docs:
    st.warning(
        "Selecciona al menos un documento en **Documentos del chat** (panel izquierdo) "
        "para poder hacer preguntas.",
        icon=":material/folder_open:",
    )
    st.stop()

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:0.7rem;
                background:rgba(56,189,248,0.07);
                border:1px solid rgba(56,189,248,0.2);
                border-radius:10px;padding:0.65rem 1rem;margin-bottom:0.4rem;">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
             stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
        </svg>
        <span style="color:#7dd3fc;font-weight:500;font-size:0.9rem;">
            Contexto:&nbsp;<strong style="color:#38bdf8;">{len(active_docs)}</strong>
            &nbsp;documento(s) seleccionado(s).
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Historial de mensajes
for msg in st.session_state.get("mensajes", []):
    _av = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(msg["role"], avatar=_av):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Fuentes ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"], start=1):
                    titulo = s["metadata"].get("title", "Documento")
                    st.markdown(f"**{i}. {titulo}** · relevancia `{s['score']}`")
                    st.caption(s["text"][:350] + ("…" if len(s["text"]) > 350 else ""))

# Botón flotante (siempre presente cuando hay historial)
_needs_scroll = st.session_state.pop("_scroll_to_bottom", False)
_chat_scroll_ui(auto_scroll=_needs_scroll)

# Entrada del usuario
if pregunta := st.chat_input("Escribe tu pregunta..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta, "sources": []})
    chat_sessions.add_message(sid, "user", pregunta)

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(pregunta)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.spinner("Buscando en los documentos y generando respuesta..."):
            resultado = rag_pipeline.answer(
                pregunta,
                k=top_k,
                doc_ids=active_docs,
            )
        st.markdown(resultado["answer"])
        st.caption(f"Generado con: **{resultado.get('provider', '?')}**")
        if resultado.get("sources"):
            with st.expander(f"Fuentes ({len(resultado['sources'])})"):
                for i, s in enumerate(resultado["sources"], start=1):
                    titulo = s["metadata"].get("title", "Documento")
                    st.markdown(f"**{i}. {titulo}** · relevancia `{s['score']}`")
                    st.caption(s["text"][:350] + ("…" if len(s["text"]) > 350 else ""))

    asst_msg = {
        "role": "assistant",
        "content": resultado["answer"],
        "sources": resultado.get("sources", []),
    }
    st.session_state.mensajes.append(asst_msg)
    chat_sessions.add_message(
        sid, "assistant", resultado["answer"], resultado.get("sources", [])
    )
    # Marcar scroll en el próximo render (después del rerun implícito de Streamlit)
    st.session_state["_scroll_to_bottom"] = True
    st.rerun()
