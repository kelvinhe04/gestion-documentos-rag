"""Página de inicio: bienvenida, métricas rápidas y guía de uso."""
from __future__ import annotations

import streamlit as st

import app_helpers as ah
from src import config

ah.sidebar_status()

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #060d1a 0%, #0c1e36 50%, #0a1f1c 100%);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 18px;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 48px rgba(0,0,0,0.6), 0 0 60px rgba(14,165,233,0.06);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            font-size: 1.85rem;
            font-weight: 800;
            background: linear-gradient(120deg, #38bdf8, #22d3ee 50%, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.04em;
            margin-bottom: 0.55rem;
            line-height: 1.15;
        ">Sistema de Gestión de Documentos Académicos</div>
        <p style="color: rgba(226,232,240,0.75); margin: 0; font-size: 0.97rem; line-height: 1.65; font-weight: 400;">
            Carga documentos académicos (PDFs), indéxalos con embeddings en
            <strong style="color: #34d399;">ChromaDB</strong> y haz preguntas sobre su contenido
            mediante <strong style="color: #34d399;">RAG</strong> (Retrieval-Augmented Generation).
        </p>
        <div style="margin-top:1.1rem;display:flex;gap:0.65rem;flex-wrap:wrap;">
            <span style="background:rgba(56,189,248,0.1);color:#7dd3fc;font-size:0.76rem;
                         font-weight:600;padding:0.28rem 0.75rem;border-radius:20px;
                         border:1px solid rgba(56,189,248,0.2);">ChromaDB</span>
            <span style="background:rgba(56,189,248,0.1);color:#7dd3fc;font-size:0.76rem;
                         font-weight:600;padding:0.28rem 0.75rem;border-radius:20px;
                         border:1px solid rgba(56,189,248,0.2);">Sentence Transformers</span>
            <span style="background:rgba(52,211,153,0.1);color:#6ee7b7;font-size:0.76rem;
                         font-weight:600;padding:0.28rem 0.75rem;border-radius:20px;
                         border:1px solid rgba(52,211,153,0.2);">KMeans ML</span>
            <span style="background:rgba(52,211,153,0.1);color:#6ee7b7;font-size:0.76rem;
                         font-weight:600;padding:0.28rem 0.75rem;border-radius:20px;
                         border:1px solid rgba(52,211,153,0.2);">Llama 3 via Groq</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

version = ah.data_version()
stats = ah.cached_statistics(version)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documentos", stats["n_documents"])
c2.metric("Fragmentos", stats["n_chunks"])
c3.metric("Páginas", stats["n_pages"])
c4.metric("Fuentes", len(stats["by_source"]) if stats["by_source"] else 0)

st.divider()

col_izq = st.container()

with col_izq:
    st.subheader("¿Cómo usar el sistema?")
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:0.75rem;margin-top:0.75rem;">

          <div style="display:flex;gap:1rem;align-items:flex-start;
                      background:rgba(56,189,248,0.05);
                      border:1px solid rgba(56,189,248,0.15);
                      border-radius:12px;padding:1rem 1.15rem;">
            <div style="background:linear-gradient(135deg,#0ea5e9,#0284c7);color:#fff;
                        font-weight:800;font-size:0.9rem;min-width:2rem;height:2rem;
                        border-radius:7px;display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;box-shadow:0 4px 12px rgba(14,165,233,0.35);">1</div>
            <div>
              <div style="color:#7dd3fc;font-weight:700;font-size:0.94rem;margin-bottom:0.18rem;">
                Ingesta de Documentos
              </div>
              <div style="color:#94a3b8;font-size:0.85rem;line-height:1.5;">
                Sube tus PDFs. El pipeline limpia el texto, lo divide en fragmentos
                y los indexa con embeddings. También puedes eliminar documentos ya indexados.
              </div>
            </div>
          </div>

          <div style="display:flex;gap:1rem;align-items:flex-start;
                      background:rgba(56,189,248,0.05);
                      border:1px solid rgba(56,189,248,0.15);
                      border-radius:12px;padding:1rem 1.15rem;">
            <div style="background:linear-gradient(135deg,#0ea5e9,#0284c7);color:#fff;
                        font-weight:800;font-size:0.9rem;min-width:2rem;height:2rem;
                        border-radius:7px;display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;box-shadow:0 4px 12px rgba(14,165,233,0.35);">2</div>
            <div>
              <div style="color:#7dd3fc;font-weight:700;font-size:0.94rem;margin-bottom:0.18rem;">
                Chatbot RAG
              </div>
              <div style="color:#94a3b8;font-size:0.85rem;line-height:1.5;">
                Crea chats independientes, cada uno con su propio contexto de documentos.
                Haz preguntas en lenguaje natural.
              </div>
            </div>
          </div>

          <div style="display:flex;gap:1rem;align-items:flex-start;
                      background:rgba(56,189,248,0.05);
                      border:1px solid rgba(56,189,248,0.15);
                      border-radius:12px;padding:1rem 1.15rem;">
            <div style="background:linear-gradient(135deg,#0ea5e9,#0284c7);color:#fff;
                        font-weight:800;font-size:0.9rem;min-width:2rem;height:2rem;
                        border-radius:7px;display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;box-shadow:0 4px 12px rgba(14,165,233,0.35);">3</div>
            <div>
              <div style="color:#7dd3fc;font-weight:700;font-size:0.94rem;margin-bottom:0.18rem;">
                Búsqueda Semántica
              </div>
              <div style="color:#94a3b8;font-size:0.85rem;line-height:1.5;">
                Busca por significado (no por palabras exactas) dentro de todos los documentos.
                Usa embeddings y distancia coseno.
              </div>
            </div>
          </div>

          <div style="display:flex;gap:1rem;align-items:flex-start;
                      background:rgba(56,189,248,0.05);
                      border:1px solid rgba(56,189,248,0.15);
                      border-radius:12px;padding:1rem 1.15rem;">
            <div style="background:linear-gradient(135deg,#0ea5e9,#0284c7);color:#fff;
                        font-weight:800;font-size:0.9rem;min-width:2rem;height:2rem;
                        border-radius:7px;display:flex;align-items:center;justify-content:center;
                        flex-shrink:0;box-shadow:0 4px 12px rgba(14,165,233,0.35);">4</div>
            <div>
              <div style="color:#7dd3fc;font-weight:700;font-size:0.94rem;margin-bottom:0.18rem;">
                Dashboard
              </div>
              <div style="color:#94a3b8;font-size:0.85rem;line-height:1.5;">
                Estadísticas de los documentos y <strong style="color:#22d3ee;">clustering KMeans</strong>
                que los agrupa por temática con visualización PCA 2D.
              </div>
            </div>
          </div>

        </div>
        """,
        unsafe_allow_html=True,
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
