# CLAUDE.md

Guía para Claude Code (claude.ai/code) al trabajar en este repositorio.

## Resumen del proyecto

**Sistema de Gestión de Documentos Académicos con RAG** — Proyecto Integrador
(Segundo Parcial), UTP, **Grupo 7**. Python + ChromaDB + Streamlit. Carga PDFs,
los indexa con embeddings y responde preguntas vía RAG.

> Antes de cambios de arquitectura, lee `PLAN.md` (plan y mapeo de requisitos) y
> `AGENT.md` (mapa del código y convenciones).

## Comandos frecuentes

```powershell
# Entorno
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pipeline por CLI
python scripts/run_pipeline.py --pdf-dir data/raw
python scripts/run_pipeline.py --pdf ruta/archivo.pdf --reset

# Dashboard
streamlit run streamlit_app.py

# Verificar que todo compila
python -m compileall src scripts streamlit_app.py app_helpers.py
```

## Arquitectura (flujo de datos)

```
PDFs (usuario) ─▶ pipeline.py ─▶ preprocessing ─▶ indexing(ChromaDB)
                                                         │
                                       ┌─────────────────┼────────────────┐
                              búsqueda semántica     RAG (llm)       ML clustering
                                       └─────────────────┴────────────────┘
                                                   Streamlit (pages/)
```

## Reglas para modificar el código

- **Idioma español** en UI, comentarios y docstrings.
- **Imports absolutos** `from src.x import y`. No usar imports relativos frágiles.
- Mantener la **técnica de ML** (clustering) ya que es requisito evaluado. No eliminarla.
- ChromaDB **no acepta** metadatos `None` ni listas: pasar siempre por
  `_sanitize_metadata`.
- El chatbot debe degradar con elegancia: si no hay LLM, **respuesta extractiva**.
  No introducir rutas que lancen excepción al usuario por falta de API key.
- No subir `.env`, `data/chroma/` ni PDFs al control de versiones (ver `.gitignore`).
- Cachés de Streamlit: tras ingestar datos, llamar `st.cache_data.clear()` para
  refrescar estadísticas y clustering.

## Dónde tocar para tareas comunes

- **Soporte de otro tipo de fuente:** `src/ingestion/` + método en `src/pipeline.py`.
- **Cambiar el modelo de embeddings:** `src/indexing/vector_store.py` (`_embedding_function`).
- **Otro modelo/Proveedor de LLM:** `src/rag/llm.py` + variables en `.env`.
- **Otra técnica de ML:** `src/ml/` y nueva sección en `pages/4_📊_Dashboard.py`.
- **Nueva página del dashboard:** crear archivo en `pages/` (numerado) e importar
  `app_helpers`.
