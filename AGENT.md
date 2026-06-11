# AGENT.md — Contexto para asistentes de IA

Este archivo da contexto a cualquier asistente de IA (Claude, Cursor, Copilot,
etc.) que trabaje en este repositorio. Para guía específica de Claude Code, ver
`CLAUDE.md`.

## Qué es este proyecto

Sistema **RAG (Retrieval-Augmented Generation)** para gestión de documentos
académicos. Carga PDFs, los indexa con embeddings en **ChromaDB** y responde
preguntas sobre su contenido. Proyecto universitario (UTP, Grupo 7) en Python con
dashboard en Streamlit.

## Cómo ejecutar

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
cp .env.example .env          # opcional: añadir GROQ_API_KEY
streamlit run streamlit_app.py
```

## Mapa del código

| Carpeta / archivo | Responsabilidad |
|---|---|
| `src/config.py` | Configuración global vía variables de entorno. |
| `src/ingestion/pdf_loader.py` | Extrae texto de PDFs (pypdf). |
| `src/preprocessing/text_processing.py` | Limpieza y *chunking* del texto. |
| `src/indexing/vector_store.py` | ChromaDB: indexar y buscar (semántica). |
| `src/ml/clustering.py` | ML: clustering KMeans de documentos. |
| `src/rag/llm.py` | Abstracción del LLM (Groq/Ollama/extractivo). |
| `src/rag/rag_pipeline.py` | Orquesta recuperación + generación (RAG). |
| `src/pipeline.py` | Orquestador ingesta→preprocesado→indexación + stats. |
| `streamlit_app.py`, `pages/`, `app_helpers.py` | Dashboard Streamlit. |
| `scripts/run_pipeline.py` | CLI del pipeline. |

## Convenciones

- **Idioma:** comentarios, docstrings y UI en **español**.
- **Imports:** absolutos desde la raíz (`from src.x import y`). Los scripts
  insertan la raíz en `sys.path`.
- **Estilo:** funciones pequeñas con docstring; type hints; sin dependencias
  pesadas innecesarias (se evita PyTorch usando los embeddings ONNX de Chroma).
- **Robustez:** el sistema debe funcionar SIEMPRE; si falla el LLM externo, se
  usa la respuesta extractiva. No romper esta garantía.
- **Metadatos de Chroma:** solo str/int/float/bool (sin None ni listas); usar
  `_sanitize_metadata` en `vector_store.py`.

## Invariantes que no hay que romper

1. Debe mantenerse **al menos una técnica de ML** (clustering).
2. El dashboard debe funcionar aunque la colección esté vacía (mensajes guía).
3. La configuración sensible (API keys) vive en `.env`, nunca en el código.
