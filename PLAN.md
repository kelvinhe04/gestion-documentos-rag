# PLAN.md — Plan del Proyecto

**Proyecto:** Sistema de Gestión de Documentos Académicos con RAG
**Curso:** Tópicos Especiales I — Segundo Parcial · UTP
**Grupo:** 7
**Tema asignado:** Grupo 7 — *Sistema de Gestión de Documentos Académicos con RAG*

---

## 1. Problemática

Los estudiantes e investigadores manejan grandes volúmenes de documentos
académicos (papers, tesis, apuntes en PDF). Encontrar información específica
dentro de decenas de PDFs es lento: la búsqueda tradicional por palabra clave
no entiende el *significado* de la consulta. Se necesita un sistema que indexe
los documentos y permita **preguntar en lenguaje natural** y obtener respuestas
fundamentadas en el contenido real de los documentos.

## 2. Objetivo

Construir un sistema que cargue PDFs académicos, los indexe con embeddings en
una base vectorial y responda preguntas sobre su contenido mediante RAG, con un
dashboard interactivo y análisis de los documentos mediante Machine Learning.

## 3. Mapeo de requisitos → implementación

| Requisito (enunciado) | Entregable | Estado |
|---|---|---|
| Pipeline de datos (ingesta de PDFs) | `src/ingestion/pdf_loader.py`, `src/pipeline.py` | ✅ |
| Preprocesar y transformar | `src/preprocessing/text_processing.py` | ✅ |
| ≥ 1 técnica de ML | `src/ml/clustering.py` (KMeans) | ✅ |
| Dashboard Streamlit | `streamlit_app.py` + `pages/` | ✅ |
| Carga/procesamiento de PDFs | `src/ingestion/pdf_loader.py` | ✅ |
| Indexación con ChromaDB | `src/indexing/vector_store.py` | ✅ |
| Búsqueda semántica | `pages/3_🔎_Búsqueda_Semántica.py` | ✅ |
| Chatbot RAG funcional | `src/rag/rag_pipeline.py` + `pages/2_💬_Chatbot_RAG.py` | ✅ |
| Dashboard con estadísticas | `pages/4_📊_Dashboard.py` | ✅ |
| README + repo GitHub | `README.md` | ✅ |

## 4. Arquitectura (fases del pipeline)

1. **Ingesta** — PDFs subidos por el usuario (`pdf_loader`).
2. **Preprocesamiento** — limpieza (`clean_text`) y segmentación (`chunk_text`)
   con solapamiento.
3. **Indexación** — embeddings *all-MiniLM-L6-v2* (ONNX) en ChromaDB persistente.
4. **Recuperación** — búsqueda semántica por similitud coseno (`vector_store.search`).
5. **Generación (RAG)** — contexto + LLM open source (Llama 3 vía Groq), con
   respaldo extractivo.
6. **ML** — clustering KMeans de documentos por temática (silueta + TF-IDF + PCA).
7. **Visualización** — dashboard Streamlit con Plotly.

## 5. Decisiones técnicas

- **Embeddings con ChromaDB DefaultEmbeddingFunction (ONNX):** evita depender de
  PyTorch; instalación más ligera y sin GPU.
- **LLM configurable (Groq / Ollama / extractivo):** el sistema funciona aunque
  no haya API key, garantizando una demo siempre operativa.
- **Clustering a nivel de documento:** se promedian los embeddings de los
  fragmentos de cada documento para agrupar documentos (no fragmentos).
- **Persistencia local:** ChromaDB en `data/chroma/` para no reindexar en cada
  ejecución.

## 6. Reparto de roles (sugerido)

- Pipeline e ingesta de datos.
- Indexación y búsqueda semántica (ChromaDB).
- RAG y chatbot (LLM).
- Machine Learning (clustering) y dashboard.
- Documentación e integración.

## 7. Posibles mejoras futuras

- Re-ranking de resultados (cross-encoder).
- Soporte de OCR para PDFs escaneados.
- Memoria conversacional en el chatbot (historial como contexto).
- Métricas de evaluación del RAG (faithfulness, relevancia).
