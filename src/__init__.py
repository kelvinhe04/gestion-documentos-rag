"""Paquete principal del Sistema de Gestión de Documentos Académicos con RAG.

Estructura:
    config            -> Configuración global (variables de entorno).
    ingestion/        -> Ingesta de PDFs subidos por el usuario.
    preprocessing/    -> Limpieza y segmentación (chunking) de texto.
    indexing/         -> Vector store con ChromaDB y búsqueda semántica.
    ml/               -> Clustering (KMeans) de documentos por temática.
    rag/              -> Proveedor de LLM y pipeline RAG (recuperar + generar).
    pipeline          -> Orquestador del pipeline de datos extremo a extremo.
"""

__version__ = "1.0.0"
