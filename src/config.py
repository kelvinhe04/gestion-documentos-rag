"""Configuración global del proyecto.

Lee las variables de entorno desde un archivo .env (si existe) y expone
constantes usadas por el resto del sistema. Centralizar la configuración
aquí facilita cambiar de proveedor de LLM o ajustar el pipeline sin tocar
el código.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Carga las variables definidas en .env (no falla si el archivo no existe).
load_dotenv()

# ----------------------------------------------------------------------
# Rutas del proyecto
# ----------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"            # PDFs descargados / subidos
PROCESSED_DIR: Path = DATA_DIR / "processed"  # Metadatos de procesamiento
CHROMA_DIR: Path = DATA_DIR / "chroma"        # Persistencia de ChromaDB

# Asegura que las carpetas existan al importar el módulo.
for _d in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------
# Vector store (ChromaDB)
# ----------------------------------------------------------------------
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documentos_academicos")
# Modelo de embeddings (el DefaultEmbeddingFunction de Chroma usa all-MiniLM-L6-v2).
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ----------------------------------------------------------------------
# Segmentación de texto (chunking)
# ----------------------------------------------------------------------
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 900)        # caracteres por fragmento
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 150)  # solapamiento entre fragmentos

# ----------------------------------------------------------------------
# Recuperación (retrieval)
# ----------------------------------------------------------------------
TOP_K: int = _get_int("TOP_K", 5)          # nº de fragmentos a recuperar por consulta
MAX_CHUNKS_PER_DOC: int = _get_int("MAX_CHUNKS_PER_DOC", 2)  # máx chunks de un mismo doc en el ranking
MIN_SCORE: float = _get_float("MIN_SCORE", 0.10)              # score mínimo para incluir un fragmento

# ----------------------------------------------------------------------
# LLM (chatbot RAG)
# ----------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()  # groq | ollama | none

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.2)


def llm_status() -> str:
    """Devuelve una descripción legible del proveedor de LLM activo."""
    if LLM_PROVIDER == "groq":
        return f"Groq ({GROQ_MODEL})" if GROQ_API_KEY else "Groq (falta GROQ_API_KEY -> modo extractivo)"
    if LLM_PROVIDER == "ollama":
        return f"Ollama ({OLLAMA_MODEL} @ {OLLAMA_HOST})"
    return "Modo extractivo (sin LLM generativo)"
