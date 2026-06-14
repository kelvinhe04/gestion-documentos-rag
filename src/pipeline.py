"""Orquestador del pipeline de datos extremo a extremo.

Une todas las etapas: ingesta -> preprocesamiento (limpieza + chunking) ->
indexación (embeddings en ChromaDB). Es el punto de entrada que usan tanto
el dashboard de Streamlit como el script de línea de comandos.

Pipeline:
    PDF  ->  extract_pdf  ->  clean_text  ->  chunk_text  ->  add_document
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from src import config
from src.indexing import vector_store
from src.ingestion import pdf_loader
from src.preprocessing.text_processing import chunk_text, clean_text


def _make_doc_id(title: str, filename: str) -> str:
    """Genera un identificador estable y único para un documento."""
    base = f"{title}|{filename}".encode("utf-8", errors="ignore")
    return hashlib.md5(base).hexdigest()[:12]


def ingest_pdf_file(path: str | Path, extra_metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Procesa e indexa un único archivo PDF.

    Devuelve un resumen con el estado del procesamiento.
    """
    path = Path(path)
    data = pdf_loader.extract_pdf(path)

    text = clean_text(data["text"])
    if not text:
        return {
            "ok": False,
            "title": data["title"],
            "filename": data["filename"],
            "reason": "No se pudo extraer texto (¿PDF escaneado sin OCR?).",
            "chunks": 0,
        }

    chunks = chunk_text(text)
    if not chunks:
        return {
            "ok": False,
            "title": data["title"],
            "filename": data["filename"],
            "reason": "El texto extraído es demasiado corto para generar fragmentos.",
            "chunks": 0,
        }
    doc_id = _make_doc_id(data["title"], data["filename"])

    metadata = {
        "title": data["title"],
        "author": data["author"],
        "num_pages": data["num_pages"],
        "filename": data["filename"],
        "source": data["source"],
        "char_count": len(text),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    n = vector_store.add_document(doc_id, chunks, metadata)
    return {
        "ok": True,
        "doc_id": doc_id,
        "title": data["title"],
        "filename": data["filename"],
        "source": metadata["source"],
        "num_pages": data["num_pages"],
        "chunks": n,
    }


def ingest_pdf_directory(directory: str | Path) -> List[Dict[str, Any]]:
    """Procesa todos los PDFs de una carpeta."""
    directory = Path(directory)
    results = []
    for pdf in sorted(directory.glob("*.pdf")):
        try:
            results.append(ingest_pdf_file(pdf))
        except Exception as exc:  # noqa: BLE001
            results.append({"ok": False, "filename": pdf.name, "reason": str(exc), "chunks": 0})
    return results


def get_statistics() -> Dict[str, Any]:
    """Calcula estadísticas agregadas de los documentos para el dashboard."""
    data = vector_store.get_all(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    if not metadatas:
        return {
            "n_documents": 0,
            "n_chunks": 0,
            "n_pages": 0,
            "by_source": {},
            "documents": [],
        }

    docs: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        meta = meta or {}
        doc_id = meta.get("doc_id", "desconocido")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "title": meta.get("title", doc_id),
                "source": meta.get("source", "local"),
                "num_pages": meta.get("num_pages", 0),
                "author": meta.get("author", ""),
                "published": meta.get("published", ""),
                "filename": meta.get("filename", ""),
                "chunks": 0,
            }
        docs[doc_id]["chunks"] += 1

    documents = list(docs.values())
    by_source = Counter(d["source"] for d in documents)

    return {
        "n_documents": len(documents),
        "n_chunks": len(metadatas),
        "n_pages": sum(int(d.get("num_pages") or 0) for d in documents),
        "by_source": dict(by_source),
        "documents": documents,
    }
