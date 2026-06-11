"""Fuente de datos 1: carga y procesamiento de PDFs locales.

Extrae texto y metadatos de archivos PDF usando pypdf. Es la fuente principal
del sistema: documentos académicos que el usuario sube al sistema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pypdf import PdfReader


def extract_pdf(path: str | Path) -> dict:
    """Extrae texto y metadatos de un PDF.

    Devuelve un diccionario con:
        title, author, num_pages, text, filename, source
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el PDF: {path}")

    reader = PdfReader(str(path))

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - una página corrupta no debe abortar todo
            pages_text.append("")
    text = "\n\n".join(pages_text)

    meta = reader.metadata
    title = _clean_meta(getattr(meta, "title", None)) or path.stem
    author = _clean_meta(getattr(meta, "author", None))

    return {
        "title": title,
        "author": author,
        "num_pages": len(reader.pages),
        "text": text,
        "filename": path.name,
        "source": "local",
    }


def _clean_meta(value: Optional[str]) -> Optional[str]:
    """Normaliza un valor de metadato del PDF (puede venir vacío o con espacios)."""
    if not value:
        return None
    value = str(value).strip()
    return value or None
