"""Limpieza y segmentación (chunking) del texto extraído de los PDFs.

Estas funciones implementan la etapa de "Preprocesar y transformar los datos"
del pipeline:
  - clean_text: normaliza el texto crudo extraído del PDF.
  - chunk_text: divide el texto en fragmentos manejables y solapados para
    indexarlos en el vector store (cada fragmento se convierte en un embedding).
"""
from __future__ import annotations

import re
from typing import List

from src import config


def clean_text(text: str) -> str:
    """Normaliza el texto extraído de un PDF.

    - Elimina caracteres nulos y de control.
    - Une palabras cortadas por guion al final de línea ("infor-\\nmación").
    - Colapsa espacios y saltos de línea redundantes conservando los párrafos.
    """
    if not text:
        return ""

    # Caracteres nulos / de control no imprimibles.
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)

    # Une palabras divididas por guion de fin de línea: "infor-\nmación" -> "información".
    text = re.sub(r"-\s*\n\s*", "", text)

    # Normaliza saltos de línea: 2+ saltos = nuevo párrafo; 1 salto = espacio.
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Colapsa espacios múltiples.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def _split_sentences(text: str) -> List[str]:
    """Divide el texto en oraciones de forma sencilla (sin dependencias pesadas)."""
    # Divide tras signos de puntuación de cierre seguidos de espacio.
    parts = re.split(r"(?<=[.!?;])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[str]:
    """Divide el texto en fragmentos de ~chunk_size caracteres con solapamiento.

    El solapamiento (overlap) mantiene contexto entre fragmentos vecinos, lo
    que mejora la recuperación en RAG. El corte respeta los límites de oración
    cuando es posible y solo parte oraciones muy largas.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    overlap = min(overlap, max(0, chunk_size - 1))

    text = (text or "").strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # Oración aislada más larga que un chunk: se parte por la fuerza.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(sentence):
                chunks.append(sentence[start:start + chunk_size].strip())
                start += max(1, chunk_size - overlap)
            continue

        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current.strip())
            # Arranca el nuevo fragmento con la cola del anterior (solapamiento).
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()

    if current.strip():
        chunks.append(current.strip())

    # Filtra fragmentos vacíos o demasiado cortos para aportar información.
    return [c for c in chunks if len(c) >= 20]
