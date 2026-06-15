"""Fuente de datos 1: carga y procesamiento de PDFs locales.

Extrae texto y metadatos de archivos PDF usando pypdf. Si el PDF es escaneado
(sin capa de texto), intenta OCR con pytesseract + pymupdf como fallback.
Es la fuente principal del sistema: documentos académicos que el usuario sube.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Intentar configurar la ruta de Tesseract en Windows si no está en PATH
def _configure_tesseract() -> None:
    try:
        import pytesseract, shutil, os
        if shutil.which("tesseract"):
            return  # ya está en PATH
        # Rutas típicas de instalación en Windows
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        ]
        # También respetar variable de entorno del usuario
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd:
            candidates.insert(0, env_cmd)
        for candidate in candidates:
            if os.path.isfile(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                logger.debug("Tesseract encontrado en: %s", candidate)
                return
        logger.warning("Tesseract no encontrado en PATH ni en rutas por defecto.")
    except ImportError:
        pass

_configure_tesseract()

# Umbral mínimo de caracteres por página para considerar que pypdf extrajo texto
_MIN_CHARS_PER_PAGE = 50


def extract_pdf(path: str | Path) -> dict:
    """Extrae texto y metadatos de un PDF.

    Intenta extracción directa con pypdf; si el texto está vacío o es muy
    corto (PDF escaneado), cae en OCR con pytesseract + pymupdf.

    Devuelve un diccionario con:
        title, author, num_pages, text, filename, source
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el PDF: {path}")

    reader = PdfReader(str(path))
    num_pages = len(reader.pages)

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            pages_text.append("")
    text = "\n\n".join(pages_text)

    meta = reader.metadata
    title = _clean_meta(getattr(meta, "title", None)) or path.stem
    author = _clean_meta(getattr(meta, "author", None))

    # Si pypdf no obtuvo suficiente texto, intentar OCR
    avg_chars = len(text.strip()) / max(num_pages, 1)
    if avg_chars < _MIN_CHARS_PER_PAGE:
        logger.info("PDF '%s' parece escaneado (%.0f chars/pág). Intentando OCR…", path.name, avg_chars)
        ocr_text = _ocr_pdf(path, num_pages)
        if ocr_text:
            text = ocr_text

    return {
        "title": title,
        "author": author,
        "num_pages": num_pages,
        "text": text,
        "filename": path.name,
        "source": "local",
    }


def _ocr_pdf(path: Path, num_pages: int) -> str:
    """Renderiza cada página con pymupdf y aplica OCR con pytesseract.

    Devuelve el texto extraído, o cadena vacía si las dependencias no están
    disponibles (falla silenciosa para no romper la ingesta normal).
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("pymupdf no está instalado. Instala: pip install pymupdf")
        return ""

    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        logger.warning("pytesseract/Pillow no están instalados. Instala: pip install pytesseract pillow")
        return ""

    doc = fitz.open(str(path))
    pages_text: list[str] = []
    for page_num, page in enumerate(doc):
        try:
            # 200 DPI — buen balance entre velocidad y precisión OCR
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_result = pytesseract.image_to_string(img, lang="spa+eng")
            pages_text.append(ocr_result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR falló en página %d de '%s': %s", page_num + 1, path.name, exc)
            pages_text.append("")
    doc.close()

    return "\n\n".join(pages_text)


def _clean_meta(value: Optional[str]) -> Optional[str]:
    """Normaliza un valor de metadato del PDF (puede venir vacío o con espacios)."""
    if not value:
        return None
    value = str(value).strip()
    return value or None
