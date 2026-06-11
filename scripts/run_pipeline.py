"""Ejecuta el pipeline de datos desde la línea de comandos.

Ejemplos de uso (desde la raíz del proyecto):

    # Indexar todos los PDFs de la carpeta data/raw
    python scripts/run_pipeline.py --pdf-dir data/raw

    # Reconstruir el índice desde cero antes de ingestar
    python scripts/run_pipeline.py --pdf-dir data/raw --reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script directamente (python scripts/run_pipeline.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import pipeline  # noqa: E402
from src.indexing import vector_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de datos del sistema RAG (Grupo 7).")
    parser.add_argument("--pdf-dir", type=str, help="Carpeta con PDFs a indexar.")
    parser.add_argument("--pdf", type=str, help="Ruta a un único PDF a indexar.")
    parser.add_argument("--reset", action="store_true", help="Vacía el índice antes de ingestar.")
    args = parser.parse_args()

    if args.reset:
        print(">> Reiniciando la colección...")
        vector_store.reset_collection()

    total_chunks = 0
    total_docs = 0

    if args.pdf:
        res = pipeline.ingest_pdf_file(args.pdf)
        _print_result(res)
        total_chunks += res.get("chunks", 0)
        total_docs += 1 if res.get("ok") else 0

    if args.pdf_dir:
        print(f">> Indexando PDFs de: {args.pdf_dir}")
        for res in pipeline.ingest_pdf_directory(args.pdf_dir):
            _print_result(res)
            total_chunks += res.get("chunks", 0)
            total_docs += 1 if res.get("ok") else 0

    if not (args.pdf or args.pdf_dir):
        parser.print_help()
        return

    print("\n" + "=" * 50)
    print(f"Documentos indexados en esta corrida: {total_docs}")
    print(f"Fragmentos añadidos: {total_chunks}")
    print(f"Total de fragmentos en el índice: {vector_store.count_chunks()}")
    print("=" * 50)


def _print_result(res: dict) -> None:
    estado = "OK " if res.get("ok") else "ERR"
    titulo = res.get("title") or res.get("filename") or "?"
    if res.get("ok"):
        print(f"  [{estado}] {titulo} -> {res.get('chunks')} fragmentos ({res.get('source')})")
    else:
        print(f"  [{estado}] {titulo} -> {res.get('reason')}")


if __name__ == "__main__":
    main()
