"""Vector store con ChromaDB: indexación y búsqueda semántica.

Cada fragmento (chunk) de texto se convierte en un embedding (vector) mediante
el modelo all-MiniLM-L6-v2 (DefaultEmbeddingFunction de Chroma, que corre con
ONNX, sin necesidad de GPU ni de torch). Los vectores se almacenan de forma
persistente en disco y permiten la búsqueda semántica por similitud de coseno.
"""
from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from src import config


@lru_cache(maxsize=1)
def _get_client() -> "chromadb.api.ClientAPI":
    """Cliente persistente de ChromaDB (cacheado: una sola instancia por proceso)."""
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


@lru_cache(maxsize=1)
def _embedding_function():
    """Función de embeddings por defecto de Chroma (all-MiniLM-L6-v2, ONNX)."""
    return embedding_functions.DefaultEmbeddingFunction()


def get_collection():
    """Devuelve (o crea) la colección de documentos académicos."""
    client = _get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma solo acepta valores str/int/float/bool en los metadatos.

    Convierte listas a texto separado por comas y descarta valores None.
    """
    clean: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def add_document(doc_id: str, chunks: List[str], metadata: Dict[str, Any]) -> int:
    """Indexa los fragmentos de un documento en el vector store.

    Usa `upsert` para que reindexar el mismo documento no genere duplicados.
    Devuelve el número de fragmentos indexados.
    """
    if not chunks:
        return 0

    collection = get_collection()
    base_meta = _sanitize_metadata(metadata)
    ingested_at = _dt.datetime.now().isoformat(timespec="seconds")

    ids = [f"{doc_id}::chunk{i}" for i in range(len(chunks))]
    metadatas = [
        {**base_meta, "doc_id": doc_id, "chunk_index": i, "ingested_at": ingested_at}
        for i in range(len(chunks))
    ]

    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def search(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    """Búsqueda semántica: devuelve los k fragmentos más similares a la consulta.

    Cada resultado incluye el texto, la metadata y un score de similitud
    (1 - distancia coseno; mayor es más relevante).
    """
    k = k or config.TOP_K
    collection = get_collection()

    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": doc,
                "metadata": meta or {},
                "distance": dist,
                "score": round(1 - dist, 4),  # similitud coseno
            }
        )
    return hits


def get_all(include: Optional[List[str]] = None) -> Dict[str, Any]:
    """Devuelve todos los registros de la colección (para estadísticas/ML)."""
    include = include or ["metadatas", "documents"]
    collection = get_collection()
    if collection.count() == 0:
        return {"ids": [], **{key: [] for key in include}}
    return collection.get(include=include)


def count_chunks() -> int:
    """Número total de fragmentos indexados."""
    return get_collection().count()


def reset_collection() -> None:
    """Elimina por completo la colección (útil para reconstruir el índice)."""
    client = _get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - no existe todavía
        pass
