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


class _FastEmbedFunction:
    """Wrapper de fastembed compatible con la interfaz de ChromaDB."""

    def __init__(self, model_name: str) -> None:
        import warnings
        from fastembed import TextEmbedding
        self._model_name = model_name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = TextEmbedding(model_name=model_name)

    def name(self) -> str:
        return f"fastembed-{self._model_name}"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return [emb.tolist() for emb in self._model.embed(texts)]

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._embed(input)

    def embed_documents(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._embed(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self._embed(input)


@lru_cache(maxsize=1)
def _embedding_function() -> "_FastEmbedFunction":
    """Embeddings multilingües vía fastembed (ONNX, sin PyTorch).

    paraphrase-multilingual-MiniLM-L12-v2: soporta español y 50+ idiomas.
    """
    return _FastEmbedFunction(model_name=config.EMBEDDING_MODEL)


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


def list_documents() -> List[Dict[str, Any]]:
    """Devuelve documentos únicos con metadatos y conteo de fragmentos."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["metadatas"])
    seen: Dict[str, Dict[str, Any]] = {}
    for meta in result["metadatas"]:
        doc_id = meta.get("doc_id", "")
        if not doc_id:
            continue
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "title": meta.get("title", doc_id),
                "filename": meta.get("filename", ""),
                "num_pages": meta.get("num_pages", "?"),
                "ingested_at": meta.get("ingested_at", ""),
                "chunk_count": 0,
            }
        seen[doc_id]["chunk_count"] += 1
    return sorted(seen.values(), key=lambda x: x["ingested_at"], reverse=True)


def delete_document(doc_id: str) -> int:
    """Elimina todos los fragmentos de un documento. Devuelve el nº eliminados."""
    collection = get_collection()
    result = collection.get(where={"doc_id": doc_id}, include=[])
    ids_to_delete = result["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)


def _query_hits(
    collection, query: str, n: int, where: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Ejecuta una query y devuelve lista de hits."""
    available = len(collection.get(where=where, include=[])["ids"])
    if available == 0:
        return []
    result = collection.query(
        query_texts=[query],
        n_results=min(n, available),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        result.get("documents", [[]])[0],
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0],
    ):
        hits.append({
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
            "score": round(1 - dist, 4),
        })
    return hits


def search_filtered(
    query: str, k: int | None = None, doc_ids: List[str] | None = None
) -> List[Dict[str, Any]]:
    """Búsqueda semántica con cobertura garantizada por documento.

    Estrategia híbrida:
    1. Query global → mejores k fragmentos entre todos los docs seleccionados.
    2. Query por documento → mejor fragmento de CADA doc (garantía de cobertura).
    Ambos conjuntos se fusionan, deduplicando por texto, y se devuelven los k
    mejores ordenados por similitud coseno.
    """
    if not doc_ids:
        return []
    k = k or config.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        return []

    seen_texts: set[str] = set()
    all_hits: List[Dict[str, Any]] = []

    def _add(hits: List[Dict[str, Any]]) -> None:
        for h in hits:
            if h["text"] not in seen_texts:
                seen_texts.add(h["text"])
                all_hits.append(h)

    # 1. Búsqueda global entre todos los documentos seleccionados
    global_where = (
        {"doc_id": {"$in": list(doc_ids)}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}
    )
    _add(_query_hits(collection, query, k, global_where))

    # 2. Garantía de cobertura: mejores 2 fragmentos de cada documento
    for doc_id in doc_ids:
        _add(_query_hits(collection, query, 2, {"doc_id": doc_id}))

    # Ordenar por similitud y devolver: al menos 1 por doc + top-k globales
    all_hits.sort(key=lambda x: x["distance"])
    return all_hits


def reset_collection() -> None:
    """Elimina por completo la colección (útil para reconstruir el índice)."""
    client = _get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - no existe todavía
        pass
