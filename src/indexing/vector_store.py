"""Vector store con ChromaDB: indexación y búsqueda híbrida (semántica + BM25).

Flujo de búsqueda:
  1. Semántica  → embeddings (paraphrase-multilingual-MiniLM-L12-v2) + similitud coseno.
  2. BM25       → conteo de palabras exactas (rank-bm25).
  3. RRF        → Reciprocal Rank Fusion combina ambos rankings en uno solo.

Esto soluciona el caso en que la búsqueda semántica no encuentra términos
específicos como números de cédula, nombres propios o IDs exactos.
"""
from __future__ import annotations

import datetime as _dt
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb

from src import config


# ---------------------------------------------------------------------------
# Cliente y embeddings
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_client() -> "chromadb.api.ClientAPI":
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
    return _FastEmbedFunction(model_name=config.EMBEDDING_MODEL)


def get_collection():
    """Devuelve (o crea) la colección de documentos académicos."""
    client = _get_client()
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Metadatos
# ---------------------------------------------------------------------------

def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma solo acepta str/int/float/bool; convierte listas y descarta None."""
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


# ---------------------------------------------------------------------------
# Indexación
# ---------------------------------------------------------------------------

def add_document(doc_id: str, chunks: List[str], metadata: Dict[str, Any]) -> int:
    """Indexa los fragmentos de un documento. Devuelve el número de chunks."""
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

    # Invalidar caché BM25 al agregar documentos nuevos
    _bm25_cache.clear()

    return len(chunks)


# ---------------------------------------------------------------------------
# BM25 (búsqueda por palabras exactas)
# ---------------------------------------------------------------------------

# Caché: clave → {"sig": str, "index": BM25Okapi, "ids": list, "docs": list, "metas": list}
_bm25_cache: Dict[str, Any] = {}


def _tokenize(text: str) -> List[str]:
    """Tokeniza texto en minúsculas separando por caracteres no alfanuméricos."""
    return re.findall(r'\w+', text.lower())


def _get_bm25_index(collection, where: Optional[Dict] = None):
    """Construye (o recupera del caché) el índice BM25 para un subconjunto de chunks.

    La firma del caché es la lista ordenada de IDs en el subconjunto — si cambian
    (nueva ingesta o eliminación), el índice se reconstruye automáticamente.
    """
    from rank_bm25 import BM25Okapi

    result = collection.get(where=where, include=["documents", "metadatas"])
    ids: List[str] = result["ids"]
    docs: List[str] = result.get("documents") or []
    metas: List[Dict] = result.get("metadatas") or []

    cache_key = str(where)
    sig = ",".join(ids)  # cambia si se agrega o elimina cualquier chunk
    cached = _bm25_cache.get(cache_key)
    if cached and cached["sig"] == sig:
        return cached["index"], ids, docs, metas

    tokenized = [_tokenize(d) for d in docs]
    index = BM25Okapi(tokenized) if tokenized else None
    _bm25_cache[cache_key] = {"sig": sig, "index": index, "ids": ids, "docs": docs, "metas": metas}
    return index, ids, docs, metas


def _bm25_search(
    query: str, collection, n: int, where: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """Búsqueda BM25: devuelve los n chunks con mayor coincidencia de palabras."""
    index, ids, docs, metas = _get_bm25_index(collection, where)
    if not ids or index is None:
        return []

    raw_scores = index.get_scores(_tokenize(query))
    max_s = max(raw_scores) if raw_scores.any() and raw_scores.max() > 0 else 1.0

    ranked = sorted(enumerate(raw_scores), key=lambda x: x[1], reverse=True)
    hits = []
    for idx, raw in ranked[:n]:
        if raw <= 0:
            break
        hits.append({
            "text": docs[idx],
            "metadata": metas[idx] or {},
            "distance": 0.0,
            "score": round(float(raw) / float(max_s), 4),
        })
    return hits


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _rrf_merge(
    semantic: List[Dict[str, Any]],
    bm25: List[Dict[str, Any]],
    k_rrf: int = 60,
) -> List[Dict[str, Any]]:
    """Fusiona dos rankings usando RRF. Score final: Σ 1/(k + posición)."""
    rrf: Dict[str, float] = {}
    data: Dict[str, Dict] = {}

    for rank, hit in enumerate(semantic, start=1):
        key = hit["text"]
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (k_rrf + rank)
        data[key] = hit

    for rank, hit in enumerate(bm25, start=1):
        key = hit["text"]
        rrf[key] = rrf.get(key, 0.0) + 1.0 / (k_rrf + rank)
        if key not in data:
            data[key] = hit

    result = []
    for text, rrf_score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
        hit = dict(data[text])
        hit["score"] = round(rrf_score, 4)
        result.append(hit)
    return result


# ---------------------------------------------------------------------------
# Búsqueda pública
# ---------------------------------------------------------------------------

def search(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    """Búsqueda híbrida (semántica + BM25) sobre todos los documentos indexados.

    Aplica RRF para combinar ambos rankings y limita a MAX_CHUNKS_PER_DOC
    fragmentos por documento para evitar que colecciones grandes dominen.
    """
    k = k or config.TOP_K
    collection = get_collection()

    if collection.count() == 0:
        return []

    n_candidates = min(k * config.MAX_CHUNKS_PER_DOC * 4, collection.count())

    # Búsqueda semántica
    sem_result = collection.query(
        query_texts=[query],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )
    semantic_hits: List[Dict] = []
    for doc, meta, dist in zip(
        sem_result.get("documents", [[]])[0],
        sem_result.get("metadatas", [[]])[0],
        sem_result.get("distances", [[]])[0],
    ):
        semantic_hits.append({
            "text": doc,
            "metadata": meta or {},
            "distance": dist,
            "score": round(1 - dist, 4),
        })

    # Búsqueda BM25
    bm25_hits = _bm25_search(query, collection, n_candidates)

    # Fusión RRF
    merged = _rrf_merge(semantic_hits, bm25_hits)

    # Aplicar límite por documento y score mínimo
    hits: List[Dict] = []
    doc_count: Dict[str, int] = {}
    for hit in merged:
        if hit["score"] < config.MIN_SCORE:
            continue
        doc_id = hit["metadata"].get("doc_id", "")
        if doc_count.get(doc_id, 0) >= config.MAX_CHUNKS_PER_DOC:
            continue
        doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
        hits.append(hit)
        if len(hits) >= k:
            break

    return hits


def _query_hits(
    collection, query: str, n: int, where: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Búsqueda semántica filtrada (uso interno de search_filtered)."""
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
    """Búsqueda híbrida restringida a doc_ids seleccionados.

    1. Semántica + BM25 sobre el subconjunto → RRF merge.
    2. Límite MAX_CHUNKS_PER_DOC por documento.
    3. Garantía de cobertura: al menos 1 fragmento de cada doc seleccionado.
    """
    if not doc_ids:
        return []
    k = k or config.TOP_K
    collection = get_collection()
    if collection.count() == 0:
        return []

    where = (
        {"doc_id": {"$in": list(doc_ids)}} if len(doc_ids) > 1 else {"doc_id": doc_ids[0]}
    )
    n_candidates = min(k * config.MAX_CHUNKS_PER_DOC * 4, collection.count())

    # Semántica
    semantic_hits = _query_hits(collection, query, n_candidates, where)

    # BM25 sobre el mismo subconjunto
    bm25_hits = _bm25_search(query, collection, n_candidates, where)

    # RRF merge
    merged = _rrf_merge(semantic_hits, bm25_hits)

    # Aplicar límite por documento
    seen_texts: set[str] = set()
    doc_count: Dict[str, int] = {}
    hits: List[Dict] = []

    for hit in merged:
        if hit["text"] in seen_texts:
            continue
        doc_id = hit["metadata"].get("doc_id", "")
        if doc_count.get(doc_id, 0) >= config.MAX_CHUNKS_PER_DOC:
            continue
        seen_texts.add(hit["text"])
        doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
        hits.append(hit)

    # Garantía de cobertura: incluir al menos 1 chunk de cada doc seleccionado
    covered = {h["metadata"].get("doc_id") for h in hits}
    for doc_id in doc_ids:
        if doc_id not in covered:
            fallback = _query_hits(collection, query, 1, {"doc_id": doc_id})
            for h in fallback:
                if h["text"] not in seen_texts:
                    seen_texts.add(h["text"])
                    hits.append(h)

    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:k] if len(hits) > k else hits


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def get_all(include: Optional[List[str]] = None) -> Dict[str, Any]:
    """Devuelve todos los registros de la colección (para estadísticas/ML)."""
    include = include or ["metadatas", "documents"]
    collection = get_collection()
    if collection.count() == 0:
        return {"ids": [], **{key: [] for key in include}}
    return collection.get(include=include)


def count_chunks() -> int:
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
    _bm25_cache.clear()
    return len(ids_to_delete)


def reset_collection() -> None:
    """Elimina por completo la colección (útil para reconstruir el índice)."""
    client = _get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    _bm25_cache.clear()
