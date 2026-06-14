"""Técnica de Machine Learning: clustering de documentos con KMeans.

Esta es la técnica de ML requerida por el proyecto (clasificación, clustering
o regresión -> elegimos CLUSTERING). El flujo es:

  1. Se recuperan los embeddings de todos los fragmentos desde ChromaDB.
  2. Se agregan a nivel de documento (vector promedio de sus fragmentos).
  3. Se elige el número de clusters k óptimo con el coeficiente de silueta.
  4. Se entrena KMeans para agrupar documentos por temática.
  5. Se etiqueta cada cluster con sus términos más representativos (TF-IDF).
  6. Se proyectan los documentos a 2D con PCA para visualizarlos.

Todo esto alimenta el dashboard de estadísticas de los documentos.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.indexing import vector_store

# Palabras vacías en español + inglés (los PDFs académicos suelen estar en inglés).
_STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien",
    "desde", "todo", "nos", "durante", "uno", "ni", "contra", "ese", "eso",
    "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos",
    "son", "es", "fue", "ser", "los", "del", "las", "un", "una", "documento",
    "documentos", "trabajo", "estudio", "resultados", "datos", "método", "modelo",
}


def _document_vectors(
    chroma_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Agrega los embeddings de los fragmentos a vectores por documento.

    Devuelve dict con: doc_ids, X (matriz n_docs x dim), titles, sources, texts,
    n_chunks_por_doc.
    """
    if chroma_data is None:
        chroma_data = vector_store.get_all(
            include=["embeddings", "metadatas", "documents"]
        )

    embeddings = chroma_data.get("embeddings")
    embeddings = [] if embeddings is None else list(embeddings)
    metadatas = chroma_data.get("metadatas")
    metadatas = [] if metadatas is None else list(metadatas)
    documents = chroma_data.get("documents")
    documents = [] if documents is None else list(documents)

    if len(embeddings) == 0:
        return {"doc_ids": [], "X": np.empty((0, 0)), "titles": [], "sources": [], "texts": [], "n_chunks": []}

    acc_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    acc_texts: Dict[str, List[str]] = defaultdict(list)
    titles: Dict[str, str] = {}
    sources: Dict[str, str] = {}

    for emb, meta, doc in zip(embeddings, metadatas, documents):
        meta = meta or {}
        doc_id = meta.get("doc_id", "desconocido")
        acc_vectors[doc_id].append(np.asarray(emb, dtype=float))
        acc_texts[doc_id].append(doc or "")
        titles.setdefault(doc_id, meta.get("title", doc_id))
        sources.setdefault(doc_id, meta.get("source", "local"))

    doc_ids = list(acc_vectors.keys())
    X = np.vstack([np.mean(acc_vectors[d], axis=0) for d in doc_ids])
    texts = [" ".join(acc_texts[d]) for d in doc_ids]
    n_chunks = [len(acc_vectors[d]) for d in doc_ids]

    return {
        "doc_ids": doc_ids,
        "X": X,
        "titles": [titles[d] for d in doc_ids],
        "sources": [sources[d] for d in doc_ids],
        "texts": texts,
        "n_chunks": n_chunks,
    }


def _best_k(X: np.ndarray, k_max: int) -> int:
    """Elige el número de clusters con el mayor coeficiente de silueta."""
    n = X.shape[0]
    if n < 3:
        return 1
    k_max = min(k_max, n - 1)
    best_k, best_score = 2, -1.0
    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _top_terms(texts: List[str], labels: np.ndarray, top_n: int = 6) -> Dict[int, List[str]]:
    """Términos más representativos de cada cluster usando TF-IDF."""
    stop = list(_STOPWORDS_ES) + list(TfidfVectorizer(stop_words="english").get_stop_words())
    try:
        vectorizer = TfidfVectorizer(max_features=2000, stop_words=stop, ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        return {}
    terms = np.array(vectorizer.get_feature_names_out())

    result: Dict[int, List[str]] = {}
    for cluster_id in sorted(set(labels)):
        rows = np.where(labels == cluster_id)[0]
        mean_tfidf = np.asarray(tfidf[rows].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        result[int(cluster_id)] = [terms[i] for i in top_idx if mean_tfidf[i] > 0]
    return result


def cluster_documents(
    n_clusters: Optional[int] = None,
    k_max: int = 8,
) -> Dict[str, Any]:
    """Ejecuta el pipeline completo de clustering de documentos.

    Parámetros
    ----------
    n_clusters: nº de clusters; si es None se elige automáticamente por silueta.
    k_max: máximo k a considerar en la búsqueda automática.

    Devuelve un diccionario con los resultados listos para el dashboard, o un
    diccionario con clave "error" si no hay suficientes documentos.
    """
    data = _document_vectors()
    X = data["X"]
    n_docs = X.shape[0] if X.size else 0

    if n_docs < 2:
        return {
            "error": "Se necesitan al menos 2 documentos para hacer clustering.",
            "n_docs": n_docs,
        }

    k = n_clusters or _best_k(X, k_max)
    k = max(1, min(k, n_docs))

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Coeficiente de silueta global (calidad del clustering).
    sil = float(silhouette_score(X, labels)) if k > 1 and len(set(labels)) > 1 else 0.0

    # Proyección 2D para graficar.
    if X.shape[1] > 2 and n_docs >= 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        coords = np.zeros((n_docs, 2))

    top_terms = _top_terms(data["texts"], labels)

    documents = []
    for i, doc_id in enumerate(data["doc_ids"]):
        documents.append(
            {
                "doc_id": doc_id,
                "title": data["titles"][i],
                "source": data["sources"][i],
                "cluster": int(labels[i]),
                "n_chunks": data["n_chunks"][i],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
            }
        )

    cluster_sizes = {int(c): int(np.sum(labels == c)) for c in sorted(set(labels))}

    return {
        "k": k,
        "silhouette": round(sil, 4),
        "n_docs": n_docs,
        "documents": documents,
        "cluster_sizes": cluster_sizes,
        "top_terms": top_terms,
    }
