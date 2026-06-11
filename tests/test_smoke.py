"""Prueba de humo del sistema: valida el núcleo sin depender de la red
(solo descarga el modelo de embeddings la primera vez).

Ejercita: indexación en ChromaDB -> búsqueda semántica -> clustering -> RAG.

Uso:
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Usa una colección AISLADA para no tocar los documentos reales del usuario.
from src import config  # noqa: E402

config.COLLECTION_NAME = "documentos_test_smoke"

from src.indexing import vector_store  # noqa: E402
from src.ml import clustering  # noqa: E402
from src.preprocessing.text_processing import chunk_text, clean_text  # noqa: E402
from src.rag import rag_pipeline  # noqa: E402

# Cuatro documentos de dos temáticas claramente distintas.
DOCUMENTOS = {
    "doc_ml_1": (
        "Las redes neuronales artificiales aprenden patrones a partir de datos. "
        "El entrenamiento ajusta los pesos mediante descenso del gradiente y "
        "retropropagación. Las redes neuronales profundas tienen muchas capas "
        "ocultas y se usan en visión por computadora y procesamiento de lenguaje. "
    ) * 6,
    "doc_ml_2": (
        "El aprendizaje automático supervisado entrena modelos con ejemplos "
        "etiquetados para clasificar o predecir. Algoritmos como árboles de "
        "decisión, máquinas de soporte vectorial y redes neuronales optimizan "
        "una función de pérdida durante el entrenamiento del modelo. "
    ) * 6,
    "doc_db_1": (
        "Las bases de datos relacionales almacenan información en tablas con "
        "filas y columnas. El lenguaje SQL permite consultar, insertar y "
        "actualizar registros. Los índices aceleran las consultas y las claves "
        "primarias garantizan la integridad de los datos almacenados. "
    ) * 6,
    "doc_db_2": (
        "Una base de datos vectorial almacena embeddings y permite búsquedas por "
        "similitud. La indexación con estructuras como HNSW hace eficiente la "
        "recuperación de los vectores más cercanos según la distancia coseno "
        "entre las representaciones numéricas de los documentos. "
    ) * 6,
}


def main() -> int:
    print(">> Reiniciando colección de prueba...")
    vector_store.reset_collection()

    print(">> Indexando documentos de prueba...")
    for doc_id, texto in DOCUMENTOS.items():
        chunks = chunk_text(clean_text(texto))
        n = vector_store.add_document(
            doc_id, chunks, {"title": doc_id, "source": "test", "num_pages": 1}
        )
        print(f"   - {doc_id}: {n} fragmentos")

    total = vector_store.count_chunks()
    assert total > 0, "No se indexó ningún fragmento"
    print(f">> Total de fragmentos: {total}")

    print(">> Probando búsqueda semántica...")
    hits = vector_store.search("entrenamiento de redes neuronales", k=3)
    assert hits, "La búsqueda no devolvió resultados"
    print(f"   Top resultado: {hits[0]['metadata']['title']} (score {hits[0]['score']})")

    print(">> Probando clustering (KMeans)...")
    res = clustering.cluster_documents()
    assert "error" not in res, f"Clustering falló: {res.get('error')}"
    print(f"   k={res['k']} · silueta={res['silhouette']} · docs={res['n_docs']}")

    print(">> Probando RAG (respuesta)...")
    ans = rag_pipeline.answer("¿Qué es una base de datos vectorial?")
    assert ans["answer"], "El RAG no devolvió respuesta"
    print(f"   Proveedor: {ans['provider']} · fuentes: {len(ans['sources'])}")

    # Limpia la colección de prueba para no dejar datos residuales.
    vector_store.reset_collection()

    print("\n[OK] Prueba de humo superada: el sistema funciona de extremo a extremo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
