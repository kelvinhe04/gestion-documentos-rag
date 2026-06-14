"""Pipeline RAG: Retrieval-Augmented Generation.

Combina la búsqueda semántica (recuperación de fragmentos relevantes desde
ChromaDB) con la generación de una respuesta por parte del LLM, citando las
fuentes. Si no hay LLM disponible, devuelve una respuesta extractiva con los
fragmentos más relevantes.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src import config
from src.indexing import vector_store
from src.rag import llm

SYSTEM_PROMPT = (
    "Eres un asistente académico experto. Respondes preguntas ÚNICAMENTE con "
    "base en el CONTEXTO proporcionado, que proviene de documentos académicos. "
    "Si la respuesta no está en el contexto, indícalo claramente en lugar de inventar. "
    "Responde en español, de forma directa y conversacional. "
    "NO menciones números de fragmento, ni de dónde sacaste la información: "
    "simplemente responde la pregunta como si lo supieras."
)

PROMPT_TEMPLATE = """Usa el siguiente CONTEXTO para responder la PREGUNTA.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""


def _build_context(hits: List[Dict[str, Any]], prev_answer: str = "") -> str:
    """Construye el bloque de contexto a partir de los fragmentos recuperados.

    Si hay una respuesta previa del bot (pregunta de seguimiento), se antepone
    como contexto adicional para que el LLM pueda conectar los turnos.
    """
    chunks = "\n\n".join(hit["text"] for hit in hits)
    if prev_answer:
        return f"[Respuesta anterior del asistente]: {prev_answer}\n\n{chunks}"
    return chunks


def _extractive_answer(hits: List[Dict[str, Any]]) -> str:
    """Respuesta de respaldo cuando no hay LLM: muestra los fragmentos clave."""
    if not hits:
        return "No encontré información relevante en los documentos indexados."
    partes = ["Según los documentos indexados, esto es lo más relevante que encontré:\n"]
    for i, hit in enumerate(hits[:3], start=1):
        titulo = hit["metadata"].get("title", "Documento")
        fragmento = hit["text"].strip()
        if len(fragmento) > 500:
            fragmento = fragmento[:500] + "…"
        partes.append(f"**{i}. {titulo}** (relevancia {hit['score']}):\n{fragmento}")
    return "\n\n".join(partes)


def _expand_query(question: str, history: List[Dict[str, Any]]) -> str:
    """Expande una query corta con contexto del turno anterior (pregunta + respuesta).

    Combina la pregunta anterior del usuario con las primeras palabras de la
    respuesta del bot para anclar la búsqueda al tema correcto.
    Solo aplica cuando la query actual es corta (≤8 palabras).
    """
    if not history or len(question.split()) > 8:
        return question

    prev_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), None
    )
    prev_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), None
    )

    parts = [question]
    if prev_user:
        parts.append(prev_user)
    if prev_assistant:
        # Primeras 30 palabras de la respuesta anterior como contexto de búsqueda
        summary = " ".join(prev_assistant.split()[:30])
        parts.append(summary)

    return " ".join(parts)


def answer(
    question: str,
    k: int | None = None,
    doc_ids: List[str] | None = None,
    history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Responde una pregunta usando RAG.

    Devuelve un diccionario con:
        answer   -> texto de la respuesta
        sources  -> fragmentos recuperados (con título y score)
        provider -> proveedor de LLM usado ("groq" / "ollama" / "extractivo")

    history: turnos previos [{"role": "user"|"assistant", "content": str}]
             para dar contexto conversacional multi-turno al LLM.
    Si se pasa ``doc_ids``, la búsqueda se restringe a esos documentos.
    """
    k = k or config.TOP_K
    history = history or []

    # Expandir la query con contexto previo para mejorar el retrieval
    search_query = _expand_query(question, history)
    if doc_ids:
        hits = vector_store.search_filtered(search_query, k=k, doc_ids=doc_ids)
    else:
        hits = vector_store.search(search_query, k=k)

    if not hits:
        return {
            "answer": "Aún no hay documentos indexados (o ninguno es relevante). "
            "Sube PDFs en la página de Ingesta para comenzar.",
            "sources": [],
            "provider": "ninguno",
        }

    # Incluir respuesta previa en el contexto para preguntas de seguimiento cortas
    prev_answer = ""
    if len(question.split()) <= 8 and history:
        prev_answer = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
        )

    context = _build_context(hits, prev_answer)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    generated = llm.generate(prompt, system=SYSTEM_PROMPT, history=history or [])

    if generated and generated.strip():
        return {"answer": generated.strip(), "sources": hits, "provider": config.LLM_PROVIDER}

    # Respaldo extractivo (sin LLM o LLM no disponible).
    return {"answer": _extractive_answer(hits), "sources": hits, "provider": "extractivo"}
