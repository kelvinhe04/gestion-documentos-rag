"""Abstracción del proveedor de LLM (open source) para el chatbot RAG.

Soporta tres modos, seleccionables con la variable de entorno LLM_PROVIDER:

  - "groq"   -> API de Groq, que sirve modelos OPEN SOURCE (Llama 3) muy rápido
                y con un tier gratuito. Requiere GROQ_API_KEY.
  - "ollama" -> LLM local servido por Ollama (100% offline).
  - "none"   -> sin LLM generativo.

Si el proveedor configurado no está disponible (sin API key, sin servidor),
`generate` devuelve None y el pipeline RAG recurre a una respuesta extractiva,
de modo que el sistema SIEMPRE responde algo útil.
"""
from __future__ import annotations

from typing import Optional

import requests

from src import config


def generate(
    prompt: str,
    system: str = "",
    history: Optional[list] = None,
) -> Optional[str]:
    """Genera texto con el proveedor configurado. Devuelve None si no hay LLM.

    history: lista de {"role": "user"|"assistant", "content": str} con turnos previos.
    """
    provider = config.LLM_PROVIDER
    history = history or []

    if provider == "groq" and config.GROQ_API_KEY:
        return _generate_groq(prompt, system, history)
    if provider == "ollama":
        return _generate_ollama(prompt, system, history)
    return None


def _generate_groq(prompt: str, system: str, history: list) -> Optional[str]:
    """Genera con la API de Groq (modelos Llama 3 open source)."""
    try:
        from groq import Groq
    except ImportError:
        print("[LLM] El paquete 'groq' no está instalado.")
        return None

    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 - cualquier fallo -> respaldo extractivo
        print(f"[LLM:Groq] Error: {exc}")
        return None


def _generate_ollama(prompt: str, system: str, history: list) -> Optional[str]:
    """Genera con un modelo local servido por Ollama."""
    try:
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": config.LLM_TEMPERATURE},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content")
    except Exception as exc:  # noqa: BLE001 - servidor caído -> respaldo extractivo
        print(f"[LLM:Ollama] Error: {exc}")
        return None
