"""Gestión persistente de sesiones de chat (JSON en data/sessions.json).

Cada sesión guarda: nombre, lista de doc_ids asociados, historial de mensajes
y fecha de creación. Los mensajes se persisten en disco para sobrevivir a
recargas de la página.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config

_SESSIONS_FILE: Path = config.DATA_DIR / "sessions.json"


def _load() -> Dict[str, Any]:
    if _SESSIONS_FILE.exists():
        try:
            return json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def _save(sessions: Dict[str, Any]) -> None:
    _SESSIONS_FILE.write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def list_sessions() -> List[Dict[str, Any]]:
    """Devuelve todas las sesiones ordenadas por fecha (más reciente primero)."""
    sessions = _load()
    result = [{"id": k, **v} for k, v in sessions.items()]
    return sorted(result, key=lambda x: x.get("created_at", ""), reverse=True)


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Devuelve los datos de una sesión o None si no existe."""
    return _load().get(session_id)


def create_session(name: str = "") -> str:
    """Crea una nueva sesión y devuelve su ID."""
    sessions = _load()
    session_id = f"chat_{uuid.uuid4().hex[:8]}"
    sessions[session_id] = {
        "name": name or f"Chat {len(sessions) + 1}",
        "doc_ids": [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    _save(sessions)
    return session_id


def rename_session(session_id: str, new_name: str) -> None:
    """Renombra una sesión existente."""
    new_name = new_name.strip()
    if not new_name:
        return
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["name"] = new_name
        _save(sessions)


def set_session_docs(session_id: str, doc_ids: List[str]) -> None:
    """Actualiza la lista de documentos asociados a una sesión."""
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["doc_ids"] = doc_ids
        _save(sessions)


def delete_session(session_id: str) -> None:
    """Elimina una sesión y su historial."""
    sessions = _load()
    sessions.pop(session_id, None)
    _save(sessions)


def add_message(
    session_id: str, role: str, content: str, sources: List[Any] | None = None
) -> None:
    """Agrega un mensaje al historial de la sesión."""
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["messages"].append(
            {"role": role, "content": content, "sources": sources or []}
        )
        _save(sessions)


def clear_messages(session_id: str) -> None:
    """Borra el historial de mensajes de una sesión."""
    sessions = _load()
    if session_id in sessions:
        sessions[session_id]["messages"] = []
        _save(sessions)


def remove_doc_from_all_sessions(doc_id: str) -> None:
    """Elimina un doc_id de todas las sesiones (tras borrar el documento)."""
    sessions = _load()
    changed = False
    for sid in sessions:
        current = sessions[sid].get("doc_ids", [])
        updated = [d for d in current if d != doc_id]
        if updated != current:
            sessions[sid]["doc_ids"] = updated
            changed = True
    if changed:
        _save(sessions)
