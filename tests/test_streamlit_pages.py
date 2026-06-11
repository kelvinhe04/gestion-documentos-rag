"""Verifica que el dashboard y todas sus páginas se ejecuten sin excepciones,
usando el framework oficial de pruebas de Streamlit (AppTest), sin navegador.

Uso:
    python tests/test_streamlit_pages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# La consola de Windows usa cp1252 y no puede imprimir emojis; forzamos UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGINAS = [
    "streamlit_app.py",
    "pages/1_📥_Ingesta_de_Documentos.py",
    "pages/2_💬_Chatbot_RAG.py",
    "pages/3_🔎_Búsqueda_Semántica.py",
    "pages/4_📊_Dashboard.py",
]


def main() -> int:
    fallos = 0
    for pagina in PAGINAS:
        try:
            at = AppTest.from_file(pagina, default_timeout=120)
            at.run()
            if len(at.exception) > 0:
                fallos += 1
                print(f"[FAIL] {pagina}")
                for exc in at.exception:
                    print(f"        {exc.value}")
            else:
                print(f"[OK]   {pagina}")
        except Exception as exc:  # noqa: BLE001
            fallos += 1
            print(f"[FAIL] {pagina} -> {exc}")

    print()
    if fallos == 0:
        print("[OK] Todas las páginas del dashboard se ejecutan sin errores.")
        return 0
    print(f"[ERROR] {fallos} página(s) con problemas.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
