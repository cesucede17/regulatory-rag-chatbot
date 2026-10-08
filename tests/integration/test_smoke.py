"""Smoke test: los modulos del chatbot se importan sin errores en tiempo de
ejecucion."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


MODULES = [
    "chatbot.config",
    "chatbot.auth",
    "chatbot.migrations",
    "chatbot.admin_users",
    "chatbot.core.date_formatter",
    "chatbot.core.http_client",
    "chatbot.core.boe_search",
    "chatbot.core.llm_handler",
    "chatbot.core.parameter_extractor",
    "chatbot.core.pdf_loader",
    "chatbot.core.vector_store",
    "chatbot.core.rag_retriever",
    "chatbot.core.workflow",
    "chatbot.core.context_manager",
    "chatbot.core.usage_repository",
]


def test_modules_import() -> None:
    for module in MODULES:
        importlib.import_module(module)


def _route_paths(app) -> set[str]:
    """Rutas registradas en la app. `app.routes` mezcla rutas con routers
    incluidos segun la version de starlette, asi que se recorre en plano y se
    ignora lo que no tenga `path`."""
    paths: set[str] = set()
    pending = list(app.routes)
    while pending:
        r = pending.pop()
        path = getattr(r, "path", None)
        if isinstance(path, str):
            paths.add(path)
        pending.extend(getattr(getattr(r, "router", None), "routes", []) or [])
        if getattr(r, "routes", None) and not isinstance(path, str):
            pending.extend(r.routes)
    return paths


def test_app_imports_and_registers_its_routes() -> None:
    """La app entera tiene que poder construirse: es lo que hace uvicorn al
    arrancar, y aqui se detecta cualquier import roto tras la separacion."""
    from chatbot.main import app

    paths = _route_paths(app)
    assert "/chat" in paths
    assert "/api/chat" in paths
    assert "/api/admin/users" in paths
    # Ninguna ruta de auditorias vive ya en esta app.
    assert not any("auditoria" in p for p in paths)


def test_las_rutas_de_datos_de_la_app_son_las_del_monitor() -> None:
    """main.py y boe_monitor.py tienen que estar de acuerdo sobre donde vive
    `data/doc_alerts.json`: si main._ROOT se queda en `src/` en vez de subir
    hasta la raiz del modulo, /api/alerts lee y borra un fichero que
    boe_monitor nunca escribe."""
    import chatbot.boe_monitor as monitor
    import chatbot.main as main

    raiz = Path(__file__).resolve().parents[2]
    assert main._ROOT == raiz
    assert main._ALERTS_FILE.parent == raiz / "data"
    assert main._ALERTS_FILE == monitor._ROOT / "data" / "doc_alerts.json"
    assert main._STATUS_FILE == monitor._ROOT / "data" / "check_status.json"


@pytest.mark.parametrize("script", ["boe_monitor.py", "chat_cleanup.py"])
def test_los_scripts_sueltos_encuentran_sus_paquetes(script):
    """boe_monitor.py y chat_cleanup.py se pueden lanzar como `python
    src/chatbot/<fichero>.py`: sys.path[0] es entonces src/chatbot/, no la
    raiz del modulo, asi que `import chatbot...` tiene que resolverse
    anadiendo src/ (no la raiz) al path."""
    raiz = Path(__file__).resolve().parents[2]
    fichero = raiz / "src" / "chatbot" / script
    codigo = (
        "import sys, importlib.util as u;"
        f"sys.path[0] = {str(fichero.parent)!r};"  # lo que pone Python al ejecutar un script
        f"spec = u.spec_from_file_location('script_suelto', {str(fichero)!r});"
        "m = u.module_from_spec(spec); spec.loader.exec_module(m); print('ok')"
    )
    env = os.environ.copy()
    env.setdefault("TAMBORA_SSO_ISSUER", "http://auth.ejemplo.invalido/realms/sge")
    env.setdefault("TAMBORA_SSO_CLIENT_ID", "tambora")
    env.setdefault("TAMBORA_SSO_CLIENT_SECRET", "secreto-de-prueba")
    env.setdefault(
        "TAMBORA_SSO_REDIRECT_URI", "http://tambora.ejemplo.invalido/sso/callback"
    )
    env.setdefault("TAMBORA_CONTEXTO_TOKEN", "secreto-de-contexto-de-prueba")
    r = subprocess.run(
        [sys.executable, "-c", codigo],
        cwd=raiz,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert r.returncode == 0, r.stderr
