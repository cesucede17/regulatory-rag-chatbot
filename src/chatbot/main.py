"""Chatbot BOE con RAG — aplicacion FastAPI independiente.

Se arranca por separado de las auditorias, desde la raiz del modulo:

    uv run python -m uvicorn chatbot.main:app --app-dir src --reload --host 127.0.0.1 --port 8501

Tiene su propia base de datos (`CHATBOT_DB_PATH`), su propio panel de admin
y sus propias plantillas y estaticos. No conoce nada del modulo de
auditorias.

Se entra SOLO por Keycloak: el login propio se retiro el 2026-09-15 (ver
src/chatbot/tambora_sso.py).
"""

import asyncio
import json
import logging
import mimetypes
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from shared import llm_usage
from shared.pricing import to_eur

from . import admin_users, auth, migrations, tambora_contexto
from .config import settings
from .core import usage_repository as chat_usage_repository
from .core.context_manager import get_available_contexts
from .core.pdf_loader import PDFLoader
from .core.vector_store import VectorStore
from .core.workflow import WorkflowOrchestrator
from starlette.middleware.sessions import SessionMiddleware

from . import tambora_sso
from .rutas import RUTA_BASE

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
# SIN root_path, y es importante: root_path significa "el proxy me pasa la
# ruta ENTERA, con prefijo". Traefik hace lo contrario, se lo quita
# (StripPrefix), asi que las dos cosas juntas se contradicen: Starlette
# busca /<prefijo>/static y solo llega /static, y no sirve NADA.
#
# Con el prefijo quitado, la aplicacion vive de verdad en la raiz. El
# prefijo solo existe en lo que se EMITE hacia el navegador, y de eso
# se encargan las plantillas.
app = FastAPI(title="Tambora · Chatbot BOE")

# El baile OIDC guarda su "state" entre la ida a Keycloak y la vuelta, y
# authlib lo guarda en request.session: sin este middleware, el callback
# falla con un "mismatching_state" que no explica nada. La cookie de sesion
# va firmada con la misma clave que los JWT -- no es la misma cookie ni el
# mismo mecanismo, solo el mismo secreto, y tener dos secretos para la misma
# instalacion solo multiplica las formas de equivocarse.
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.jwt_secret_key,
    same_site="lax",  # el callback llega de otro sitio; con strict no viaja
    https_only=settings.cookie_secure,
)

# Registrar el SSO aqui, al construir la app, y NO en el evento de arranque:
# si falta configuracion revienta ahora, al importar, en vez de dejar una
# herramienta que arranca y a la que nadie puede entrar. Es la consecuencia
# de no tener login propio.
tambora_sso.register_sso_routes(app)
# "Quien esta trabajando ahora" para el recibidor. Tambora NO ofrece contexto
# (no tiene nada que reanudar) pero si sabe quien esta dentro. Solo lectura:
# el registro lo escribe auth.get_current_user en la navegacion normal.
tambora_contexto.register_activos(app)

app.include_router(admin_users.router)

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent  # raiz del proyecto (src/chatbot/ -> src/ -> raiz)

ASSET_VERSION = "20260915a"  # sube esta fecha cada vez que cambie algo en static/

# En algunas instalaciones de Windows el registro resuelve ".js" a text/plain,
# lo que rompe la carga de <script type="module"> (comprobacion estricta de MIME).
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class NoCacheStatic(StaticFiles):
    """StaticFiles que fuerza revalidacion.

    El `?v=` de un <script type="module"> no se propaga a los `import`
    internos de esos modulos, asi que sin esto los submodulos pueden
    quedarse cacheados con codigo viejo de forma indetectable.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", NoCacheStatic(directory=_HERE / "static"), name="static")
jinja_env = Environment(
    loader=FileSystemLoader(str(_HERE / "templates")),
    autoescape=select_autoescape(["html", "htm"]),
)
# Global del entorno y no variable de cada render: asi las cuatro
# plantillas lo tienen sin tocar ninguna de las tres llamadas a
# render(), y la quinta que se anada tambien.
jinja_env.globals["p"] = RUTA_BASE

# ---------------------------------------------------------------------------
# Estado global del proceso (app local: compartido por todos los usuarios
# autenticados)
# ---------------------------------------------------------------------------
_pdf_loader = PDFLoader()
_vector_store = VectorStore()
_workflow = WorkflowOrchestrator(vector_store=_vector_store)
_check_running = False
_ALERTS_FILE = _ROOT / "data" / "doc_alerts.json"
_STATUS_FILE = _ROOT / "data" / "check_status.json"

_scheduler = AsyncIOScheduler()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verificacion del BOE (usada por el scheduler y por el disparo manual)
# ---------------------------------------------------------------------------


def _execute_boe_check() -> None:
    global _check_running
    try:
        from . import boe_monitor

        boe_monitor.run_check()
        _logger.info("[BOEScheduler] Verificacion completada.")
    except Exception as e:
        _logger.error(f"[BOEScheduler] Error en verificacion: {e}")
    finally:
        _check_running = False


async def _run_boe_check_job() -> None:
    global _check_running
    if _check_running:
        return
    _check_running = True
    _logger.info("[BOEScheduler] Iniciando verificacion BOE programada (10:00).")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _execute_boe_check)


# ---------------------------------------------------------------------------
# Ciclo de vida
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    await auth.init_db()

    applied = await migrations.run(settings.db_path)
    if applied:
        _logger.info("[Chatbot] Migraciones aplicadas: %s", applied)

    # users.keycloak_sub: el identificador con el que el SSO reconoce a cada
    # persona. Idempotente, corre en cada arranque.
    async with aiosqlite.connect(settings.db_path) as db:
        db.row_factory = aiosqlite.Row
        if await tambora_sso.migrar_keycloak_sub(db):
            _logger.info("[Chatbot] users.keycloak_sub anadida.")

    _scheduler.add_job(
        _run_boe_check_job,
        "cron",
        hour=10,
        minute=0,
        id="boe_daily_check",
        replace_existing=True,
    )
    _scheduler.start()
    _logger.info("[BOEScheduler] Programado: verificacion BOE diaria a las 10:00.")


@app.on_event("shutdown")
async def shutdown() -> None:
    _scheduler.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Entrada y salida
#
# El login propio se retiro el 2026-09-15: usuario y contrasena, el
# formulario, su CSRF y su limite de intentos. Se entra SOLO por Keycloak, y
# /sso/login, /sso/callback y /logout los registra chatbot/tambora_sso.py.
#
# Con ello desaparecen ADMIN_PASSWORD y SGE_PASSWORD -- dos contrasenas que
# compartian varias personas, que es justo lo que se queria quitar.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Manejador del 401: al SSO si es navegacion, JSON si es API
# ---------------------------------------------------------------------------


@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc):
    """Navegacion del navegador -> al SSO; llamadas de API -> JSON.

    Antes mandaba a /login, que ya no existe. Enviar a /sso/login hace que
    una sesion caducada se renueve sola: si la sesion de Keycloak sigue viva,
    la persona vuelve a donde estaba sin teclear nada."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept and not request.url.path.startswith("/api/"):
        return RedirectResponse(url=f"{RUTA_BASE}/sso/login", status_code=302)
    return JSONResponse({"error": str(exc.detail)}, status_code=401)


# ---------------------------------------------------------------------------
# Salud: la UNICA ruta publica, y lo es a proposito
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Sin sesion, a proposito: la usa el HEALTHCHECK del contenedor.

    Si dependiera de auth.get_current_user recibiria un 401 y el contenedor
    nunca se marcaria sano -- que es exactamente lo que le paso a PALBE antes
    de meter su /health en la lista blanca. Y Traefik no enruta a un
    contenedor que no esta sano, asi que la herramienta no apareceria.

    No revela nada: ni version, ni rutas, ni si existe tal usuario."""
    return {"status": "ok", "sso": tambora_sso.load_config() is not None}


# ---------------------------------------------------------------------------
# Paginas (protegidas)
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Ya no hay recibidor compartido con las auditorias: esta app ES el
    chatbot, y su unica pagina es /chat."""
    return RedirectResponse(url=f"{RUTA_BASE}/chat", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(
    request: Request,
    user: dict = Depends(auth.get_current_user),
):
    api_key_ok = bool(
        settings.anthropic_api_key
        and settings.anthropic_api_key != "your-anthropic-api-key-here"
    )
    template = jinja_env.get_template("chat.html")
    return HTMLResponse(
        template.render(
            api_key_ok=api_key_ok,
            user=user,
            asset_v=ASSET_VERSION,
            plataforma_url=tambora_sso.PLATAFORMA_URL,
        )
    )


@app.post("/api/chat")
async def chat(
    request: Request,
    user: dict = Depends(auth.get_current_user),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    data = await request.json()
    prompt: str = data.get("message", "").strip()
    history: list = data.get("history", [])
    active_contexts: list = data.get("active_contexts", [])
    chat_id: str | None = data.get("chat_id")

    if not prompt:
        return JSONResponse({"error": "Mensaje vacío"}, status_code=400)

    # 1. Daily rate limit
    await auth.check_and_increment_usage(user["id"], db)

    # 2. Concurrency queue (max 2 simultaneous per user)
    sem = auth.get_user_semaphore(user["id"])
    llm_usage.start_capture()
    try:
        async with sem:
            _workflow.llm_handler._active_contexts = active_contexts
            intent = _workflow.classify_intent(prompt)
            response, sources = _workflow.process_query(
                prompt, conversation_history=history
            )
    finally:
        # Persistir SIEMPRE lo capturado, incluso si process_query lanza a
        # mitad de camino (p.ej. tras una llamada de parameter_extraction
        # que sí tuvo éxito y facturó, pero antes de que la de chat_response
        # falle) — de lo contrario ese coste ya generado en la API de
        # Anthropic desaparece sin dejar rastro en chat_llm_usage. La
        # excepción original, si la hubo, sigue propagándose después de este
        # bloque.
        for purpose, usage in llm_usage.collect_capture():
            await chat_usage_repository.record(
                db, user_id=user["id"], purpose=purpose, usage=usage
            )

    intent_labels = {
        "sumario": "Sumario BOE",
        "search": "Legislación consolidada",
        "pdf": "Documentos cargados",
        "hybrid": "BOE + Documentos",
    }

    # 3. Persist to SQLite
    now = datetime.now(timezone.utc).isoformat()
    sources_json = json.dumps(sources, ensure_ascii=False)

    if not chat_id:
        chat_id = str(uuid.uuid4())
        title = prompt[:60]
        await db.execute(
            "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
            (chat_id, user["id"], title, now, now),
        )
    else:
        # Verify ownership
        row = await db.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])
        )
        if not await row.fetchone():
            chat_id = str(uuid.uuid4())
            title = prompt[:60]
            await db.execute(
                "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                (chat_id, user["id"], title, now, now),
            )

    await db.execute(
        "INSERT INTO messages (chat_id, role, content, intent, sources, created_at) VALUES (?,?,?,?,?,?)",
        (chat_id, "user", prompt, None, None, now),
    )
    await db.execute(
        "INSERT INTO messages (chat_id, role, content, intent, sources, created_at) VALUES (?,?,?,?,?,?)",
        (chat_id, "assistant", response, intent, sources_json, now),
    )
    await db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
    await db.commit()

    return JSONResponse(
        {
            "response": response,
            "sources": sources,
            "intent": intent,
            "intent_label": intent_labels.get(intent, intent),
            "chat_id": chat_id,
        }
    )


# ---------------------------------------------------------------------------
# Chat history routes
# ---------------------------------------------------------------------------


@app.get("/api/chats")
async def list_chats(
    user: dict = Depends(auth.get_current_user),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    rows = await db.execute(
        """SELECT c.id, c.title, c.updated_at,
                  COUNT(m.id) AS message_count
           FROM chats c
           LEFT JOIN messages m ON m.chat_id = c.id
           WHERE c.user_id = ?
           GROUP BY c.id
           ORDER BY c.updated_at DESC
           LIMIT 30""",
        (user["id"],),
    )
    chats = [dict(r) for r in await rows.fetchall()]
    return JSONResponse(chats)


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    user: dict = Depends(auth.get_current_user),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    # Admins can read any chat; regular users only their own
    if user["role"] == "admin":
        row = await db.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
    else:
        row = await db.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])
        )
    if not await row.fetchone():
        return JSONResponse({"error": "Conversación no encontrada."}, status_code=404)

    msgs = await db.execute(
        "SELECT role, content, intent, sources, created_at FROM messages WHERE chat_id = ? ORDER BY created_at",
        (chat_id,),
    )
    result = []
    for m in await msgs.fetchall():
        item = dict(m)
        if item.get("sources"):
            try:
                item["sources"] = json.loads(item["sources"])
            except Exception:
                pass
        result.append(item)
    return JSONResponse(result)


@app.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: dict = Depends(auth.get_current_user),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    row = await db.execute(
        "SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user["id"])
    )
    if not await row.fetchone():
        return JSONResponse({"error": "Conversación no encontrada."}, status_code=404)
    await db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    await db.commit()
    return JSONResponse({"success": True})


# ---------------------------------------------------------------------------
# Document routes (protected)
# ---------------------------------------------------------------------------


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    user: dict = Depends(auth.get_current_user),
):
    settings.documents_folder.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name  # strip any path separators
    if not safe_name:
        return JSONResponse({"error": "Nombre de archivo inválido"}, status_code=400)
    save_path = settings.documents_folder / safe_name
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Auto-index in background so the PDF is immediately available for queries
    def _auto_index():
        try:
            chunks = _pdf_loader.load_all_pdfs()
            if chunks:
                _vector_store.clear_collection()
                _workflow.rag_retriever.vector_store = _vector_store
                _workflow.rag_retriever.invalidate_filename_cache()
                _vector_store.add_documents(chunks)
        except Exception as e:
            _logger.error(f"[AutoIndex] Error: {e}")

    threading.Thread(target=_auto_index, daemon=True).start()

    return JSONResponse({"success": True, "filename": safe_name})


@app.post("/api/index")
async def index_docs(user: dict = Depends(auth.get_current_user)):
    chunks = _pdf_loader.load_all_pdfs()
    if chunks:
        _vector_store.clear_collection()
        _workflow.rag_retriever.vector_store = _vector_store
        _workflow.rag_retriever.invalidate_filename_cache()
        _vector_store.add_documents(chunks)
        return JSONResponse({"success": True, "chunks": len(chunks)})
    return JSONResponse({"success": False, "chunks": 0})


@app.get("/api/docs")
async def get_docs(user: dict = Depends(auth.get_current_user)):
    pdfs = _pdf_loader.get_pdf_list()
    count = _vector_store.get_collection_count()
    return JSONResponse({"pdfs": pdfs, "count": count})


@app.delete("/api/docs")
async def clear_docs(user: dict = Depends(auth.get_current_user)):
    _vector_store.clear_collection()
    return JSONResponse({"success": True})


@app.get("/api/pdf/{filename:path}")
async def serve_pdf(filename: str, user: dict = Depends(auth.get_current_user)):
    safe_name = Path(filename).name
    pdf_path = settings.documents_folder / safe_name
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        return JSONResponse({"error": "Documento no encontrado"}, status_code=404)
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@app.get("/pdfviewer", response_class=HTMLResponse)
async def pdfviewer(
    request: Request,
    user: dict = Depends(auth.get_current_user),
):
    template = jinja_env.get_template("pdfviewer.html")
    return HTMLResponse(template.render(asset_v=ASSET_VERSION))


# ---------------------------------------------------------------------------
# BOE alert routes (protected)
# ---------------------------------------------------------------------------


@app.get("/api/alerts")
async def get_alerts(user: dict = Depends(auth.get_current_user)):
    if not _ALERTS_FILE.exists():
        return JSONResponse({})
    try:
        return JSONResponse(json.loads(_ALERTS_FILE.read_text(encoding="utf-8")))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/alerts/check-status")
async def get_check_status(user: dict = Depends(auth.get_current_user)):
    if not _STATUS_FILE.exists():
        return JSONResponse({"running": False})
    try:
        return JSONResponse(json.loads(_STATUS_FILE.read_text(encoding="utf-8")))
    except Exception:
        return JSONResponse({"running": False})


@app.delete("/api/alerts/{filename:path}")
async def clear_alerts(filename: str, user: dict = Depends(auth.require_admin)):
    """Admin-only: clears alerts array AND verification state for a document."""
    safe_name = Path(filename).name
    if not _ALERTS_FILE.exists():
        return JSONResponse({"success": True})
    alerts_data = json.loads(_ALERTS_FILE.read_text(encoding="utf-8"))
    if safe_name in alerts_data:
        alerts_data[safe_name]["alerts"] = []
        alerts_data[safe_name].pop("verification", None)
        tmp = _ALERTS_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(alerts_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(_ALERTS_FILE)
    return JSONResponse({"success": True})


@app.post("/api/alerts/run-check")
async def run_alert_check(user: dict = Depends(auth.require_admin)):
    """Admin-only: trigger BOE check immediately."""
    global _check_running
    if _check_running:
        return JSONResponse(
            {"status": "already_running", "message": "Verificación ya en curso"}
        )
    _check_running = True
    threading.Thread(target=_execute_boe_check, daemon=True).start()
    return JSONResponse(
        {"status": "started", "message": "Verificación BOE iniciada en segundo plano"}
    )


# ---------------------------------------------------------------------------
# Context & debug routes
# ---------------------------------------------------------------------------


@app.get("/api/contexts")
async def list_contexts(user: dict = Depends(auth.get_current_user)):
    return JSONResponse({"contexts": get_available_contexts()})


@app.get("/api/debug-rag")
async def debug_rag(
    q: str = "auditoría energética ISO 50001",
    user: dict = Depends(auth.get_current_user),
):
    count = _vector_store.get_collection_count()
    embedding = _vector_store.generate_embedding(q)
    if not embedding:
        return JSONResponse({"error": "embedding vacío", "count": count})
    results = _vector_store.query_by_embedding(embedding, n_results=5)
    docs = (results.get("documents") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    hits = [
        {
            "distancia": round(d, 4),
            "archivo": m.get("filename", "?"),
            "pagina": m.get("page", "?"),
            "texto": doc[:200],
        }
        for doc, d, m in zip(docs, dists, metas)
    ]
    return JSONResponse({"count": count, "query": q, "resultados": hits})


# ---------------------------------------------------------------------------
# Admin routes (admin role required)
# ---------------------------------------------------------------------------


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    user: dict = Depends(auth.require_admin),
):
    template = jinja_env.get_template("admin.html")
    return HTMLResponse(
        template.render(
            user=user, asset_v=ASSET_VERSION, plataforma_url=tambora_sso.PLATAFORMA_URL
        )
    )


@app.get("/api/admin/stats")
async def admin_stats(
    user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    total_today_row = await db.execute(
        "SELECT COALESCE(SUM(count), 0) FROM daily_usage WHERE date = ?", (today,)
    )
    total_today = (await total_today_row.fetchone())[0]

    total_chats_row = await db.execute("SELECT COUNT(*) FROM chats")
    total_chats = (await total_chats_row.fetchone())[0]

    per_user_rows = await db.execute(
        """SELECT u.username, u.role, COALESCE(d.count, 0) AS count
           FROM users u
           LEFT JOIN daily_usage d ON d.user_id = u.id AND d.date = ?
           ORDER BY count DESC""",
        (today,),
    )
    per_user = [dict(r) for r in await per_user_rows.fetchall()]

    pdfs = _pdf_loader.get_pdf_list()
    count = _vector_store.get_collection_count()

    return JSONResponse(
        {
            "total_today": total_today,
            "total_chats": total_chats,
            "total_pdfs": len(pdfs),
            "total_chunks": count,
            "per_user": per_user,
        }
    )


@app.get("/api/admin/chats")
async def admin_chats(
    user_filter: str = None,
    user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    if user_filter:
        rows = await db.execute(
            """SELECT c.id, c.title, c.updated_at, u.username,
                      COUNT(m.id) AS message_count
               FROM chats c
               JOIN users u ON u.id = c.user_id
               LEFT JOIN messages m ON m.chat_id = c.id
               WHERE u.username = ?
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT 100""",
            (user_filter,),
        )
    else:
        rows = await db.execute(
            """SELECT c.id, c.title, c.updated_at, u.username,
                      COUNT(m.id) AS message_count
               FROM chats c
               JOIN users u ON u.id = c.user_id
               LEFT JOIN messages m ON m.chat_id = c.id
               GROUP BY c.id
               ORDER BY c.updated_at DESC
               LIMIT 100""",
        )
    return JSONResponse([dict(r) for r in await rows.fetchall()])


@app.get("/api/admin/chatbot/consumo")
async def admin_chatbot_consumo(
    desde: str | None = None,
    hasta: str | None = None,
    usuario: str | None = None,
    user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    user_id: int | None = None
    if usuario:
        row = await db.execute("SELECT id FROM users WHERE username = ?", (usuario,))
        found = await row.fetchone()
        if found is None:
            empty_totals = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0.0,
                "cost_eur": 0.0,
            }
            return JSONResponse({"rows": [], "totals": empty_totals})
        user_id = found["id"]

    summary = await chat_usage_repository.get_admin_usage_summary(
        db, since=desde, until=hasta, user_id=user_id
    )

    def _with_eur(rows):
        return [{**r, "cost_eur": round(to_eur(r["cost_usd"] or 0.0), 6)} for r in rows]

    return JSONResponse(
        {
            "rows": _with_eur(summary["rows"]),
            "totals": _with_eur([summary["totals"]])[0],
        }
    )
