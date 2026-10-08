"""
La capa SSO: entrar en Tambora con la identidad de Keycloak.

Lo que protegen estos tests, en una frase cada uno:

- El rol lo manda Keycloak, y se recalcula en CADA entrada.
- Un usuario que ya existia por su nombre NO se duplica: se le graba el sub y
  conserva su id, y con el id sus conversaciones. Es el fallo que mas caro
  saldria, porque no da ningun error: la persona entra y no ve su historial.
- Un usuario desactivado no entra, aunque Keycloak diga que si.
- La cookie va con SameSite=Lax y no Strict. Con Strict el navegador NO la
  envia al volver de Keycloak -- es una navegacion entre sitios -- y la
  persona aterriza sin sesion, en bucle. El login propio usaba Strict porque
  su formulario era del mismo sitio.
- Sin configuracion completa, la aplicacion falla al arrancar. Al no haber
  login propio, una configuracion a medias deja la herramienta INACCESIBLE:
  mejor un error ruidoso que una puerta tapiada en silencio.

El esquema de la base se monta con auth._DB_SCHEMA, el de verdad, para que un
cambio de esquema se note aqui en vez de pasar desapercibido.
"""

import sys
from pathlib import Path

import aiosqlite
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatbot import auth, migrations, tambora_sso  # noqa: E402


def claims(
    sub="sub-1",
    username="ilasierra",
    nombre="Marta",
    apellido="Lasierra",
    roles=("tecnico", "tambora"),
    email="",
):
    """Un ID token de Keycloak, reducido a lo que la capa usa."""
    return {
        "sub": sub,
        "preferred_username": username,
        "given_name": nombre,
        "family_name": apellido,
        "email": email,
        "realm_access": {"roles": list(roles)},
    }


@pytest.fixture
async def db(tmp_path):
    """Base real, esquema real y MIGRACIONES reales, vacia.

    Las migraciones no son opcionales en el fixture: el esquema base de
    auth._DB_SCHEMA no tiene first_name ni last_name -- los anade
    migrations.auth_0001 reconstruyendo la tabla, y es lo que corre la
    aplicacion al arrancar. Sin ellas el fixture monta un esquema que no
    existe en ninguna parte."""
    ruta = tmp_path / "prueba.db"
    async with aiosqlite.connect(str(ruta)) as conn:
        await conn.executescript(auth._DB_SCHEMA)
        await conn.commit()
    await migrations.run(str(ruta))

    conn = await aiosqlite.connect(str(ruta))
    conn.row_factory = aiosqlite.Row
    await tambora_sso.migrar_keycloak_sub(conn)
    yield conn
    await conn.close()


# --------------------------------------------------------------- los roles


@pytest.mark.parametrize(
    "roles, esperado",
    [
        (("admin",), "admin"),
        (("tecnico",), "sge"),
        (("responsable_proyecto",), "sge"),
        ((), "sge"),
        (("tecnico", "admin"), "admin"),
        (("inventado",), "sge"),
    ],
)
def test_el_rol_sale_de_keycloak(roles, esperado):
    assert tambora_sso.rol_desde_claims(claims(roles=roles)) == esperado


# --------------------------------------------------------------- la migracion


async def test_la_migracion_anade_la_columna_y_es_idempotente(tmp_path):
    ruta = tmp_path / "m.db"
    async with aiosqlite.connect(str(ruta)) as c0:
        await c0.executescript(auth._DB_SCHEMA)
        await c0.commit()
    await migrations.run(str(ruta))
    conn = await aiosqlite.connect(str(ruta))
    conn.row_factory = aiosqlite.Row

    async def columnas():
        cur = await conn.execute("PRAGMA table_info(users)")
        return {r["name"] for r in await cur.fetchall()}

    assert "keycloak_sub" not in await columnas()
    assert await tambora_sso.migrar_keycloak_sub(conn) is True
    assert "keycloak_sub" in await columnas()
    # Relanzarla no debe fallar ni cambiar nada: se ejecuta en cada arranque.
    assert await tambora_sso.migrar_keycloak_sub(conn) is False
    assert "keycloak_sub" in await columnas()
    await conn.close()


# --------------------------------------------------------------- resolver


async def test_una_identidad_nueva_crea_su_usuario(db):
    user = await tambora_sso.resolver_identidad(db, claims())

    assert user is not None
    assert user["username"] == "ilasierra"
    assert user["role"] == "sge"
    assert user["is_active"] == 1

    cur = await db.execute(
        "SELECT keycloak_sub, first_name, last_name FROM users WHERE id = ?",
        (user["id"],),
    )
    fila = await cur.fetchone()
    assert fila["keycloak_sub"] == "sub-1"
    assert fila["first_name"] == "Marta"
    assert fila["last_name"] == "Lasierra"


async def test_un_usuario_previo_conserva_su_id_y_sus_conversaciones(db):
    """El caso que no da ningun error y duele: si se crea un duplicado, la
    persona entra y su historial no esta -- las conversaciones cuelgan de
    chats.user_id."""
    await db.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
        ("ilasierra", "hash-viejo", "sge", "2026-01-01T00:00:00"),
    )
    await db.commit()
    cur = await db.execute("SELECT id FROM users WHERE username = 'ilasierra'")
    id_previo = (await cur.fetchone())["id"]
    await db.execute(
        "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (
            "chat-1",
            id_previo,
            "Consulta del BOE",
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
        ),
    )
    await db.commit()

    user = await tambora_sso.resolver_identidad(db, claims())

    assert user["id"] == id_previo, "se ha creado un usuario duplicado"
    cur = await db.execute("SELECT COUNT(*) AS n FROM users")
    assert (await cur.fetchone())["n"] == 1
    cur = await db.execute(
        "SELECT COUNT(*) AS n FROM chats WHERE user_id = ?", (id_previo,)
    )
    assert (await cur.fetchone())["n"] == 1, "la persona ha perdido su historial"
    cur = await db.execute("SELECT keycloak_sub FROM users WHERE id = ?", (id_previo,))
    assert (await cur.fetchone())["keycloak_sub"] == "sub-1"


async def test_entrar_dos_veces_no_crea_nada(db):
    primera = await tambora_sso.resolver_identidad(db, claims())
    segunda = await tambora_sso.resolver_identidad(db, claims())

    assert primera["id"] == segunda["id"]
    cur = await db.execute("SELECT COUNT(*) AS n FROM users")
    assert (await cur.fetchone())["n"] == 1


async def test_el_rol_se_recalcula_en_cada_entrada(db):
    """Keycloak manda. Si alguien deja de ser admin alli, deja de serlo aqui
    en su siguiente entrada, sin que nadie toque la base a mano."""
    user = await tambora_sso.resolver_identidad(db, claims(roles=("admin",)))
    assert user["role"] == "admin"

    user = await tambora_sso.resolver_identidad(db, claims(roles=("tecnico",)))
    assert user["role"] == "sge"

    cur = await db.execute("SELECT role FROM users WHERE id = ?", (user["id"],))
    assert (await cur.fetchone())["role"] == "sge"


async def test_el_nombre_de_usuario_cambiado_en_keycloak_se_sigue_por_el_sub(db):
    """El sub es el identificador de verdad: si alguien se renombra en
    Keycloak, sigue siendo la misma persona y conserva su historial."""
    user = await tambora_sso.resolver_identidad(db, claims())
    cur = await db.execute("SELECT COUNT(*) AS n FROM users")
    assert (await cur.fetchone())["n"] == 1

    otro = await tambora_sso.resolver_identidad(db, claims(username="ilasierra2"))
    assert otro["id"] == user["id"]
    assert otro["username"] == "ilasierra2"
    cur = await db.execute("SELECT COUNT(*) AS n FROM users")
    assert (await cur.fetchone())["n"] == 1


async def test_un_usuario_desactivado_no_entra(db):
    user = await tambora_sso.resolver_identidad(db, claims())
    await db.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user["id"],))
    await db.commit()

    assert await tambora_sso.resolver_identidad(db, claims()) is None


async def test_un_token_sin_sub_no_entra(db):
    c = claims()
    del c["sub"]
    assert await tambora_sso.resolver_identidad(db, c) is None


async def test_un_token_sin_nombre_de_usuario_no_entra(db):
    """Sin nombre de usuario no hay con que crear la ficha, y username es
    NOT NULL UNIQUE. Mejor rechazar que insertar una cadena vacia que
    colisionaria con la siguiente persona en el mismo caso."""
    c = claims(username="")
    assert await tambora_sso.resolver_identidad(db, c) is None


# --------------------------------------------------------------- la cookie


def test_la_cookie_va_con_samesite_lax_y_no_strict():
    """Con Strict el navegador no envia la cookie al volver de Keycloak -- es
    una navegacion entre sitios -- y la persona aterriza sin sesion, en
    bucle. El login propio podia usar Strict porque su formulario era del
    mismo sitio; este camino no."""
    from fastapi.responses import RedirectResponse

    resp = RedirectResponse("/chat", status_code=302)
    tambora_sso.poner_cookie(resp, {"id": 7, "role": "sge"})

    cabecera = resp.headers["set-cookie"]
    assert auth.COOKIE_NAME in cabecera
    assert "samesite=lax" in cabecera.lower()
    assert "strict" not in cabecera.lower()
    assert "httponly" in cabecera.lower()


def test_la_cookie_lleva_lo_mismo_que_ponia_el_login():
    """El resto de la aplicacion lee el usuario de esta cookie a traves de
    auth.get_current_user. Si el contenido no es el mismo, las ~20 rutas
    protegidas dejan de funcionar."""
    from fastapi.responses import RedirectResponse

    resp = RedirectResponse("/chat", status_code=302)
    tambora_sso.poner_cookie(resp, {"id": 7, "role": "admin"})

    valor = resp.headers["set-cookie"].split("=", 1)[1].split(";")[0]
    payload = auth.verify_token(valor)
    assert payload["sub"] == "7"
    assert payload["role"] == "admin"


# --------------------------------------------------------------- configuracion


def test_sin_configuracion_completa_la_aplicacion_no_arranca(monkeypatch):
    """No hay login propio al que caer: una configuracion a medias dejaria la
    herramienta inaccesible. Que se note al arrancar."""
    for v in (
        "TAMBORA_SSO_ISSUER",
        "TAMBORA_SSO_CLIENT_ID",
        "TAMBORA_SSO_CLIENT_SECRET",
        "TAMBORA_SSO_REDIRECT_URI",
    ):
        monkeypatch.delenv(v, raising=False)

    assert tambora_sso.load_config() is None

    from fastapi import FastAPI

    with pytest.raises(RuntimeError) as exc:
        tambora_sso.register_sso_routes(FastAPI())
    assert "TAMBORA_SSO" in str(exc.value)


def test_con_la_configuracion_puesta_se_registran_las_rutas(monkeypatch):
    monkeypatch.setenv("TAMBORA_SSO_ISSUER", "http://auth.ejemplo/realms/sge")
    monkeypatch.setenv("TAMBORA_SSO_CLIENT_ID", "tambora")
    monkeypatch.setenv("TAMBORA_SSO_CLIENT_SECRET", "secreto")
    monkeypatch.setenv(
        "TAMBORA_SSO_REDIRECT_URI", "http://tambora.ejemplo/sso/callback"
    )

    from fastapi import FastAPI

    app = FastAPI()
    tambora_sso.register_sso_routes(app)
    rutas = {getattr(r, "path", None) for r in app.routes}
    assert "/sso/login" in rutas
    assert "/sso/callback" in rutas


def test_la_url_de_cierre_lleva_siempre_el_client_id(monkeypatch):
    """Keycloak exige id_token_hint O client_id para validar la URI de
    vuelta. Sin ninguno de los dos devuelve un 400 en crudo. Es un defecto
    que este proyecto ya cometio dos veces, en PALBE y en el recibidor."""
    monkeypatch.setenv("TAMBORA_SSO_ISSUER", "http://auth.ejemplo/realms/sge")
    monkeypatch.setenv("TAMBORA_SSO_CLIENT_ID", "tambora")
    monkeypatch.setenv("TAMBORA_SSO_CLIENT_SECRET", "secreto")
    monkeypatch.setenv(
        "TAMBORA_SSO_REDIRECT_URI", "http://tambora.ejemplo/sso/callback"
    )
    monkeypatch.setenv("TAMBORA_SSO_POST_LOGOUT_URI", "http://tambora.ejemplo/")

    url = tambora_sso.url_de_cierre("")
    assert "client_id=tambora" in url
    assert "post_logout_redirect_uri=" in url

    con_token = tambora_sso.url_de_cierre("un-id-token")
    assert "id_token_hint=un-id-token" in con_token
    assert "client_id=tambora" in con_token


# --------------------------------------------------------------- sin login propio


def _rutas(app):
    salida = set()
    pendientes = list(app.routes)
    while pendientes:
        r = pendientes.pop()
        path = getattr(r, "path", None)
        if isinstance(path, str):
            salida.add(path)
        pendientes.extend(getattr(getattr(r, "router", None), "routes", []) or [])
    return salida


def test_el_login_propio_ya_no_existe():
    """Se retiro el 2026-09-15. Si alguien lo reintroduce, vuelven las
    contrasenas compartidas que este trabajo venia a quitar."""
    from chatbot.main import app

    rutas = _rutas(app)
    assert "/login" not in rutas
    assert "/sso/login" in rutas
    assert "/sso/callback" in rutas
    assert "/logout" in rutas


def test_la_app_lleva_el_middleware_de_sesion():
    """authlib guarda el state del baile OIDC en request.session. Sin este
    middleware el callback falla con un mismatching_state que no explica
    nada, y solo se ve intentando entrar de verdad."""
    from starlette.middleware.sessions import SessionMiddleware

    from chatbot.main import app

    assert any(m.cls is SessionMiddleware for m in app.user_middleware)


def test_una_navegacion_sin_sesion_va_al_sso():
    """Antes iba a /login. Mandarla al SSO hace que una sesion caducada se
    renueve sola si la de Keycloak sigue viva."""
    from fastapi.testclient import TestClient

    from chatbot.main import app

    with TestClient(app) as cliente:
        r = cliente.get(
            "/chat", headers={"accept": "text/html"}, follow_redirects=False
        )
    assert r.status_code == 302
    assert r.headers["location"] == "/sso/login"


def test_una_llamada_de_api_sin_sesion_sigue_dando_json():
    """El navegador se redirige; una llamada de API no, o el cliente recibe
    una pagina HTML donde esperaba datos."""
    from fastapi.testclient import TestClient

    from chatbot.main import app

    with TestClient(app) as cliente:
        r = cliente.get("/api/docs", headers={"accept": "application/json"})
    assert r.status_code == 401
    assert r.headers["content-type"].startswith("application/json")


def test_health_responde_sin_sesion():
    """La usa el HEALTHCHECK del contenedor. Si pidiera sesion, recibiria un
    401, el contenedor no se marcaria sano y Traefik no enrutaria a el: la
    herramienta no apareceria, sin ningun error que lo explique."""
    from fastapi.testclient import TestClient

    from chatbot.main import app

    with TestClient(app) as cliente:
        r = cliente.get("/health")

    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["sso"] is True
    # Que no filtre nada de mas: ni version, ni rutas, ni usuarios.
    assert set(cuerpo) == {"status", "sso"}


def test_no_queda_ninguna_contrasena_compartida():
    """Las dos contrasenas compartidas desaparecieron con el login. Que no
    vuelvan por la puerta de atras en la configuracion.

    Se comprueban ASIGNACIONES y ATRIBUTOS, no menciones: los comentarios que
    explican que esas variables se retiraron son utiles y las nombran. Mi
    primera version de este test buscaba la cadena suelta y fallaba por un
    comentario propio -- un test que se dispara con su propia documentacion
    acaba borrandola."""
    import re as _re
    from pathlib import Path as _P

    from chatbot.config import settings

    # La configuracion ya no expone esos campos
    assert not hasattr(settings, "admin_password")
    assert not hasattr(settings, "sge_password")

    raiz = _P(__file__).resolve().parents[2]

    # Ni se leen del entorno en ningun sitio
    for fichero in [
        "src/chatbot/auth.py",
        "src/chatbot/config.py",
        "src/chatbot/main.py",
    ]:
        texto = (raiz / fichero).read_text(encoding="utf-8")
        assert not _re.search(r'getenv\(\s*["\']\s*(ADMIN|SGE)_PASSWORD', texto), (
            fichero
        )
        assert not _re.search(r"settings\.(admin|sge)_password", texto), fichero

    # Ni se declaran como variables en la plantilla de entorno
    plantilla = (raiz / ".env.example").read_text(encoding="utf-8")
    for linea in plantilla.splitlines():
        limpia = linea.strip()
        if limpia.startswith("#"):
            continue
        assert not limpia.startswith(("ADMIN_PASSWORD", "SGE_PASSWORD")), limpia
