"""
Quien esta trabajando ahora, en Tambora.

OJO: Tambora NO sale en la franja "Sigue donde lo dejaste" -- no tiene nada
que reanudar -- pero SI en este bloque, porque el punto verde solo dice que
estas dentro y eso Tambora lo sabe perfectamente. Los dos incrementos son
independientes y aqui se ve.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from chatbot import auth, migrations, tambora_sso

RUTA = "/api/plataforma/activos"
CABECERA = "X-SGE-Plataforma"
TOKEN = "secreto-de-contexto-de-prueba"
SUB = "sub-de-cesar"


@pytest.fixture(autouse=True)
def _limpio(monkeypatch):
    monkeypatch.setenv("TAMBORA_CONTEXTO_TOKEN", TOKEN)
    auth._VISTOS.clear()
    yield
    auth._VISTOS.clear()


@pytest.fixture
def base(tmp_path):
    ruta = tmp_path / "activos.db"

    async def montar():
        async with aiosqlite.connect(str(ruta)) as c:
            await c.executescript(auth._DB_SCHEMA)
            await c.commit()
        await migrations.run(str(ruta))
        async with aiosqlite.connect(str(ruta)) as c:
            c.row_factory = aiosqlite.Row
            await tambora_sso.migrar_keycloak_sub(c)
            await c.execute(
                "INSERT INTO users (id, username, password_hash, role, is_active,"
                " created_at, keycloak_sub) VALUES (1,'cesar','x','sge',1,?,?)",
                (datetime.now(timezone.utc).isoformat(), SUB),
            )
            await c.execute(
                "INSERT INTO users (id, username, password_hash, role, is_active,"
                " created_at) VALUES (2,'sin-vincular','x','sge',1,?)",
                (datetime.now(timezone.utc).isoformat(),),
            )
            await c.commit()

    asyncio.run(montar())
    return ruta


@pytest.fixture
def cliente(base):
    from chatbot.main import app

    async def _get_db():
        async with aiosqlite.connect(str(base)) as c:
            c.row_factory = aiosqlite.Row
            yield c

    app.dependency_overrides[auth.get_db] = _get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _pedir(c, token=TOKEN):
    cabeceras = {CABECERA: token} if token is not None else {}
    return c.get(RUTA, headers=cabeceras)


def test_pide_secreto(cliente):
    assert _pedir(cliente, token=None).status_code == 401
    assert _pedir(cliente, token="otro").status_code == 401


def test_una_cabecera_con_acentos_no_da_un_500(cliente):
    """La ruta es publica. En BYTES porque httpx se niega a enviar un str
    no-ASCII, asi que con un str el test fallaria en el cliente."""
    acentuado = ("secreto-con-" + chr(0xE9)).encode("latin-1")

    assert cliente.get(RUTA, headers={CABECERA: acentuado}).status_code == 401


def test_sin_nadie_dentro_la_lista_esta_vacia(cliente):
    assert _pedir(cliente).json()["activos"] == []


def test_una_peticion_autenticada_te_pone_en_la_lista(base):
    """Se llama a `get_current_user` DIRECTAMENTE: el cliente de tests puede
    sustituirla, y entonces la funcion real -- donde esta el registro -- no
    corre y el test pasaria sin probar nada."""
    from starlette.datastructures import Headers
    from starlette.requests import Request

    token = auth.create_access_token({"sub": "1"})

    async def autenticar():
        peticion = Request(
            {
                "type": "http",
                "headers": Headers({"cookie": f"{auth.COOKIE_NAME}={token}"}).raw,
            }
        )
        async with aiosqlite.connect(str(base)) as db:
            db.row_factory = aiosqlite.Row
            return await auth.get_current_user(peticion, db)

    usuario = asyncio.run(autenticar())

    assert usuario["username"] == "cesar"
    assert list(auth._VISTOS) == [1]


def test_la_ruta_devuelve_a_quien_esta_registrado(cliente):
    auth._VISTOS[1] = datetime.now(timezone.utc)

    activos = _pedir(cliente).json()["activos"]

    assert [a["nombre"] for a in activos] == ["cesar"]
    assert activos[0]["sub"] == SUB


def test_quien_no_esta_vinculado_no_sale(cliente):
    """Sin `keycloak_sub` no hay con que agrupar, asi que se omite en vez de
    inventarse un identificador."""
    auth._VISTOS[2] = datetime.now(timezone.utc)

    assert _pedir(cliente).json()["activos"] == []


def test_preguntar_no_te_pone_en_la_lista(cliente):
    """La ruta LEE el dato que la navegacion escribe. Si se apuntara a si
    misma, preguntar quien esta dentro pondria a alguien dentro."""
    _pedir(cliente)
    _pedir(cliente)

    assert auth._VISTOS == {}


def test_el_compose_pasa_el_secreto_al_contenedor():
    """Novena vez que este proyecto vigila una variable leyendo el compose."""
    import re

    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(
        encoding="utf-8"
    )

    assert re.search(r"^\s+TAMBORA_CONTEXTO_TOKEN:\s*\S", compose, re.MULTILINE)


def test_se_manda_el_nombre_completo_y_no_el_usuario(cliente, base):
    """Mismo fallo y mismo arreglo que en Bartolo: la misma persona salia con
    su usuario aqui y con su nombre en el recibidor."""

    async def poner_nombre():
        async with aiosqlite.connect(str(base)) as c:
            await c.execute(
                "UPDATE users SET first_name=?, last_name=? WHERE id=1",
                ("Cesar", "Suela"),
            )
            await c.commit()

    asyncio.run(poner_nombre())
    auth._VISTOS[1] = datetime.now(timezone.utc)

    assert _pedir(cliente).json()["activos"][0]["nombre"] == "Cesar Suela"


def test_sin_nombre_guardado_se_manda_el_usuario(cliente):
    auth._VISTOS[1] = datetime.now(timezone.utc)

    assert _pedir(cliente).json()["activos"][0]["nombre"] == "cesar"
