"""
Quien esta trabajando ahora, en Tambora.

Incremento 2 del contrato de contexto. Contrato completo en
docs/runbooks/RUNBOOK_SERVIDOR_LINUX.md, seccion "Quien esta trabajando ahora".

OJO: Tambora NO ofrece contexto de usuario -- no sale en la franja "Sigue
donde lo dejaste" -- porque no tiene nada que reanudar: `/chat` no acepta una
conversacion concreta y su front arranca en blanco. Aqui si entra, porque el
punto verde solo dice que estas dentro y eso Tambora lo sabe perfectamente.

De SOLO LECTURA. El registro lo escribe `auth.get_current_user` en la
navegacion normal; esta ruta solo lo lee, y que no escriba importa: si se
apuntara a si misma, preguntar quien esta dentro pondria a alguien dentro.
"""

from __future__ import annotations

import logging
import os
import secrets

import aiosqlite
from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from . import auth

logger = logging.getLogger(__name__)

RUTA_ACTIVOS = "/api/plataforma/activos"
CABECERA = "X-SGE-Plataforma"

# Mismo cuerpo para "secreto ausente" y "secreto equivocado": que no se pueda
# distinguir evita ofrecer un oraculo de identidades.
_NO_AUTORIZADO = JSONResponse({"error": "no autorizado"}, status_code=401)


def _nombre_completo(fila) -> str:
    """ "Nombre Apellidos" si se sabe, y el usuario si no. Copia byte-identica
    de la de Bartolo, que es la convencion de este repositorio."""
    partes = [(fila["first_name"] or "").strip(), (fila["last_name"] or "").strip()]
    return " ".join(p for p in partes if p) or fila["username"]


def _token_configurado() -> str:
    return os.environ.get("TAMBORA_CONTEXTO_TOKEN", "").strip()


async def _activos(db: aiosqlite.Connection) -> dict:
    """Quien ha hecho algo en Tambora, con su sub de Keycloak.

    Quien no tiene `keycloak_sub` se omite: sin sub el recibidor no puede
    agrupar a la misma persona en varias herramientas.

    No se filtra por antiguedad: las fechas van en crudo y la ventana la
    aplica el recibidor, para que el criterio viva en un solo sitio."""
    vistos = dict(auth._VISTOS)
    if not vistos:
        return {"herramienta": "tambora", "activos": []}

    marcas = ",".join("?" for _ in vistos)
    cur = await db.execute(
        f"SELECT id, username, first_name, last_name, keycloak_sub FROM users "
        f"WHERE id IN ({marcas}) AND keycloak_sub IS NOT NULL",
        tuple(vistos),
    )
    activos = [
        # El nombre COMPLETO, que Tambora ya tiene de cuando auto-aprovisiona
        # por SSO. Mandar el `username` hacia que la misma persona saliera
        # distinta segun donde estuviera.
        {
            "sub": f["keycloak_sub"],
            "nombre": _nombre_completo(f),
            "visto_en": vistos[f["id"]].isoformat(),
        }
        for f in await cur.fetchall()
    ]
    return {"herramienta": "tambora", "activos": activos}


def register_activos(app, token: str | None = None) -> bool:
    """Registra la ruta. Sin secreto no se registra NADA -- 404, no 401 --
    para que una copia suelta en un portatil no exponga nada."""
    configurado = _token_configurado() if token is None else token.strip()
    if not configurado:
        logger.info("[CONTEXTO] sin secreto: no se registra %s", RUTA_ACTIVOS)
        return False

    @app.get(RUTA_ACTIVOS)
    async def activos_de_plataforma(
        request: Request,
        db: aiosqlite.Connection = Depends(auth.get_db),
    ):
        esperado = _token_configurado() or configurado
        # En bytes: compare_digest lanza TypeError con cadenas no-ASCII y
        # Starlette decodifica las cabeceras como latin-1. La ruta es publica.
        recibido = request.headers.get(CABECERA, "").encode("utf-8", "replace")
        if not secrets.compare_digest(recibido, esperado.encode("utf-8")):
            return _NO_AUTORIZADO
        return JSONResponse(await _activos(db))

    return True
