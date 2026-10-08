"""
La capa OIDC contra Keycloak: la UNICA forma de entrar en Tambora.

Vive en su propio fichero, como `palbe_sso.py` en PALBE, y por el mismo
motivo: el baile OIDC no tiene nada que ver con el chatbot y se revisa mejor
aparte.

**La diferencia con PALBE, que es la razon de ser de este modulo:** aqui no
hay login propio. PALBE conservo el suyo como via de emergencia; Tambora lo
retira, asi que esto no es una alternativa a la contrasena sino el unico
camino. Dos consecuencias que hay que tener presentes:

1. **Sin configuracion completa la aplicacion no arranca.** En PALBE una
   configuracion a medias dejaba las rutas /sso/* sin registrar y el login
   propio seguia funcionando. Aqui eso dejaria la herramienta tapiada en
   silencio, asi que `register_sso_routes` revienta a proposito.
2. **Si Keycloak esta caido, nadie entra.** Decision consciente, tomada el
   2026-09-15 (ver docs/superpowers/specs/2026-09-15-tambora-en-la-plataforma-design.md).

El contrato con el resto de la aplicacion es minusculo: al final del baile se
pone la MISMA cookie JWT que ponia el login (`auth.create_access_token` con
`sub` y `role`). Las ~20 rutas protegidas leen de `auth.get_current_user` y no
se enteran de por donde entro nadie.

Y una trampa que no es obvia: esa cookie va con **SameSite=Lax**, no Strict.
El login propio usaba Strict y era correcto -- su formulario era del mismo
sitio -- pero al volver de Keycloak la navegacion es ENTRE SITIOS y el
navegador no envia una cookie Strict: la persona aterrizaria en /chat sin
sesion, y de ahi otra vez al SSO. Un bucle sin ningun error.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlencode

import aiosqlite
from authlib.integrations.starlette_client import OAuth
from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse

from chatbot import auth
from chatbot.config import settings
from chatbot.rutas import RUTA_BASE

logger = logging.getLogger(__name__)

# El rol de realm de Keycloak que concede administracion en Tambora. Los
# demas (tecnico, responsable_proyecto, o ninguno) son usuario estandar.
# Tambora solo distingue dos niveles: users.role admite 'admin' o 'sge'.
_ROL_ADMIN_KEYCLOAK = "admin"

# La direccion del recibidor de la Plataforma SGE. Vacia -> no se pinta el
# enlace de vuelta, que es lo correcto cuando Tambora corre suelta en un
# portatil. Contrato completo en la seccion "El camino de vuelta" del
# runbook: el enlace va a la izquierda, lejos del boton de salir, y volver
# NO cierra la sesion.
PLATAFORMA_URL = os.environ.get("TAMBORA_PLATAFORMA_URL", "").strip()


@dataclass(frozen=True)
class SSOConfig:
    issuer: str
    client_id: str
    client_secret: str
    redirect_uri: str

    @property
    def metadata_url(self) -> str:
        return f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"


def load_config() -> Optional[SSOConfig]:
    """La configuracion, o None si falta CUALQUIERA de las cuatro.

    Todo o nada: una configuracion a medias falla a mitad del baile OIDC con
    un error que no dice nada util."""
    issuer = os.environ.get("TAMBORA_SSO_ISSUER", "").strip()
    client_id = os.environ.get("TAMBORA_SSO_CLIENT_ID", "").strip()
    client_secret = os.environ.get("TAMBORA_SSO_CLIENT_SECRET", "").strip()
    redirect_uri = os.environ.get("TAMBORA_SSO_REDIRECT_URI", "").strip()
    if not (issuer and client_id and client_secret and redirect_uri):
        return None
    return SSOConfig(issuer, client_id, client_secret, redirect_uri)


_ERROR_ENTRADA = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>No se ha podido entrar</title></head>
<body style="font-family:system-ui;max-width:34rem;margin:4rem auto;line-height:1.6">
<h1>No se ha podido entrar</h1>
<p>Te has identificado correctamente, pero tu cuenta no puede acceder a esta
herramienta ahora mismo.</p>
<p>Si crees que deberia funcionar, avisa a quien administra la plataforma.</p>
<p><a href="{RUTA_BASE}/sso/login">Volver a intentarlo</a></p>
</body></html>"""


def rol_desde_claims(claims: dict) -> str:
    """El rol de Tambora a partir de los roles de realm del token.

    Keycloak manda, siempre. Se recalcula en cada entrada, asi que quitarle
    el rol admin en Keycloak se lo quita aqui sin tocar la base a mano."""
    roles = (claims.get("realm_access") or {}).get("roles") or []
    return "admin" if _ROL_ADMIN_KEYCLOAK in roles else "sge"


async def migrar_keycloak_sub(db: aiosqlite.Connection) -> bool:
    """Anade users.keycloak_sub si no esta. Devuelve True si la anadio.

    Idempotente porque corre en cada arranque. Se usa ALTER TABLE y no el
    patron de reconstruir la tabla que hay en chatbot/migrations.py: anadir
    una columna nueva SI lo permite SQLite, y reconstruir la tabla para esto
    seria mover filas sin motivo.

    No lleva UNIQUE en el ALTER: SQLite no acepta anadir una columna con
    restriccion UNIQUE. Se crea un indice unico aparte, que es equivalente y
    ademas es lo que de verdad hace cumplir la unicidad."""
    cur = await db.execute("PRAGMA table_info(users)")
    columnas = {fila["name"] for fila in await cur.fetchall()}
    if "keycloak_sub" in columnas:
        return False

    await db.execute("ALTER TABLE users ADD COLUMN keycloak_sub TEXT")
    await db.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_keycloak_sub "
        "ON users(keycloak_sub) WHERE keycloak_sub IS NOT NULL"
    )
    await db.commit()
    logger.info("[SSO] users.keycloak_sub anadida.")
    return True


async def resolver_identidad(db: aiosqlite.Connection, claims: dict) -> Optional[dict]:
    """Devuelve el usuario de Tambora para esta identidad, creandolo si hace
    falta. None si no puede entrar.

    Tres pasos, y el segundo es el que importa:

    1. Por `keycloak_sub`: es el identificador de verdad. Si la persona se
       renombra en Keycloak sigue siendo ella.
    2. Por `username`: Tambora YA tiene usuarios del login viejo. Sin este
       paso, su primera entrada por SSO crearia un duplicado y **perderian su
       historial**, porque las conversaciones cuelgan de chats.user_id. No da
       ningun error: simplemente entran y no esta.
    3. Crear. Auto-aprovisionamiento a proposito -- al contrario que PALBE,
       que tenia 5 usuarios con datos previos que proteger y aqui no hay nada
       que proteger.
    """
    sub = (claims.get("sub") or "").strip()
    if not sub:
        logger.warning("[SSO] token sin sub; no se entra.")
        return None

    username = (claims.get("preferred_username") or "").strip().lower()
    if not username:
        # username es NOT NULL UNIQUE: insertar "" colisionaria con la
        # siguiente persona en el mismo caso, y el error saldria lejos de
        # aqui.
        logger.warning("[SSO] token sin preferred_username; no se entra.")
        return None

    rol = rol_desde_claims(claims)
    nombre = (claims.get("given_name") or "").strip()
    apellido = (claims.get("family_name") or "").strip()

    cur = await db.execute(
        "SELECT id, username, role, is_active FROM users WHERE keycloak_sub = ?", (sub,)
    )
    fila = await cur.fetchone()

    if fila is None:
        cur = await db.execute(
            "SELECT id, username, role, is_active FROM users WHERE username = ?",
            (username,),
        )
        fila = await cur.fetchone()
        if fila is not None:
            # Paso 2: se vincula, no se duplica.
            await db.execute(
                "UPDATE users SET keycloak_sub = ? WHERE id = ?", (sub, fila["id"])
            )
            logger.info("[SSO] usuario previo '%s' vinculado a su identidad.", username)

    if fila is None:
        # Paso 3: alta. La contrasena local no existe ya como concepto, pero
        # la columna es NOT NULL: se guarda un marcador que ningun hash de
        # bcrypt puede igualar, asi que no hay contrasena que pueda validar.
        ahora = datetime.now(timezone.utc).isoformat()
        cur = await db.execute(
            "INSERT INTO users (username, password_hash, role, created_at, keycloak_sub,"
            " first_name, last_name) VALUES (?,?,?,?,?,?,?)",
            (username, "!sso", rol, ahora, sub, nombre, apellido),
        )
        await db.commit()
        logger.info("[SSO] alta de '%s' con rol %s.", username, rol)
        cur = await db.execute(
            "SELECT id, username, role, is_active FROM users WHERE id = ?",
            (cur.lastrowid,),
        )
        return dict(await cur.fetchone())

    if not fila["is_active"]:
        # Keycloak no puede conceder lo que la base niega. Da igual que alla
        # siga habilitado: aqui esta de baja.
        logger.warning("[SSO] '%s' esta desactivado; no entra.", fila["username"])
        return None

    # Refrescar lo que manda Keycloak: rol y nombre. El username tambien,
    # porque el sub es la identidad y el nombre puede cambiar alli.
    #
    # El correo NO se guarda: users no tiene columna email -- la migracion
    # auth_0001 reconstruyo la tabla sin ella. Keycloak lo tiene y es su
    # sitio; duplicarlo aqui obligaria a otra migracion para nada.
    await db.execute(
        "UPDATE users SET role = ?, username = ?, first_name = ?, last_name = ?"
        " WHERE id = ?",
        (rol, username, nombre, apellido, fila["id"]),
    )
    await db.commit()
    cur = await db.execute(
        "SELECT id, username, role, is_active FROM users WHERE id = ?", (fila["id"],)
    )
    return dict(await cur.fetchone())


def poner_cookie(response, user: dict) -> None:
    """La misma cookie que ponia el login, con una diferencia deliberada.

    El contenido es identico -- `sub` y `role` -- porque
    auth.get_current_user lo lee asi y las ~20 rutas protegidas dependen de
    ello.

    SameSite=**Lax** en vez de Strict: al volver de Keycloak la navegacion es
    entre sitios y el navegador no manda una cookie Strict. Con Strict, la
    persona aterriza en /chat sin sesion y vuelve al SSO: un bucle mudo. Lax
    sigue protegiendo de CSRF en peticiones que no son de navegacion, que es
    lo que importa aqui."""
    token = auth.create_access_token({"sub": str(user["id"]), "role": user["role"]})
    response.set_cookie(
        key=auth.COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_hours * 3600,
        secure=settings.cookie_secure,
    )


def register_sso_routes(app) -> None:
    """Registra /sso/login, /sso/callback y /logout.

    **Revienta si falta configuracion**, y eso es la diferencia de fondo con
    PALBE: sin login propio al que caer, unas rutas /sso/* ausentes dejan la
    herramienta inaccesible sin decir por que. Mejor no arrancar."""
    cfg = load_config()
    if cfg is None:
        raise RuntimeError(
            "Faltan variables TAMBORA_SSO_*: el SSO es la unica forma de entrar "
            "en Tambora, asi que sin ellas la herramienta seria inaccesible. "
            "Se necesitan TAMBORA_SSO_ISSUER, TAMBORA_SSO_CLIENT_ID, "
            "TAMBORA_SSO_CLIENT_SECRET y TAMBORA_SSO_REDIRECT_URI."
        )

    oauth = OAuth()
    oauth.register(
        name="keycloak",
        server_metadata_url=cfg.metadata_url,
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
        client_kwargs={"scope": "openid profile email"},
    )

    @app.get("/sso/login")
    async def sso_login(request: Request):
        return await oauth.keycloak.authorize_redirect(request, cfg.redirect_uri)

    @app.get("/sso/callback")
    async def sso_callback(request: Request):
        try:
            token = await oauth.keycloak.authorize_access_token(request)
        except Exception as exc:
            # Keycloak caido, code caducado, state que no cuadra. Se registra
            # el tipo y el mensaje, NUNCA el token ni el code, y truncado:
            # es texto de una libreria de terceros yendo a un log.
            logger.warning(
                "[SSO] fallo en el callback: %s: %s", type(exc).__name__, str(exc)[:300]
            )
            return HTMLResponse(_ERROR_ENTRADA, status_code=403)

        claims = token.get("userinfo") or {}

        async with aiosqlite.connect(str(settings.db_path)) as db:
            db.row_factory = aiosqlite.Row
            user = await resolver_identidad(db, claims)

        if user is None:
            return HTMLResponse(_ERROR_ENTRADA, status_code=403)

        response = RedirectResponse(f"{RUTA_BASE}/chat", status_code=302)
        poner_cookie(response, user)
        # El id_token se guarda para el cierre de sesion unico: sin el,
        # Keycloak no puede invalidar su propia sesion al salir.
        response.set_cookie(
            key="tambora_id_token",
            value=token.get("id_token", ""),
            httponly=True,
            samesite="lax",
            max_age=settings.jwt_expire_hours * 3600,
            secure=settings.cookie_secure,
        )
        return response

    # GET *y* POST: el panel de administracion cierra sesion con un
    # formulario (chatbot/templates/admin.html), asi que solo con GET ese
    # boton devolveria un 405. La ruta que se retiro aceptaba los dos.
    @app.api_route("/logout", methods=["GET", "POST"])
    async def logout(request: Request):
        """Cierre de sesion UNICO: borrar solo la cookie de Tambora dejaria
        viva la sesion de Keycloak, y volver a entrar pasaria de largo sin
        pedir nada. En un equipo compartido eso sorprende."""
        destino = (
            url_de_cierre(request.cookies.get("tambora_id_token", ""))
            or f"{RUTA_BASE}/sso/login"
        )
        response = RedirectResponse(destino, status_code=302)
        response.delete_cookie(auth.COOKIE_NAME, path="/", samesite="lax")
        response.delete_cookie("tambora_id_token", path="/", samesite="lax")
        return response


def url_de_cierre(id_token: str) -> Optional[str]:
    """La URL de cierre de sesion de Keycloak, o None sin configuracion.

    `client_id` va SIEMPRE: Keycloak exige id_token_hint O client_id para
    validar post_logout_redirect_uri contra las URIs registradas. Sin ninguno
    de los dos devuelve un 400 en crudo en vez de la redireccion -- defecto
    que este proyecto ya cometio dos veces, en PALBE y en el recibidor."""
    cfg = load_config()
    if cfg is None:
        return None
    params = {"client_id": cfg.client_id}
    destino = os.environ.get("TAMBORA_SSO_POST_LOGOUT_URI", "").strip()
    if destino:
        params["post_logout_redirect_uri"] = destino
    if id_token:
        params["id_token_hint"] = id_token
    return (
        f"{cfg.issuer.rstrip('/')}/protocol/openid-connect/logout?{urlencode(params)}"
    )
