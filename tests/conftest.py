"""Conftest de la suite del Chatbot BOE: pone `src/` en `sys.path` para que
`chatbot.*` y `shared.*` se importen igual que cuando arranca uvicorn desde
aqui."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Desde el 2026-09-15 la aplicacion EXIGE configuracion de SSO para poder
# construirse: chatbot.main llama a tambora_sso.register_sso_routes() al
# importarse y revienta si falta. Es deliberado -- sin login propio, una
# configuracion a medias dejaria la herramienta inaccesible -- y la
# consecuencia es que cualquier test que importe la app necesita estas
# variables. Valores de mentira: aqui no se habla con ningun Keycloak.
os.environ.setdefault("TAMBORA_SSO_ISSUER", "http://auth.ejemplo.invalido/realms/sge")
os.environ.setdefault("TAMBORA_SSO_CLIENT_ID", "tambora")
os.environ.setdefault("TAMBORA_SSO_CLIENT_SECRET", "secreto-de-prueba")
os.environ.setdefault(
    "TAMBORA_SSO_REDIRECT_URI", "http://tambora.ejemplo.invalido/sso/callback"
)
# La ruta de activos solo se registra si hay secreto, y el registro ocurre al
# importar la app: tiene que estar puesta ANTES, como las de SSO.
os.environ.setdefault("TAMBORA_CONTEXTO_TOKEN", "secreto-de-contexto-de-prueba")
