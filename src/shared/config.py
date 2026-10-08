"""Ajustes comunes a las dos herramientas.

Aqui SOLO van las variables que de verdad comparten el chatbot y las
auditorias: la clave de la API de Anthropic, la temperatura de muestreo y la
tasa de conversion USD -> EUR de los paneles de coste. Todo lo demas
(rutas de base de datos, modelos, plantillas, limites) es propiedad de
`chatbot/config.py` y `auditorias/config.py`, que no se conocen
entre si.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_env() -> None:
    """Carga `.env` desde el directorio de trabajo. Relativo al CWD a
    proposito (es como funcionaba antes): las dos apps se arrancan desde la
    raiz del repositorio."""
    load_dotenv(dotenv_path=Path(".env"), override=False)


@dataclass(frozen=True)
class SharedSettings:
    anthropic_api_key: str
    temperature: float
    usd_eur_rate: float


def _load() -> SharedSettings:
    load_env()
    return SharedSettings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        temperature=get_float("TEMPERATURE", 0.1),
        usd_eur_rate=get_float("USD_EUR_RATE", 0.92),
    )


settings = _load()
