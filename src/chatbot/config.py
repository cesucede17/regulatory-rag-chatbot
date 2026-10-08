"""Configuracion del Chatbot BOE.

Solo contiene lo que esta herramienta necesita. Las auditorias tienen su
propio `auditorias/config.py` y no comparten este objeto: lo unico
comun (clave de Anthropic, temperatura, tasa USD->EUR) vive en
`shared/config.py`.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from shared.config import get_float, get_int, load_env
from shared.config import settings as shared_settings


def _optional_path(name: str, default: str) -> Optional[Path]:
    """Ruta opcional: la variable de entorno vacia ("") la desactiva
    explicitamente (devuelve None); si no esta definida, usa `default`."""
    value = os.getenv(name)
    if value is not None and value.strip() == "":
        return None
    return Path(value) if value else Path(default)


@dataclass(frozen=True)
class Settings:
    # Anthropic / Claude
    anthropic_api_key: str
    claude_model: str
    temperature: float

    # Embeddings (local, via sentence-transformers)
    embedding_model: str

    # API del BOE
    boe_sumario_api_url: str
    boe_legislacion_api_url: str

    # Procesado de documentos / RAG
    documents_folder: Path
    vector_db_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    # Distancia coseno: los fragmentos con distancia >= este valor se descartan
    similarity_threshold: float
    embedding_cache_ttl: float
    context_max_tokens: int
    history_max_messages: int

    # Auth / multiusuario
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_expire_hours: int
    # True solo si se sirve por HTTPS; False para HTTP (LAN / desarrollo)
    cookie_secure: bool
    db_path: Path
    daily_query_limit: int
    concurrent_query_limit: int
    chat_retention_days: int

    # Tasa de conversion USD -> EUR para el panel de coste del admin
    usd_eur_rate: float


def _load() -> Settings:
    load_env()
    return Settings(
        anthropic_api_key=shared_settings.anthropic_api_key,
        claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
        temperature=shared_settings.temperature,
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        boe_sumario_api_url=os.getenv(
            "BOE_SUMARIO_API_URL",
            "https://www.boe.es/datosabiertos/api/boe/sumario/",
        ),
        boe_legislacion_api_url=os.getenv(
            "BOE_LEGISLACION_API_URL",
            "https://www.boe.es/datosabiertos/api/legislacion-consolidada",
        ),
        documents_folder=Path(os.getenv("DOCUMENTS_FOLDER", "./data/documents")),
        vector_db_path=Path(os.getenv("VECTOR_DB_PATH", "./vectordb")),
        chunk_size=get_int("CHUNK_SIZE", 1800),
        chunk_overlap=get_int("CHUNK_OVERLAP", 360),
        top_k_results=get_int("TOP_K_RESULTS", 5),
        similarity_threshold=get_float("SIMILARITY_THRESHOLD", 0.65),
        embedding_cache_ttl=get_float("EMBEDDING_CACHE_TTL", 300.0),
        context_max_tokens=get_int("CONTEXT_MAX_TOKENS", 1200),
        history_max_messages=get_int("HISTORY_MAX_MESSAGES", 8),
        jwt_secret_key=os.getenv(
            "JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"
        ),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        jwt_expire_hours=get_int("JWT_EXPIRE_HOURS", 8),
        cookie_secure=os.getenv("COOKIE_SECURE", "false").lower() == "true",
        db_path=Path(os.getenv("CHATBOT_DB_PATH", "data/chatbot.db")),
        daily_query_limit=get_int("DAILY_QUERY_LIMIT", 50),
        concurrent_query_limit=get_int("CONCURRENT_QUERY_LIMIT", 2),
        chat_retention_days=get_int("CHAT_RETENTION_DAYS", 15),
        usd_eur_rate=shared_settings.usd_eur_rate,
    )


settings = _load()
