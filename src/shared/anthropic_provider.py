"""Cliente Anthropic compartido entre el chatbot y el módulo de auditorías.

Centraliza la creación (lazy, singleton) de los clientes sync/async y el
envoltorio opcional de LangSmith, que hasta ahora vivía duplicado dentro de
`chatbot.llm_handler`.
"""

import os

import anthropic

from shared.config import settings

# Singletons lazy — se crean la primera vez que se llama a get_sync_client()/
# get_async_client() y se cachean aquí. reset_clients() los fuerza a None de
# nuevo para que los tests puedan recrearlos limpios entre casos.
_sync_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None

# Modelos que rechazan temperature/top_p/top_k con un 400. Se comprueba por
# prefijo (no por lista cerrada de nombres exactos) para que una variante con
# sufijo de fecha (p.ej. "claude-sonnet-5-20260301") siga funcionando.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
)


def _make_anthropic_client() -> anthropic.Anthropic:
    """Crea el cliente sync, envuelto con LangSmith si el tracing está activo.

    Movido tal cual desde `chatbot.llm_handler._make_anthropic_client`.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    if os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        try:
            from langsmith.wrappers import wrap_anthropic

            client = wrap_anthropic(client)
            print("[LANGSMITH] Tracing activado")
        except ImportError:
            print("[LANGSMITH] langsmith no instalado — tracing desactivado")
    return client


def get_sync_client() -> anthropic.Anthropic:
    """Singleton lazy del cliente sync (con envoltorio LangSmith si aplica)."""
    global _sync_client
    if _sync_client is None:
        _sync_client = _make_anthropic_client()
    return _sync_client


def get_async_client() -> anthropic.AsyncAnthropic:
    """Singleton lazy del cliente async. Sin envoltorio LangSmith: mismo
    comportamiento que tenía hoy `LLMHandler.__init__`, que lo crea inline sin
    pasar por `_make_anthropic_client`."""
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _async_client


def reset_clients() -> None:
    """Fuerza a recrear ambos singletons en la siguiente llamada a
    get_sync_client()/get_async_client(). Pensado para tests."""
    global _sync_client, _async_client
    _sync_client = None
    _async_client = None


def sampling_kwargs(model: str, temperature: float | None = None) -> dict:
    """kwargs de muestreo a pasar a `messages.create(**sampling_kwargs(model))`.

    Los modelos en `_NO_SAMPLING_PREFIXES` rechazan temperature/top_p/top_k
    con un 400 → devuelve {}. Para el resto devuelve
    {"temperature": ...}.

    `temperature` explícita, para quien NO quiere la de `settings`
    (2026-09-22): hay llamadas que piden `temperature=0` a propósito —la
    extracción de parámetros de Tambora, que quiere que la misma frase dé
    siempre lo mismo— y encauzarlas con el default las habría subido a 0.1 sin
    que nadie lo pidiera. Con este parámetro el intento se conserva:
    `sampling_kwargs(model, temperature=0)`.

    **En los modelos sin muestreo eso no se puede conservar**, porque la API
    ya no acepta el parámetro: ahí no hay grado de libertad que ajustar. Quien
    dependa de `temperature=0` para ser determinista tiene que saberlo antes
    de cambiar de modelo, no descubrirlo después.
    """
    if model.startswith(_NO_SAMPLING_PREFIXES):
        return {}
    return {"temperature": settings.temperature if temperature is None else temperature}


def texto_de(response) -> str:
    """El primer bloque de TEXTO de una respuesta, o cadena vacia.

    Y no `response.content[0].text`, que es como estaba escrito en las cuatro
    llamadas de Tambora hasta el 2026-09-22. Desde Sonnet 5 el razonamiento va
    **activado por defecto**, asi que `content[0]` puede ser un bloque de
    pensamiento y el acceso revienta con

        'ThinkingBlock' object has no attribute 'text'

    Medido en el servidor el mismo dia que se subio Tambora de modelo: la
    consulta llegaba a buscar en el BOE correctamente y se rompia justo al
    leer la respuesta. Bartolo no lo sufrio porque o transmite en flujo --y
    `text_stream` ya filtra-- o itera el contenido buscando bloques de
    herramienta.

    Se busca por `type == "text"` en vez de por posicion: el orden de los
    bloques es cosa del modelo, y asumirlo es justo el error que esto arregla.
    """
    for bloque in getattr(response, "content", None) or []:
        if getattr(bloque, "type", None) == "text":
            return getattr(bloque, "text", "") or ""
    return ""
