"""Cálculo de coste de una llamada a la API de Anthropic (envoltorio sobre
`common.pricing`) y captura de uso aislada por tarea de asyncio.

`LlmUsage`/`usage_from_response` vivían solo en `auditorias/llm/service.py`;
se mueven aquí porque `chatbot/*` (Task 13) también los necesita y ese
módulo tiene explícitamente prohibido importar de `auditorias/*`
(ver docstring de `auditorias/service.py`) — la dirección correcta es que
ambos dependan de `common/`, nunca uno del otro.

`start_capture`/`record`/`collect_capture` existen porque el chatbot no
tiene (a diferencia de auditorias/llm/service.py) un único punto de retorno
por el que pasar un callback `on_usage` explícito: `workflow.process_query`
despacha a 5 ramas distintas que llaman a `LLMHandler.generate_response`/
`ParameterExtractor._extract_*_llm` sin devolver el uso al llamador. Un
`ContextVar` da aislamiento correcto por tarea de asyncio — necesario porque
`auth.get_user_semaphore` es un semáforo POR USUARIO, así que dos usuarios
distintos sí pueden ejecutar `process_query` concurrentemente; una lista
compartida en un objeto singleton (p.ej. una instancia de `LLMHandler`)
mezclaría el consumo de un usuario con el de otro.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from shared.pricing import cost_usd


@dataclass(frozen=True)
class LlmUsage:
    """Snapshot de uso/coste de una llamada, listo para insertarse en una
    tabla de consumo (`audit_llm_usage`, `chat_llm_usage`) sin recalcular
    nada."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    cost_usd: float


def usage_from_response(model: str, api_usage: Any) -> LlmUsage:
    """`api_usage` es el objeto `.usage` de una respuesta de la API de
    Anthropic. Los campos de caché se leen con `getattr(..., 0) or 0` porque
    no todas las respuestas los traen, y el SDK puede devolver `None` en vez
    de 0 cuando no aplican."""
    input_tokens = getattr(api_usage, "input_tokens", 0) or 0
    output_tokens = getattr(api_usage, "output_tokens", 0) or 0
    cache_creation_tokens = getattr(api_usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(api_usage, "cache_read_input_tokens", 0) or 0

    cost = cost_usd(
        model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    return LlmUsage(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=cache_read_tokens,
        cost_usd=cost,
    )


_capture: ContextVar[list[tuple[str, LlmUsage]] | None] = ContextVar(
    "_llm_usage_capture", default=None
)


def start_capture() -> None:
    """Abre una captura nueva y vacía para la tarea de asyncio actual.
    Llamar al principio de cada request que vaya a hacer llamadas al LLM
    cuyo coste se quiera contabilizar."""
    _capture.set([])


def record(purpose: str, usage: LlmUsage) -> None:
    """No-op si no hay una captura abierta en la tarea actual (p.ej. un
    script o test que llama a generate_response() sin pasar por
    start_capture()) — nunca lanza, para no romper al llamador real por un
    detalle de contabilidad."""
    sink = _capture.get()
    if sink is not None:
        sink.append((purpose, usage))


def collect_capture() -> list[tuple[str, LlmUsage]]:
    """Devuelve lo capturado y cierra la captura (siguiente `collect_capture()`
    sin un `start_capture()` de por medio devuelve lista vacía)."""
    sink = _capture.get() or []
    _capture.set(None)
    return sink
