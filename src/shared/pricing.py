"""Cálculo de coste de llamadas a la API de Anthropic.

USD por millón de tokens. Cache write = 1.25x el precio de input,
cache read = 0.1x el precio de input.
"""

import logging

from shared.config import settings

logger = logging.getLogger(__name__)

PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}


def cost_usd(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Coste en USD de una llamada, a partir de los tokens reportados por la API.

    Si `model` no está en PRICING, no lanza excepción: registra un warning y
    devuelve 0.0 — un modelo nuevo o mal escrito no debe romper la petición
    que lo usa, solo perder la contabilidad de coste de esa llamada.
    """
    prices = PRICING.get(model)
    if prices is None:
        logger.warning("cost_usd: modelo desconocido %r, coste no contabilizado", model)
        return 0.0

    input_price = prices["input"]
    output_price = prices["output"]

    cost = (
        input_tokens * input_price
        + output_tokens * output_price
        + cache_creation_tokens * input_price * 1.25
        + cache_read_tokens * input_price * 0.1
    ) / 1_000_000

    return round(cost, 6)


def to_eur(usd: float) -> float:
    """Convierte un importe en USD a EUR usando settings.usd_eur_rate."""
    return usd * settings.usd_eur_rate
