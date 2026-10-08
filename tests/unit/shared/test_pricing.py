"""Tests de common/pricing.py — la lógica de coste no es trivial (cache
write/read tienen multiplicadores distintos del precio base) y un modelo
desconocido no debe romper la llamada que lo usa."""

import dataclasses
import logging

from shared import pricing


def test_cost_usd_input_and_output_only():
    # claude-sonnet-4-6: input 3.00, output 15.00 USD/millón
    cost = pricing.cost_usd(
        "claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == 18.0


def test_cost_usd_includes_cache_creation_and_read_multipliers():
    # cache_creation = 1.25x precio input, cache_read = 0.1x precio input
    cost = pricing.cost_usd(
        "claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    assert cost == round(3.00 * 1.25 + 3.00 * 0.1, 6)


def test_cost_usd_sonnet_5_es_un_tercio_mas_barato():
    """claude-sonnet-5: input 2.00, output 10.00 USD/millón.

    Estuvo en la tabla a 3.00/15.00, copiado de la fila de claude-sonnet-4-6.
    El id existía, así que no daba coste 0: inflaba el coste un 50% en los
    paneles de gasto. Este test es lo que impide que vuelva."""
    cost = pricing.cost_usd(
        "claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000
    )
    assert cost == 12.0


def test_cost_usd_sonnet_5_cache_multipliers():
    """La aritmética de caché sobre el precio nuevo, no sobre el heredado."""
    cost = pricing.cost_usd(
        "claude-sonnet-5",
        input_tokens=0,
        output_tokens=0,
        cache_creation_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    assert cost == round(2.00 * 1.25 + 2.00 * 0.1, 6)


def test_cost_usd_zero_tokens_is_zero():
    assert pricing.cost_usd("claude-sonnet-4-6", input_tokens=0, output_tokens=0) == 0.0


def test_cost_usd_unknown_model_returns_zero_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="shared.pricing"):
        cost = pricing.cost_usd(
            "modelo-que-no-existe", input_tokens=1000, output_tokens=1000
        )
    assert cost == 0.0
    assert any("modelo-que-no-existe" in record.message for record in caplog.records)


def test_cost_usd_does_not_raise_for_unknown_model():
    # Requisito explícito: un modelo nuevo o mal escrito no debe romper la
    # petición que lo usa.
    pricing.cost_usd("claude-super-nuevo-9000", input_tokens=1, output_tokens=1)


def test_to_eur_uses_settings_rate(monkeypatch):
    # Settings es un dataclass frozen: no se puede mutar in-place, se
    # sustituye la instancia módulo-local por una copia con el rate deseado.
    patched_settings = dataclasses.replace(pricing.settings, usd_eur_rate=0.5)
    monkeypatch.setattr(pricing, "settings", patched_settings)
    assert pricing.to_eur(10.0) == 5.0
