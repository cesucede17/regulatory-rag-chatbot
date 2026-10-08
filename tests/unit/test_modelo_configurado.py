"""El modelo que corre de verdad tiene que estar en la tabla de precios.

Nada ataba las dos cosas, y el hueco es silencioso en los dos sentidos:
`pricing.cost_usd` devuelve 0.0 y un WARNING en el log si no conoce el
modelo -- deliberado, para que un id nuevo no rompa la peticion que lo usa --
asi que una errata en CLAUDE_MODEL, o un sufijo de fecha, no da error: da una
factura que parece gratis. Y al reves, un precio copiado de otra fila (le paso
a claude-sonnet-5, que estuvo a 3.00/15.00 en vez de 2.00/10.00) infla el
coste sin que nadie se entere.

Este test no comprueba que el precio sea correcto -- eso es
tests/shared/test_pricing.py -- solo que el modelo configurado esta en la
tabla.
"""

from chatbot.config import settings
from shared.pricing import PRICING


def test_el_modelo_del_chat_esta_en_la_tabla_de_precios():
    assert settings.claude_model in PRICING, (
        f"CLAUDE_MODEL={settings.claude_model!r} no esta en PRICING: su coste "
        "se apuntaria como 0.0 y solo quedaria un WARNING en el log."
    )


def test_el_modelo_por_defecto_es_sonnet_5():
    """Tambora paso a `claude-sonnet-5` el 2026-09-22, el mismo dia que
    Bartolo. Cuesta 2/10 USD por millon en vez de 3/15.

    No se pudo hacer antes por una razon concreta: sus cuatro llamadas
    mandaban `temperature` a mano, y Sonnet 5 la rechaza con un 400 -- las
    habria tumbado TODAS. Ahora pasan por `sampling_kwargs`.
    """
    assert settings.claude_model == "claude-sonnet-5"


def test_el_modelo_configurado_y_el_muestreo_no_se_contradicen():
    """El fallo que esto vigila seria un 400 en todas las llamadas."""
    from shared.anthropic_provider import _NO_SAMPLING_PREFIXES, sampling_kwargs

    if settings.claude_model.startswith(_NO_SAMPLING_PREFIXES):
        assert sampling_kwargs(settings.claude_model) == {}, (
            "el modelo configurado rechaza temperature pero sampling_kwargs la manda"
        )
        assert sampling_kwargs(settings.claude_model, temperature=0) == {}
    else:
        assert "temperature" in sampling_kwargs(settings.claude_model)
