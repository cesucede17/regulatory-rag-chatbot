"""Tests de common/anthropic_provider.py: singletons lazy, reset_clients() y
la lógica de sampling_kwargs (qué modelos rechazan temperature/top_p/top_k)."""

import anthropic
import pytest

from shared import anthropic_provider as ap


@pytest.fixture(autouse=True)
def _reset_singletons():
    ap.reset_clients()
    yield
    ap.reset_clients()


def test_get_sync_client_is_lazy_singleton():
    client_a = ap.get_sync_client()
    client_b = ap.get_sync_client()
    assert client_a is client_b


def test_get_async_client_is_lazy_singleton_and_plain_async_client():
    client_a = ap.get_async_client()
    client_b = ap.get_async_client()
    assert client_a is client_b
    assert isinstance(client_a, anthropic.AsyncAnthropic)


def test_reset_clients_forces_recreation():
    first = ap.get_sync_client()
    ap.reset_clients()
    second = ap.get_sync_client()
    assert first is not second


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "claude-opus-5-20260301",
        "claude-sonnet-5",
        "claude-sonnet-5-20260615",
        "claude-opus-4-7",
        "claude-opus-4-7-20260101",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-fable-5-20261201",
    ],
)
def test_sampling_kwargs_empty_for_models_that_reject_sampling(model):
    assert ap.sampling_kwargs(model) == {}


@pytest.mark.parametrize(
    "model",
    [
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-opus-4-6",
        "claude-opus-4",
        "some-future-model",
    ],
)
def test_sampling_kwargs_includes_temperature_for_other_models(model):
    kwargs = ap.sampling_kwargs(model)
    assert kwargs == {"temperature": ap.settings.temperature}


# ---------------------------------------------------------------------------
# temperature explicita (2026-09-22)
#
# Hay llamadas que piden `temperature=0` a proposito --la extraccion de
# parametros de Tambora, que quiere que la misma frase de siempre lo mismo--
# y encauzarlas con el default las habria subido a 0.1 sin que nadie lo
# pidiera. En los modelos sin muestreo eso NO se puede conservar, porque la
# API ya no acepta el parametro.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-4-6", "claude-haiku-4-5", "some-future-model"],
)
def test_sampling_kwargs_conserva_la_temperature_explicita(model):
    assert ap.sampling_kwargs(model, temperature=0) == {"temperature": 0}


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-5", "claude-opus-5", "claude-opus-4-8"],
)
def test_en_los_modelos_sin_muestreo_la_explicita_tampoco_se_manda(model):
    """Y esto es lo que hay que saber antes de cambiar de modelo: quien
    dependa de `temperature=0` para ser determinista deja de tener ese grado
    de libertad, no lo pierde en silencio a otro valor."""
    assert ap.sampling_kwargs(model, temperature=0) == {}


def test_sin_temperature_explicita_manda_la_de_settings():
    """El contraste: no haber pasado nada no es lo mismo que haber pasado 0."""
    assert ap.sampling_kwargs("claude-sonnet-4-6") == {
        "temperature": ap.settings.temperature
    }


# ---------------------------------------------------------------------------
# texto_de: el primer bloque de TEXTO, no el primero a secas (2026-09-22)
# ---------------------------------------------------------------------------


class _Bloque:
    def __init__(self, tipo, texto=None):
        self.type = tipo
        if texto is not None:
            self.text = texto


class _Respuesta:
    def __init__(self, bloques):
        self.content = bloques


def test_texto_de_salta_el_bloque_de_pensamiento():
    """El fallo exacto del servidor: desde Sonnet 5 el razonamiento va
    activado por defecto y `content[0]` es un ThinkingBlock, que **no tiene
    atributo `text`** -- reventaba con
    `'ThinkingBlock' object has no attribute 'text'`."""
    r = _Respuesta([_Bloque("thinking"), _Bloque("text", "la respuesta")])
    assert ap.texto_de(r) == "la respuesta"


def test_texto_de_con_el_texto_primero_sigue_funcionando():
    r = _Respuesta([_Bloque("text", "directo")])
    assert ap.texto_de(r) == "directo"


def test_texto_de_sin_bloques_de_texto_devuelve_cadena_vacia():
    """Sin texto NO se lanza: quien llama ya tiene su propio respaldo (el
    "No he podido generar una respuesta" del chat)."""
    assert ap.texto_de(_Respuesta([_Bloque("thinking")])) == ""
    assert ap.texto_de(_Respuesta([])) == ""
    assert ap.texto_de(_Respuesta(None)) == ""


def test_texto_de_ignora_los_bloques_de_herramienta():
    r = _Respuesta([_Bloque("thinking"), _Bloque("tool_use"), _Bloque("text", "buena")])
    assert ap.texto_de(r) == "buena"
