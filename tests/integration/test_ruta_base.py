"""Tambora colgada de una ruta, y no de un nombre propio.

Cuando las cinco herramientas comparten un solo nombre --`sge.example.com`-- y
se reparten por ruta, Tambora deja de ser dueña de la raíz. Entonces un
`href="/static/css/chat.css"` apunta un nivel por encima de donde está el
fichero: la página sale sin estilos, sin JavaScript y con los enlaces rotos.

Y lo hace **en silencio**. El servidor nunca ve esas peticiones, porque el
navegador las manda a `/static/...` y ahí no hay nada de Tambora; no hay un
error en los logs, no hay un test rojo, solo una pantalla mal. Por eso esto
se prueba, y por eso se prueba de dos maneras:

 1. Renderizando con prefijo y comprobando que lo que sale lo lleva.
 2. Y sobre todo: barriendo plantillas y JavaScript en busca de rutas
    absolutas nuevas. Esa es la que protege de verdad, porque el día que
    alguien añada un `href="/admin"` a mano, aquí salta -- y no en
    producción con la página a medias.
"""

import re
from pathlib import Path

import pytest

from chatbot.rutas import _normalizar

AQUI = Path(__file__).resolve().parents[2] / "src" / "chatbot"
PLANTILLAS = sorted((AQUI / "templates").glob("*.html"))
JS = sorted((AQUI / "static").rglob("*.js"))


@pytest.mark.parametrize(
    "dado,esperado",
    [
        ("", ""),
        ("tambora", "/tambora"),
        ("/tambora", "/tambora"),
        ("/tambora/", "/tambora"),
        ("  /tambora/  ", "/tambora"),
        ("/", ""),
    ],
)
def test_el_prefijo_se_normaliza(dado, esperado):
    """Las tres formas de escribirlo son la misma intención, y concatenar
    tiene que dar `/tambora/chat` en los tres casos: ni `//chat` ni
    `tambora/chat`."""
    assert _normalizar(dado) == esperado


# Lo que NO lleva prefijo, y no es un descuido:
#  - `//` es un enlace a otro sitio sin esquema.
#  - `/` a secas dentro de un texto no es una URL.
ABSOLUTA = re.compile(r'(href|src|action)="(/(?!/)[^"]*)"')
FETCH = re.compile(r"""fetch\(\s*['"`]/""")
# Una navegacion a una ruta absoluta se lleva al usuario fuera del prefijo,
# y ahi no hay nada: sale un 404 del portal. Hoy no hay ninguna; esto es
# para que siga siendo verdad.
NAVEGA = re.compile(
    r"""location\.(assign|replace)\(\s*['"`]/|location\.href\s*=\s*['"`]/|window\.open\(\s*['"`]/"""
)


@pytest.mark.parametrize("f", PLANTILLAS, ids=lambda f: f.name)
def test_ninguna_plantilla_da_por_hecha_la_raiz(f):
    texto = f.read_text(encoding="utf-8")
    sueltas = [m.group(2) for m in ABSOLUTA.finditer(texto)]
    assert not sueltas, (
        f"{f.name} tiene rutas absolutas sin prefijo: {sueltas}. "
        'Van con {{ p }} delante: href="{{ p }}/static/...".'
    )
    assert not FETCH.search(texto), (
        f"{f.name} llama a fetch() con una ruta absoluta. Van con P delante: "
        "fetch(P + '/api/...') o fetch(`${P}/api/...`)."
    )


@pytest.mark.parametrize("f", JS, ids=lambda f: f.name)
def test_ningun_script_da_por_hecha_la_raiz(f):
    texto = f.read_text(encoding="utf-8")
    assert not FETCH.search(texto), (
        f"{f.name} llama a fetch() con una ruta absoluta. `P` la deja "
        "base.html en el navegador; aquí se usa tal cual."
    )
    assert not NAVEGA.search(texto), (
        f"{f.name} navega a una ruta absoluta. Va con P delante, o el usuario acaba fuera del prefijo."
    )
    sueltas = [m.group(2) for m in ABSOLUTA.finditer(texto)]
    assert not sueltas, (
        f"{f.name} construye HTML con rutas absolutas: {sueltas}. "
        "Dentro de un literal de plantilla van con ${P} delante."
    )


def test_base_html_deja_P_para_el_javascript():
    """`chat.js` no es una plantilla, así que no puede leer `{{ p }}`. Lo
    recibe de esta línea, que tiene que ir ANTES de cargarlo."""
    base = (AQUI / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'var P = "{{ p }}"' in base
    assert (
        base.index('var P = "{{ p }}"') < base.index("chat.js")
        if "chat.js" in base
        else True
    )


@pytest.mark.parametrize("prefijo", ["", "/tambora"])
def test_lo_renderizado_cuelga_del_prefijo(prefijo):
    """La prueba de verdad: se renderizan las cuatro plantillas con el
    prefijo puesto y se comprueba que TODO lo que sale hacia el navegador
    empieza por él. Con prefijo vacío, esto mismo afirma que no cambia nada
    respecto a como funciona hoy."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(AQUI / "templates")),
        autoescape=select_autoescape(["html", "htm"]),
    )
    env.globals["p"] = prefijo

    for f in PLANTILLAS:
        if f.name == "base.html":
            continue  # es el esqueleto; se rinde a traves de chat.html
        html = env.get_template(f.name).render(
            user={"username": "prueba", "role": "admin", "full_name": "Prueba"},
            asset_v="1",
            plataforma_url="http://portal.invalido/",
        )
        for m in ABSOLUTA.finditer(html):
            url = m.group(2)
            assert url.startswith(prefijo + "/") or prefijo == "", (
                f"{f.name} emite {url}, que no cuelga de {prefijo}"
            )
        if prefijo:
            assert f'var P = "{prefijo}"' in html, (
                f"{f.name} no le pasa el prefijo al JavaScript"
            )


# ── El test que faltaba, y que habria cogido la caida del 2026-09-30 ────────


def test_con_prefijo_puesto_los_recursos_SIGUEN_sirviendose():
    """Lo que ninguno de los de arriba comprobaba: que se puedan PEDIR.

    Todos los anteriores afirman que las URLs se EMITEN con el prefijo. Este
    afirma la otra mitad, que es la que fallo en produccion: que la
    aplicacion siga sirviendo cuando la peticion llega **sin** el prefijo,
    porque Traefik se lo ha quitado (`StripPrefix`).

    El fallo real: se habia puesto `root_path=RUTA_BASE`. root_path significa
    lo contrario de lo que hace falta aqui --"el proxy me pasa la ruta
    ENTERA"-- y con las dos cosas juntas Starlette buscaba
    `/tambora/static/...` mientras solo llegaba `/static/...`. Las paginas
    cargaban con su contenido y sin una sola hoja de estilo, y ni un error en
    los logs.

    Probe las piezas y no el montaje. Esto prueba el montaje.
    """
    import importlib
    import os

    from starlette.testclient import TestClient

    previo = os.environ.get("TAMBORA_RUTA_BASE")
    os.environ["TAMBORA_RUTA_BASE"] = "/tambora"
    try:
        from chatbot import rutas as rutas_mod

        importlib.reload(rutas_mod)
        assert rutas_mod.RUTA_BASE == "/tambora"

        from chatbot import main as main_mod

        importlib.reload(main_mod)

        assert main_mod.app.root_path == "", (
            "la app lleva root_path=%r. Con StripPrefix delante no sirve NADA: "
            "Starlette busca la ruta con prefijo y solo le llega sin el."
            % main_mod.app.root_path
        )

        with TestClient(main_mod.app) as c:
            r = c.get("/static/img/tambora_logo.png")
        assert r.status_code == 200, (
            "con el prefijo puesto, /static/... deja de servirse (%s). Es "
            "exactamente el fallo del 2026-09-30." % r.status_code
        )
        assert r.headers["content-type"].startswith("image/")
    finally:
        if previo is None:
            os.environ.pop("TAMBORA_RUTA_BASE", None)
        else:
            os.environ["TAMBORA_RUTA_BASE"] = previo
        from chatbot import rutas as rutas_mod

        importlib.reload(rutas_mod)
        from chatbot import main as main_mod

        importlib.reload(main_mod)
