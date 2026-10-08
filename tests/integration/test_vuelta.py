"""
El camino de vuelta al recibidor.

Contrato completo en la seccion «El camino de vuelta» del runbook. Aqui se
protegen sus cuatro reglas: la variable llega al contenedor, el enlace se
pinta solo cuando esta configurada, va a la izquierda y lejos del boton de
salir, y **volver no cierra la sesion**.

Y una quinta cosa, aprendida a base de romperla: la cabecera del chat es un
flex con justify-between, asi que anadir un tercer hijo reparte el espacio
entre los tres y descoloca el migajon. Se cuentan los hijos.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLATAFORMA = "http://sge.local:8080/"
CHAT = ROOT / "src" / "chatbot" / "templates" / "chat.html"


def _render_chat(plataforma_url):
    from chatbot.main import jinja_env

    return jinja_env.get_template("chat.html").render(
        api_key_ok=True,
        user={"username": "ilasierra", "role": "sge"},
        asset_v="test",
        plataforma_url=plataforma_url,
    )


def test_el_compose_pasa_la_url_al_contenedor():
    """Sin esta linea el boton no existe y NADA falla al arrancar: la
    cabecera se pinta igual, solo que sin salida. Es la quinta vez que en
    este proyecto una variable se queda fuera de un bloque environment, de
    ahi que se lea el fichero en vez de fiarse."""
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    # Una ASIGNACION, no la aparicion del texto: el fichero lleva comentarios
    # que nombran la variable, y un test que se dispara con su propia
    # documentacion acaba borrandola.
    assert re.search(r"^\s+TAMBORA_PLATAFORMA_URL:\s*\S", compose, re.MULTILINE), (
        "TAMBORA_PLATAFORMA_URL no llega al contenedor: Tambora se quedaria "
        "sin camino de vuelta al recibidor, sin ningun error visible"
    )


def test_con_la_url_configurada_la_cabecera_lleva_el_enlace():
    html = _render_chat(PLATAFORMA)

    assert PLATAFORMA in html
    assert "Plataforma SGE" in html


def test_sin_la_url_no_se_pinta_nada():
    """Tambora suelta en un portatil no ensena un boton que no lleva a
    ningun sitio. Mismo criterio que el SSO."""
    html = _render_chat("")

    assert "Plataforma SGE" not in html


def test_volver_no_es_cerrar_sesion():
    """Los dos caminos coexisten y son distintos. Si alguien "simplifica"
    esto apuntando el enlace a /logout, el boton deja de resolver el
    problema por el que se puso."""
    html = _render_chat(PLATAFORMA)

    enlace = re.search(
        r'<a href="([^"]*)"[^>]*title="Volver a la Plataforma SGE"', html
    )
    assert enlace, "no se encontro el enlace de vuelta"
    assert "/logout" not in enlace.group(1)
    # Y el de salir sigue estando, que es lo otro que se puede querer.
    assert 'href="/logout"' in html


def test_la_cabecera_del_chat_sigue_teniendo_dos_hijos():
    """La cabecera es un flex con justify-between: con dos hijos, uno a cada
    extremo. Un tercero reparte el espacio entre los tres y el migajon se va
    al centro. Por eso el enlace vive DENTRO del grupo de la izquierda.

    Este test cuenta hijos, no pixeles: no comprueba que se vea bien, pero
    caza la causa concreta de que se descoloque."""
    from html.parser import HTMLParser

    html = _render_chat(PLATAFORMA)

    class Hijos(HTMLParser):
        VACIAS = {"img", "br", "hr", "input", "meta", "link", "path", "circle", "svg"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.dentro = False
            self.prof = 0
            self.hijos = []

        def handle_starttag(self, tag, attrs):
            if not self.dentro and tag == "header":
                self.dentro = True
                return
            if self.dentro and tag not in self.VACIAS:
                if self.prof == 0:
                    self.hijos.append(tag)
                self.prof += 1

        def handle_endtag(self, tag):
            if not self.dentro or tag in self.VACIAS:
                return
            if tag == "header" and self.prof == 0:
                self.dentro = False
                return
            self.prof -= 1

    p = Hijos()
    p.feed(html)
    assert len(p.hijos) == 2, (
        f"la cabecera tiene {len(p.hijos)} hijos directos y es un flex con "
        f"justify-between, asi que el contenido se descoloca: {p.hijos}"
    )


@pytest.mark.parametrize("metodo", ["get", "post"])
def test_logout_acepta_get_y_post(metodo):
    """El panel de administracion cierra sesion con un formulario POST
    (chatbot/templates/admin.html). La ruta que se retiro aceptaba los dos
    metodos; la nueva tambien, o ese boton devolveria un 405."""
    from fastapi.testclient import TestClient

    from chatbot.main import app

    with TestClient(app) as cliente:
        r = getattr(cliente, metodo)("/logout", follow_redirects=False)

    assert r.status_code != 405, f"/logout no acepta {metodo.upper()}"
    assert r.status_code in (302, 303, 307)
    # Y borra las dos cookies: la de sesion y la del id_token.
    borradas = " ".join(r.headers.get_list("set-cookie"))
    from chatbot import auth

    assert auth.COOKIE_NAME in borradas
    assert "tambora_id_token" in borradas


def test_el_panel_de_administracion_tambien_tiene_vuelta():
    """Quien entra en /admin tambien quiere salir de ahi sin cerrar sesion."""
    from chatbot.main import jinja_env

    plantilla = (ROOT / "src/chatbot/templates/admin.html").read_text(encoding="utf-8")
    assert "plataforma_url" in plantilla
    assert "Volver a la Plataforma SGE" in plantilla
    # Y que la plantilla al menos compile con la variable puesta.
    jinja_env.get_template("admin.html")


def test_la_cabecera_lleva_la_marca_de_tambora():
    """El icono de su tarjeta del recibidor, en su cabecera.

    La cabecera de Tambora no tenia marca ninguna: a la izquierda solo el
    enlace de vuelta y el migajon de la conversacion. Ahora lleva icono y
    nombre, DENTRO del primer hijo -- la cabecera es un flex con
    justify-between y un tercer hijo descolocaria el contenido.

    Se busca dentro del <header> y no en el fichero entero porque el logo
    aparece tambien en la barra lateral, antes, y un indexOf sobre todo el
    fichero encontraria ese y no este.
    """
    plantilla = CHAT.read_text(encoding="utf-8")
    cabecera = plantilla[plantilla.index("<header") : plantilla.index("</header>")]

    assert "/static/img/tambora_logo.png" in cabecera, (
        "la cabecera no referencia el icono de la herramienta"
    )

    vuelta = cabecera.index("Plataforma SGE")
    icono = cabecera.index("/static/img/tambora_logo.png")
    migajon = cabecera.index('id="header-crumb"')

    assert vuelta < icono < migajon, (
        "el orden es: vuelta al recibidor, marca de la herramienta, migajon. "
        "La vuelta va primero (regla 3 del camino de vuelta)"
    )


def test_el_fichero_del_icono_de_tambora_existe():
    png = ROOT / "src" / "chatbot" / "static" / "img" / "tambora_logo.png"
    assert png.is_file(), f"falta {png}"
    assert png.stat().st_size > 1024


def test_el_favicon_es_el_mismo_icono():
    """La pestana lleva la marca de la herramienta, y es la de su tarjeta.

    Tambora ya lo hacia: su favicon apunta al mismo PNG que su cabecera, asi
    que al sustituir ese fichero por el icono del catalogo cambiaron las dos
    cosas a la vez, sin tocar una linea. Esto lo fija para que siga siendo
    verdad -- si alguien le da un fichero propio al favicon, las dos marcas
    se separan y nadie se entera."""
    base = ROOT / "src" / "chatbot" / "templates" / "base.html"
    contenido = base.read_text(encoding="utf-8")
    assert "/static/img/tambora_logo.png" in contenido
    assert 'rel="icon"' in contenido
