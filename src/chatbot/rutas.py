"""Dónde vive Tambora dentro de la URL.

Hasta ahora Tambora era dueña de un nombre entero --`tambora.<dominio>`-- y
por tanto de la raíz: `/chat`, `/static/...`, `/api/...`. Todas sus URLs
absolutas daban eso por hecho, que es lo normal cuando lo es.

Deja de serlo cuando las cinco herramientas comparten un solo nombre y se
reparten por ruta: `sge.example.com/tambora`. Entonces `/static/css/chat.css`
apunta un nivel por encima de donde está el fichero, y la página carga sin
estilos, sin JavaScript y con los enlaces rotos -- sin un solo error en el
servidor, porque el servidor nunca llega a ver esas peticiones.

`RUTA_BASE` es ese prefijo. Vacía por defecto: sin la variable puesta, todo
queda exactamente como estaba, que es lo que permite hacer el cambio
herramienta por herramienta en vez de todo a la vez.

Traefik quita el prefijo antes de pasar la petición (`StripPrefix`), así que
las RUTAS QUE SE DEFINEN no lo llevan: `@app.get("/chat")` sigue siendo
`/chat`. Lo lleva todo lo que se EMITE hacia el navegador -- enlaces,
recursos, redirecciones, `fetch` --, porque eso lo resuelve el navegador
contra el nombre completo y ahí el prefijo sí existe.
"""

import os


def _normalizar(valor: str) -> str:
    """`tambora`, `/tambora` y `/tambora/` son la misma intención.

    Se acepta cualquiera de las tres y se devuelve siempre `/tambora`, para
    que concatenar `RUTA_BASE + "/chat"` no produzca nunca `//chat` ni
    `tambora/chat`. Vacío se queda vacío.
    """
    v = (valor or "").strip().strip("/")
    return f"/{v}" if v else ""


RUTA_BASE = _normalizar(os.environ.get("TAMBORA_RUTA_BASE", ""))
