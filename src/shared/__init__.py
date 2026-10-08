"""Librería de apoyo del Chatbot BOE: código puro, sin estado de aplicación.

Aquí vive el cliente de Anthropic, la contabilidad de tokens, la tabla de
precios y el runner de migraciones. Nada de rutas HTTP, rutas de base de datos
ni plantillas — eso es propiedad exclusiva del paquete `chatbot/`.

Es una **copia**: el proyecto vecino `Bartolo` tiene la suya, con estos
mismos ficheros más `sse.py`, `docx_assets.py` y `docx_xml.py`, que el chatbot
no usa. Los dos proyectos son independientes a propósito, así que un arreglo
aquí no llega solo al otro: si tocas algo de este directorio, comprueba si su
copia lo necesita también.
"""
