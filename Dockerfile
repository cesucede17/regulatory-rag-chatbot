# Tambora -- el chatbot del BOE, como herramienta de la Plataforma SGE.
#
# Base: python:3.14-slim, la misma que PALBE y la misma que corre en el
# portatil. No es una eleccion por defecto: se comprobo en uv.lock que
# torch 2.14.0 trae rueda cp314 para manylinux_2_28_x86_64. Sin ella, el
# build moriria compilando torch desde fuente, que en esta VM no acaba.
FROM python:3.14-slim

# libgomp1: torch lo necesita en tiempo de ejecucion (OpenMP). Es el mismo
# paquete que PALBE necesita para lightgbm, y la misma razon. Nada mas:
# httpx y anthropic usan el bundle de certifi, no el almacen del sistema.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Solo los manifiestos primero: asi un cambio de codigo no rehace la capa de
# dependencias, que aqui son 1,3 GB y varios minutos.
COPY pyproject.toml uv.lock ./

# --frozen: instala EXACTAMENTE lo que dice uv.lock, sin resolver por su
# cuenta. Si el lockfile y el pyproject no cuadran, falla aqui y no en el
# servidor con versiones inesperadas.
# --no-dev: pytest, ruff y compania no entran en la imagen.
#
# UV_NO_CACHE: medido el 2026-09-15. Sin esto, uv deja en /root/.cache/uv las
# ruedas que acaba de descargar -- 1,4 GB dentro de la imagen final que no
# sirven para nada, porque el entorno ya esta construido. No se ve en el
# du del venv: hay que mirar / entero.
RUN UV_NO_CACHE=1 uv sync --frozen --no-dev --no-install-project

# El modelo de embeddings, DENTRO de la imagen.
#
# Son 458 MB y es la decision que mas afecta al despliegue. Si se deja para
# el arranque, cada vez que se recrea el contenedor hay que descargarlo otra
# vez: minutos de espera y dependencia de tener salida a internet desde el
# servidor justo en ese momento. Bajarlo en el build lo congela en la capa.
#
# HF_HOME apunta a una ruta fija y legible por el usuario no-root: por
# defecto sentence-transformers escribe en ~/.cache, y ~ cambia al bajar
# privilegios -- el modelo quedaria fuera del alcance de quien lo necesita.
ENV HF_HOME=/opt/modelos \
    HF_HUB_DISABLE_TELEMETRY=1
RUN /app/.venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'); \
print('modelo de embeddings dentro de la imagen')"

# Codigo de la aplicacion (.dockerignore excluye .venv/, data/, vectordb/,
# .env, tests/ y docs/).
COPY . .

# Usuario no-root. Los directorios de datos se crean antes de bajar
# privilegios para que un volumen vacio y propiedad de root no rompa el
# primer arranque.
RUN groupadd -r tambora && useradd -r -g tambora -d /app tambora \
    && mkdir -p /data/db /data/documentos /data/vectordb \
    && chown -R tambora:tambora /app /data /opt/modelos
USER tambora

ENV PATH="/app/.venv/bin:$PATH" \
    CHATBOT_DB_PATH=/data/db/chatbot.db \
    DOCUMENTS_FOLDER=/data/documentos \
    VECTOR_DB_PATH=/data/vectordb

EXPOSE 8501

# /health responde SIN sesion: es ruta publica a proposito. Si dependiera de
# auth.get_current_user, el healthcheck recibiria un 401 y el contenedor
# nunca se marcaria sano -- que es exactamente lo que le paso a PALBE antes
# de meter /health en su lista blanca.
#
# start-period generoso: al arrancar se carga torch y el modelo de
# embeddings, y eso tarda bastante mas que una aplicacion web normal. Medido
# en el portatil: unos 20 segundos hasta responder.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/health', timeout=3)" || exit 1

# Sin --workers: el planificador de APScheduler (verificacion diaria del BOE)
# y los semaforos por usuario de auth.get_user_semaphore viven en memoria del
# proceso. Con dos workers habria dos planificadores comprobando el BOE y dos
# juegos de semaforos que no se ven entre si.
CMD ["uvicorn", "chatbot.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8501"]
