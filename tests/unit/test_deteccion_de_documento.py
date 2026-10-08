"""Que una consulta se ancle a UN documento tiene que ser deliberado.

El fallo, encontrado el 2026-09-22 probando una consulta real: «Dime la
normativa relacionada con los objetivos minimos de biocombustibles para las
comercializadoras de combustibles fosiles» se anclo al PDF de riesgo
electrico, y con la intencion en `pdf` la busqueda en el BOE **no se ejecuta
nunca** -- Tambora contesto «el contexto no incluye esta informacion» a una
pregunta que el BOE responde.

La causa era esta linea de `detect_document_filter`:

    tokens = [t for t in tokens if len(t) > 3]
    if any(t in q for t in tokens):

Bastaba UN token de mas de 3 letras del nombre del fichero. Y ese nombre
--«Guia tecnica **para** la evaluacion...»-- contiene «para», cuatro letras.
Cualquier consulta con «para» quedaba anclada a el.

No era un caso aislado: con los seis documentos indexados hoy los tokens
peligrosos incluyen «para», «riesgo», «frente», «criterios», «directiva»,
«catalogo», «eficiencia», «energetica», «2023» y «2018». Una consulta que
diga «eficiencia energetica» --lo mas comun del dominio-- se anclaba a la
Directiva.
"""

import pytest

from chatbot.core.rag_retriever import RAGRetriever

# Los seis que hay indexados de verdad en el servidor (2026-09-22).
DOCUMENTOS = [
    "20251219 Criterios verificación CAE v5.0.pdf",
    "Directiva_2023_1791 - eficiencia energetica.pdf",
    "Guía técnica para la evaluación y prevención de los riesgos relacionados "
    "con la protección frente al riesgo eléctrico.pdf",
    "ISO 50001-2018.pdf",
    "Resolución_30_07_2025 - actualización catálogo.pdf",
    "UNE-EN_16247-1=2023.pdf",
]


class _VectorStoreDoble:
    def __init__(self, filenames):
        self._filenames = filenames

    def get_distinct_filenames(self):
        return list(self._filenames)


def _retriever(filenames=None):
    r = object.__new__(RAGRetriever)
    r.vector_store = _VectorStoreDoble(
        filenames if filenames is not None else DOCUMENTOS
    )
    r._cached_filenames = None
    return r


# ---------------------------------------------------------------------------
# Lo que NO debe anclarse: preguntas normales que el BOE responde
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "consulta",
    [
        # El caso del usuario, palabra por palabra.
        "Dime la normativa relacionada con los objetivos minimos de biocombustibles "
        "para las comercializadoras de combustibles fosiles",
        # La misma sin «relacionada», que tambien fallaba.
        "Dime la normativa de los objetivos minimos de biocombustibles para las "
        "comercializadoras de combustibles fosiles",
        # «para» a secas, que es el token culpable.
        "que obligaciones hay para las empresas de transporte",
        # Palabras del dominio que aparecen en los nombres de fichero y en casi
        # cualquier pregunta.
        "criterios de calculo de ahorros energeticos",
        "obligaciones de las empresas frente al riesgo de sancion",
    ],
)
def test_una_pregunta_normal_no_se_ancla_a_un_documento(consulta):
    assert _retriever().detect_document_filter(consulta) is None, (
        "la consulta se anclo a un documento y con eso la busqueda en el BOE "
        "no se ejecuta"
    )


# ---------------------------------------------------------------------------
# Lo que SI debe anclarse: cuando la consulta nombra el documento
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "consulta,esperado",
    [
        ("que dice la ISO 50001 sobre la revision energetica", "ISO 50001-2018.pdf"),
        ("resumeme la norma UNE-EN 16247", "UNE-EN_16247-1=2023.pdf"),
        (
            "que exige la Directiva 2023/1791",
            "Directiva_2023_1791 - eficiencia energetica.pdf",
        ),
        (
            "explicame la guia tecnica de riesgo electrico",
            "Guía técnica para la evaluación y prevención de los riesgos relacionados "
            "con la protección frente al riesgo eléctrico.pdf",
        ),
        (
            "criterios de verificacion CAE",
            "20251219 Criterios verificación CAE v5.0.pdf",
        ),
    ],
)
def test_nombrar_el_documento_si_lo_ancla(consulta, esperado):
    f = _retriever().detect_document_filter(consulta)
    assert f is not None, "la consulta nombra el documento y no se detecto"
    assert f["filename"] == esperado, f


def test_se_elige_el_documento_QUE_MAS_COINCIDE_no_el_primero():
    """Antes devolvia el primero de la lista que casara con un solo token, asi
    que el resultado dependia del orden de `get_distinct_filenames()`."""
    f = _retriever().detect_document_filter(
        "comparame la guia tecnica de riesgo electrico con la ISO 50001"
    )
    assert f is not None
    # Cinco tokens de la guia frente a dos de la ISO: gana la guia.
    assert "Guía técnica" in f["filename"], f


def test_sin_documentos_indexados_no_hay_filtro():
    assert _retriever(filenames=[]).detect_document_filter("cualquier cosa") is None


# ---------------------------------------------------------------------------
# El caso que NO se resuelve aqui, y por que
# ---------------------------------------------------------------------------


def test_hablar_del_tema_de_un_documento_si_lo_ancla_y_es_inevitable():
    """«eficiencia energetica» ES el tema del nombre de la Directiva.

    A este nivel no hay forma de distinguir «la directiva de eficiencia
    energetica» (nombrar el documento) de «que dice la normativa sobre
    eficiencia energetica» (hablar del tema): las dos traen las mismas dos
    palabras. Se deja anclado a proposito y se resuelve **un nivel mas
    arriba**, en `workflow.classify_intent`, que manda a BOE cualquier
    pregunta con «normativa» antes de llegar a la deteccion de documento.

    Este test existe para que quede escrito que es una decision y no un
    descuido -- y para que si algun dia se cambia, se cambie sabiendo que la
    otra defensa es la que sostiene el caso del usuario. Ver
    tests/unit/test_workflow_intent.py.
    """
    f = _retriever().detect_document_filter(
        "que dice la normativa sobre eficiencia energetica en la industria"
    )
    assert f is not None and "Directiva" in f["filename"]


# ---------------------------------------------------------------------------
# La confirmacion semantica: nombrar no basta, tiene que ser ademas el mejor
# ---------------------------------------------------------------------------
#
# El anclaje lexico decide con las palabras del NOMBRE del fichero, y eso
# tiene un techo: el test de aqui arriba deja escrito que «que dice la
# normativa sobre eficiencia energetica» se ancla a la Directiva sin remedio.
# La defensa que lo sostenia era una lista de palabras en `classify_intent`,
# y una lista de palabras no cubre lo que no esta en ella: «que ayudas hay
# para la mejora de la eficiencia energetica en la industria» sigue anclada
# hoy, y con la intencion en `pdf` el BOE no se consulta.
#
# `documento_nombrado` anade la segunda vuelta: el documento que las palabras
# eligieron tiene que ser ADEMAS el mas cercano semanticamente a la consulta.
# Si no lo es, no se ancla y la pregunta sigue su camino (hybrid o BOE).


class _VectorStoreConSemantica(_VectorStoreDoble):
    """Doble que ademas responde a la consulta por embedding.

    `mejor` es el nombre de fichero del trozo mas cercano, que es lo unico
    que la confirmacion mira.
    """

    def __init__(self, filenames, mejor, distancia=0.35):
        super().__init__(filenames)
        self._mejor = mejor
        self._distancia = distancia
        self.consultas = 0

    def query_by_embedding(self, embedding, n_results, where=None):
        self.consultas += 1
        if self._mejor is None:
            return {"metadatas": [[]], "distances": [[]], "documents": [[]]}
        return {
            "metadatas": [[{"filename": self._mejor, "page": 1}]],
            "distances": [[self._distancia]],
            "documents": [["texto"]],
        }


def _retriever_con_semantica(mejor, filenames=None, embedding=(0.1, 0.2)):
    r = _retriever(filenames)
    r.vector_store = _VectorStoreConSemantica(
        filenames if filenames is not None else DOCUMENTOS, mejor
    )
    r._cached_filenames = None
    r._get_embedding_cached = lambda q: list(embedding) if embedding else None
    return r


DIRECTIVA = "Directiva_2023_1791 - eficiencia energetica.pdf"
CATALOGO = "Resolución_30_07_2025 - actualización catálogo.pdf"


def test_el_caso_que_hoy_falla_deja_de_anclarse():
    """«que ayudas hay para la mejora de la eficiencia energetica...».

    Las palabras la anclan a la Directiva --«eficiencia» y «energetica» estan
    en su nombre--, pero la pregunta es de ayudas, y lo mas cercano en el
    indice es el catalogo. Al no coincidir, no se ancla: la pregunta llega al
    nivel 4 y de ahi al BOE, que es donde estan las ayudas.
    """
    r = _retriever_con_semantica(mejor=CATALOGO)
    assert (
        r.detect_document_filter(
            "que ayudas hay para la mejora de la eficiencia energetica en la industria"
        )
        is not None
    )
    assert (
        r.documento_nombrado(
            "que ayudas hay para la mejora de la eficiencia energetica en la industria"
        )
        is None
    )


def test_nombrar_un_documento_de_verdad_sigue_anclando():
    """La otra mitad: confirmar no puede romper el caso bueno."""
    r = _retriever_con_semantica(mejor=DIRECTIVA)
    f = r.documento_nombrado("que exige la Directiva 2023/1791")
    assert f == {"filename": DIRECTIVA}


def test_sin_candidato_lexico_no_se_consulta_la_semantica():
    """Si las palabras no eligen documento, no hay nada que confirmar.

    Importa porque la confirmacion cuesta una consulta al indice: no se paga
    en las preguntas que no iban a anclarse de todos modos.
    """
    r = _retriever_con_semantica(mejor=DIRECTIVA)
    assert r.documento_nombrado("sanciones por no presentar la auditoria") is None
    assert r.vector_store.consultas == 0


def test_sin_respuesta_semantica_no_se_ancla():
    """En la duda, no anclar.

    Si el indice no devuelve nada --o el embedding falla-- no hay
    confirmacion posible. Anclar a ciegas es justo el fallo que se esta
    cerrando, asi que se deja pasar la pregunta.
    """
    assert (
        _retriever_con_semantica(mejor=None).documento_nombrado(
            "que exige la Directiva 2023/1791"
        )
        is None
    )
    assert (
        _retriever_con_semantica(mejor=DIRECTIVA, embedding=None).documento_nombrado(
            "que exige la Directiva 2023/1791"
        )
        is None
    )
