"""Workflow intent classification tests."""

from chatbot.core.workflow import WorkflowOrchestrator
from chatbot.core.rag_retriever import RAGRetriever


class _DummyRetriever:
    def __init__(self, is_empty: bool) -> None:
        self._is_empty = is_empty

    def is_database_empty(self) -> bool:
        return self._is_empty

    def detect_document_filter(self, query: str):
        return None

    def documento_nombrado(self, query: str):
        return None

    def has_relevant_content(self, query: str) -> bool:
        return not self._is_empty


def _make_orchestrator(is_db_empty: bool = True) -> WorkflowOrchestrator:
    orchestrator = object.__new__(WorkflowOrchestrator)
    orchestrator.rag_retriever = _DummyRetriever(is_empty=is_db_empty)
    return orchestrator


def test_classify_intent_latest_boe_goes_to_sumario() -> None:
    orchestrator = _make_orchestrator(is_db_empty=True)
    intent = orchestrator.classify_intent("que dice el ultimo boe sobre energia?")
    assert intent == "sumario"


def test_classify_intent_latest_boe_with_accent_goes_to_sumario() -> None:
    orchestrator = _make_orchestrator(is_db_empty=True)
    intent = orchestrator.classify_intent(
        "que dice el \u00faltimo BOE sobre temas energeticos?"
    )
    assert intent == "sumario"


def test_classify_intent_search_keywords_go_to_search() -> None:
    orchestrator = _make_orchestrator(is_db_empty=True)
    intent = orchestrator.classify_intent("busca normas sobre energia")
    assert intent == "search"


def test_classify_intent_generic_goes_to_hybrid_if_db_has_relevant_content() -> None:
    # Tier 4: no doc-filter match + has_relevant_content=True → "hybrid"
    orchestrator = _make_orchestrator(is_db_empty=False)
    intent = orchestrator.classify_intent("explica este tema")
    assert intent == "hybrid"


def test_rag_retriever_uses_injected_vector_store() -> None:
    class DummyVectorStore:
        pass

    dummy = DummyVectorStore()
    retriever = RAGRetriever(vector_store=dummy)  # type: ignore[arg-type]
    assert retriever.vector_store is dummy


# ---------------------------------------------------------------------------
# Preguntar "que normativa regula X" es una pregunta de BOE (2026-09-22)
#
# El caso que lo motivo: «Dime la normativa relacionada con los objetivos
# minimos de biocombustibles para las comercializadoras de combustibles
# fosiles» caia al nivel 3, se anclaba al PDF de riesgo electrico, y con la
# intencion en `pdf` la busqueda en el BOE NO se ejecuta. Tambora contesto
# «el contexto no incluye esta informacion» a una pregunta que el BOE
# responde.
#
# El nivel 2 corre ANTES de la deteccion de documento, asi que estas palabras
# lo cortan de raiz. La otra defensa --que «para» ya no ancle nada-- esta en
# tests/unit/test_deteccion_de_documento.py.
# ---------------------------------------------------------------------------


class _RetrieverQueAncla:
    """Un retriever que anclaria SIEMPRE: si el nivel 2 hace su trabajo, la
    consulta no llega a preguntarle."""

    def is_database_empty(self) -> bool:
        return False

    def detect_document_filter(self, query: str):
        return {"filename": "cualquier-documento.pdf"}

    def documento_nombrado(self, query: str):
        # Anclaria tambien con la confirmacion semantica: este doble existe
        # para probar que el nivel 2 corta ANTES de preguntar por el anclaje.
        return {"filename": "cualquier-documento.pdf"}

    def has_relevant_content(self, query: str) -> bool:
        return True


def _orquestador_que_ancla() -> WorkflowOrchestrator:
    o = object.__new__(WorkflowOrchestrator)
    o.rag_retriever = _RetrieverQueAncla()
    return o


def test_la_consulta_de_biocombustibles_va_al_boe() -> None:
    """La del usuario, palabra por palabra."""
    o = _orquestador_que_ancla()
    assert (
        o.classify_intent(
            "Dime la normativa relacionada con los objetivos minimos de "
            "biocombustibles para las comercializadoras de combustibles fosiles"
        )
        == "search"
    )


def test_preguntar_por_normativa_o_legislacion_va_al_boe() -> None:
    o = _orquestador_que_ancla()
    for consulta in (
        "que normativa regula los certificados de ahorro energetico",
        "que dice la legislacion sobre auditorias energeticas obligatorias",
        "que reglamento aplica a las instalaciones termicas",
        "que ley obliga a hacer auditorias energeticas",
        "cual es el marco normativo de los CAE",
        "que requisitos legales tiene una empresa de servicios energeticos",
    ):
        assert o.classify_intent(consulta) == "search", consulta


def test_una_pregunta_sin_esas_palabras_sigue_su_camino() -> None:
    """El contraste: las palabras nuevas no se comen todo el enrutado. Sin
    ellas, una consulta que nombra un documento sigue yendo a `pdf`."""
    o = _orquestador_que_ancla()
    assert o.classify_intent("resumeme la ISO 50001") == "pdf"


# ---------------------------------------------------------------------------
# El nivel 3 pregunta por el anclaje CONFIRMADO, no por el lexico
# ---------------------------------------------------------------------------


class _RetrieverQueNoConfirma:
    """Las palabras eligen documento; la semantica dice que no es ese.

    Es el caso de «que ayudas hay para la mejora de la eficiencia
    energetica»: «eficiencia» y «energetica» estan en el nombre de la
    Directiva, pero la pregunta no va de la Directiva.
    """

    def is_database_empty(self) -> bool:
        return False

    def detect_document_filter(self, query: str):
        return {"filename": "Directiva_2023_1791 - eficiencia energetica.pdf"}

    def documento_nombrado(self, query: str):
        return None

    def has_relevant_content(self, query: str) -> bool:
        return True


def test_sin_confirmacion_semantica_el_nivel_3_no_manda_a_pdf() -> None:
    """Si el nivel 3 mirase el anclaje lexico, esto saldria `pdf` y el BOE
    no se consultaria nunca."""
    o = object.__new__(WorkflowOrchestrator)
    o.rag_retriever = _RetrieverQueNoConfirma()
    assert (
        o.classify_intent(
            "que ayudas hay para la mejora de la eficiencia energetica en la industria"
        )
        == "hybrid"
    )
