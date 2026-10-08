"""Parameter extractor heuristic tests."""

from chatbot.core.parameter_extractor import ParameterExtractor


def _build_extractor() -> ParameterExtractor:
    # Avoid creating OpenAI client in unit tests.
    return object.__new__(ParameterExtractor)


def test_fallback_extracts_date_range_topic_and_limit() -> None:
    extractor = _build_extractor()
    params = extractor._extract_search_params_fallback(
        "Busca normas sobre energia entre 2024-01-01 y 2024-01-31 maximo 5"
    )

    assert params["from_date"] == "20240101"
    assert params["to_date"] == "20240131"
    assert params["limit"] == 5
    assert "energia" in str(params["texto"])


def test_fallback_extracts_materia_limit_offset() -> None:
    extractor = _build_extractor()
    params = extractor._extract_search_params_fallback(
        "Normas por materia empleo ultimo mes top 3 offset 20"
    )

    assert params["materia"] is not None
    assert "empleo" in str(params["materia"])
    assert params["limit"] == 3
    assert params["offset"] == 20
    assert params["from_date"] is not None
    assert params["to_date"] is not None


def test_sanitize_defaults_to_original_query_when_text_missing() -> None:
    extractor = _build_extractor()
    params = extractor._sanitize_search_params(
        {
            "texto": None,
            "query": None,
            "materia": None,
            "from_date": None,
            "to_date": None,
            "limit": 10,
            "offset": 0,
        },
        "consulta de prueba",
    )

    assert params is not None
    assert params["texto"] == "consulta de prueba"


def test_coerce_date_supports_common_formats() -> None:
    extractor = _build_extractor()
    assert extractor._coerce_date("2024-02-03") == "20240203"
    assert extractor._coerce_date("03/02/2024") == "20240203"
