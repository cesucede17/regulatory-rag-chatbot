"""Parameter extraction from user queries using Claude + heuristics."""

import json
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import anthropic

from shared.anthropic_provider import sampling_kwargs, texto_de
from shared.llm_usage import record, usage_from_response
from ..config import settings


class ParameterExtractor:
    """Extract parameters from user queries for BOE APIs."""

    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.claude_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_date(self, query: str) -> Optional[str]:
        """Extract a date from query and return it in YYYYMMDD format."""
        llm_date = self._extract_date_llm(query)
        if llm_date:
            return llm_date
        return self._extract_date_fallback(query)

    def extract_search_params(self, query: str) -> Dict[str, Any]:
        """
        Extract BOE search params from natural language.

        Supported API params: texto, query, materia, from_date, to_date, limit, offset.
        """
        llm_params = self._extract_search_params_llm(query)
        if llm_params:
            return llm_params
        return self._extract_search_params_fallback(query)

    # ------------------------------------------------------------------
    # LLM-based extraction (Claude)
    # ------------------------------------------------------------------

    def _extract_date_llm(self, query: str) -> Optional[str]:
        system_prompt = (
            "Extrae la fecha mencionada y devuélvela estrictamente como YYYYMMDD (8 dígitos). "
            "Sin espacios ni texto extra. Si no hay fecha, responde NO_DATE."
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=30,
                system=system_prompt,
                messages=[{"role": "user", "content": query}],  # type: ignore[arg-type]
                # `temperature=0` explicita: extraer un parametro de una frase
                # tiene que dar siempre lo mismo. Ver `sampling_kwargs`.
                **sampling_kwargs(self.model, temperature=0),
            )
            record(
                "parameter_extraction", usage_from_response(self.model, response.usage)
            )
            content = texto_de(response).strip()
            if not content or content == "NO_DATE":
                return None
            return self._coerce_date(content)
        except Exception as exc:
            print(f"Error extracting date with LLM: {exc}")
            return None

    def _extract_search_params_llm(self, query: str) -> Optional[Dict[str, Any]]:
        """Use Claude to map free-form query into BOE search params."""
        system_prompt = (
            "Convierte la consulta del usuario en un JSON para la API del BOE.\n\n"
            "Devuelve SOLO un objeto JSON con estas claves exactas:\n"
            '{"texto": string|null, "query": string|null, "materia": string|null, '
            '"from_date": string|null, "to_date": string|null, "limit": number, "offset": number}\n\n'
            "Reglas:\n"
            "- Fechas siempre en YYYYMMDD.\n"
            "- Si no hay valor, usa null.\n"
            "- limit entre 1 y 50 (por defecto 10).\n"
            "- offset >= 0 (por defecto 0).\n"
            "- Si la consulta es general, usa 'texto'."
        )
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": query}],  # type: ignore[arg-type]
                # `temperature=0` explicita: extraer un parametro de una frase
                # tiene que dar siempre lo mismo. Ver `sampling_kwargs`.
                **sampling_kwargs(self.model, temperature=0),
            )
            record(
                "parameter_extraction", usage_from_response(self.model, response.usage)
            )
            content = texto_de(response)
            raw = self._extract_json_object(content)
            return self._sanitize_search_params(raw, query)
        except Exception as exc:
            print(f"Error extracting search params with LLM: {exc}")
            return None

    # ------------------------------------------------------------------
    # Fallback heuristic extraction
    # ------------------------------------------------------------------

    def _extract_search_params_fallback(self, query: str) -> Dict[str, Any]:
        normalized = self._normalize_text(query)

        from_date, to_date = self._extract_date_range(query, normalized)
        limit = self._extract_limit(normalized)
        offset = self._extract_offset(normalized)
        materia = self._extract_materia(normalized)
        texto = self._extract_topic(normalized)
        if materia and self._is_generic_topic(texto):
            texto = None

        if not texto:
            texto = None if materia else query.strip()

        return {
            "texto": texto,
            "query": None,
            "materia": materia,
            "from_date": from_date,
            "to_date": to_date,
            "limit": limit,
            "offset": offset,
        }

    def _sanitize_search_params(
        self,
        raw: Any,
        original_query: str,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None

        from .boe_search import BOESearchClient

        texto_raw = self._coerce_text(raw.get("texto")) or self._coerce_text(
            raw.get("query")
        )
        materia_raw = self._coerce_text(raw.get("materia"))
        limit = self._coerce_int(
            raw.get("limit"), default=10, min_value=1, max_value=50
        )
        offset = self._coerce_int(
            raw.get("offset"), default=0, min_value=0, max_value=100000
        )

        # Use original query as fallback, then sanitize for the BOE API
        if not texto_raw:
            texto_raw = original_query
        texto = BOESearchClient.sanitize_texto(texto_raw) or None
        materia = BOESearchClient.sanitize_texto(materia_raw) if materia_raw else None

        return {
            "texto": texto,
            "materia": materia,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_json_object(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    _MESES_ES = {
        "enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "septiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12,
    }
    _RE_FECHA_ES = re.compile(
        r"(\d{1,2})\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto"
        r"|septiembre|octubre|noviembre|diciembre)\s+de\s+(\d{4})",
        re.IGNORECASE,
    )

    def _extract_date_fallback(self, query: str) -> Optional[str]:
        normalized = self._normalize_text(query)
        today = datetime.now()

        if "hoy" in normalized:
            return today.strftime("%Y%m%d")
        if "ayer" in normalized:
            return (today - timedelta(days=1)).strftime("%Y%m%d")

        # Fechas en español natural: "15 de marzo de 2024"
        m = self._RE_FECHA_ES.search(query)
        if m:
            day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
            month = self._MESES_ES.get(self._normalize_text(month_str))
            if month:
                try:
                    return datetime(year, month, day).strftime("%Y%m%d")
                except ValueError:
                    pass

        matches = re.findall(
            r"\d{4}-\d{2}-\d{2}|\d{8}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4}",
            query,
        )
        for item in matches:
            parsed = self._coerce_date(item)
            if parsed:
                return parsed

        return None

    def _extract_date_range(
        self,
        query: str,
        normalized: str,
    ) -> tuple[Optional[str], Optional[str]]:
        pair_patterns = [
            r"entre\s+([0-9/\-]{8,10})\s+y\s+([0-9/\-]{8,10})",
            r"desde\s+([0-9/\-]{8,10})\s+hasta\s+([0-9/\-]{8,10})",
            r"del\s+([0-9/\-]{8,10})\s+al\s+([0-9/\-]{8,10})",
        ]

        for pattern in pair_patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            first = self._coerce_date(match.group(1))
            second = self._coerce_date(match.group(2))
            if first and second:
                return first, second

        today = datetime.now()
        if "hoy" in normalized:
            value = today.strftime("%Y%m%d")
            return value, value
        if "ayer" in normalized:
            value = (today - timedelta(days=1)).strftime("%Y%m%d")
            return value, value
        if "ultima semana" in normalized or "ultimos 7 dias" in normalized:
            return (today - timedelta(days=7)).strftime("%Y%m%d"), today.strftime(
                "%Y%m%d"
            )
        if "ultimo mes" in normalized or "ultimos 30 dias" in normalized:
            return (today - timedelta(days=30)).strftime("%Y%m%d"), today.strftime(
                "%Y%m%d"
            )

        one = self._extract_date_fallback(query)
        if one:
            return one, one

        return None, None

    def _extract_limit(self, normalized: str) -> int:
        patterns = [
            r"(?:top|primeros?|ultimos?|maximo|hasta)\s+(\d{1,3})",
            r"limit\s*(\d{1,3})",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return self._coerce_int(
                    match.group(1), default=10, min_value=1, max_value=50
                )
        return 10

    def _extract_offset(self, normalized: str) -> int:
        patterns = [
            r"offset\s*(\d+)",
            r"desde\s+resultado\s+(\d+)",
            r"a\s+partir\s+de\s+(\d+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                return self._coerce_int(
                    match.group(1), default=0, min_value=0, max_value=100000
                )
        return 0

    def _extract_materia(self, normalized: str) -> Optional[str]:
        match = re.search(
            r"(?:materia|tema|ambito)\s*(?:de|sobre)?\s+([a-z0-9\s\-_,.]+)",
            normalized,
        )
        if not match:
            return None
        value = match.group(1)
        value = re.split(
            r"\b(desde|hasta|entre|top|limit|offset|maximo|ultimo|ultimos|hoy|ayer)\b",
            value,
        )[0]
        return value.strip(" ,.;") or None

    def _extract_topic(self, normalized: str) -> str:
        patterns = [
            r"sobre\s+([a-z0-9\s\-_,.]+)",
            r"acerca\s+de\s+([a-z0-9\s\-_,.]+)",
            r"relacionad[ao]s?\s+con\s+([a-z0-9\s\-_,.]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                value = re.split(
                    r"\b(desde|hasta|entre|top|limit|offset|maximo|materia|tema)\b",
                    match.group(1),
                )[0].strip(" ,.;")
                if value:
                    return value

        cleaned = re.sub(
            r"\b(desde|hasta|entre|top|limit|offset|maximo|materia|tema)\b.*$",
            "",
            normalized,
        ).strip(" ,.;")
        return cleaned

    def _is_generic_topic(self, topic: Optional[str]) -> bool:
        if not topic:
            return True
        compact = topic.strip().lower()
        if not compact:
            return True
        generic = {
            "busca",
            "encuentra",
            "dame",
            "normas",
            "normativa",
            "legislacion",
            "dame legislacion",
            "dame legislacion por",
            "que normativa hay",
            "que legislacion hay",
        }
        return compact in generic

    def _coerce_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _coerce_int(
        self, value: Any, default: int, min_value: int, max_value: int
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, parsed))

    def _coerce_date(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        normalized = self._normalize_text(text)
        today = datetime.now()
        if normalized == "hoy":
            return today.strftime("%Y%m%d")
        if normalized == "ayer":
            return (today - timedelta(days=1)).strftime("%Y%m%d")
        for date_format in ("%Y%m%d", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text, date_format).strftime("%Y%m%d")
            except ValueError:
                continue
        return None

    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return normalized.encode("ascii", "ignore").decode("ascii")
