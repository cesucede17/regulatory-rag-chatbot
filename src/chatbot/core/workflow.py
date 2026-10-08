"""Workflow orchestration for the chatbot."""

import json
import re as _re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from .boe_search import BOESearchClient
from .date_formatter import format_date
from .http_client import BOESumarioClient
from .llm_handler import LLMHandler
from .parameter_extractor import ParameterExtractor
from .rag_retriever import RAGRetriever
from .vector_store import VectorStore

# Technical/legal terms that indicate a deep query about a specific norm.
# Presence of these in the user query raises the depth score.
_DEPTH_TECH_TERMS = frozenset(
    {
        # M&V / energy efficiency protocols
        "ipmvp",
        "mvp",
        "medicion",
        "verificacion",
        "medicion y verificacion",
        "linea base",
        "baseline",
        "ahorro",
        "calculo",
        "formula",
        # Norm structure references
        "articulo",
        "apartado",
        "disposicion",
        "adicional",
        "transitoria",
        "derogatoria",
        "final",
        "anexo",
        "apendice",
        "parrafo",
        "inciso",
        # Analytical terms
        "requisito",
        "procedimiento",
        "protocolo",
        "metodo",
        "opcion",
        "umbral",
        "parametro",
        "indicador",
        "alcance",
        "ambito",
        "excepcion",
        "salvaguarda",
        "exclusion",
        "definicion",
        # Common regulatory domains
        "cae",
        "ee",
        "eficiencia",
        "obligacion",
        "acreditacion",
        "certificacion",
        "auditoria",
        "conformidad",
        "cumplimiento",
        "compatibilidad",
        "aplicabilidad",
        "validez",
        "vigencia",
    }
)

# Compressor token budgets mapped to depth levels 1–3
_DEPTH_BUDGETS = {1: 1800, 2: 2800, 3: 4000}

# Tier 1 (sumario) de classify_intent: co-ocurrencia de una referencia al BOE
# y una marca temporal, en cualquier orden y sin frase exacta — la lista de
# frases literales (sumario_keywords) no cubría variantes naturales como
# "¿Qué ha publicado el BOE hoy?" (el propio ejemplo del README), que no
# contiene ninguna de esas frases fijas.
_BOE_WORD_RE = _re.compile(r"\b(boe|boletin)\b")
_SUMARIO_TIME_RE = _re.compile(r"\b(hoy|ayer|recient\w*|semana|mes(?:es)?|dias?)\b")


class WorkflowOrchestrator:
    """Orchestrate the complete chatbot workflow."""

    def __init__(self, vector_store: Optional[VectorStore] = None) -> None:
        self.param_extractor = ParameterExtractor()
        self.sumario_client = BOESumarioClient()
        self.search_client = BOESearchClient()
        self.llm_handler = LLMHandler()
        self.rag_retriever = RAGRetriever(vector_store=vector_store)

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> str:
        """
        Classify user query intent.

        Priority:
          1. Sumario keywords          → sumario
          2. BOE-legislation keywords  → search
          3. Explicit document name    → pdf  (RAG-only, filtered to that doc)
          4. RAG probe good match      → hybrid (docs + BOE)
          5. No docs / no match        → search

        Returns: 'sumario', 'search', 'pdf', or 'hybrid'
        """
        q = self._normalize(query)

        # --- Tier 0: Specific norm reference (RD X/Y, RDL X/Y, Ley X/Y…) --
        # Must come first so "RD 36/2023" is always routed to BOE search,
        # even when local PDFs are loaded.
        from .boe_search import _NORM_REF_RE as _NR

        if _NR.search(query):
            return "search"

        # --- Tier 1: BOE daily summary -----------------------------------
        sumario_keywords = [
            "sumario",
            "boe del",
            "boletin del",
            "publicaciones del",
            "ultimo boe",
            "ultimos boe",
            "boe de hoy",
            "boe de ayer",
            "boe mas reciente",
            "ultimo boletin",
            "ultimo sumario",
            "que salio hoy en el boe",
            "que publico el boe",
        ]
        if any(kw in q for kw in sumario_keywords):
            return "sumario"
        if _BOE_WORD_RE.search(q) and _SUMARIO_TIME_RE.search(q):
            return "sumario"

        # --- Tier 2: Clear BOE legislation (Real Decreto, Ley, etc.) -----
        boe_keywords = [
            "real decreto",
            "orden ministerial",
            "ley organica",
            "convenio colectivo",
            "resolucion de",
            "instruccion de",
            "circular de",
            "boe num",
            "boletin oficial",
            # Anadidas el 2026-09-22. Preguntar "que NORMATIVA regula X" es una
            # pregunta de BOE, y sin estas palabras la consulta caia al nivel 3
            # y se anclaba a un documento suelto -- con lo que la busqueda en
            # el BOE no se ejecutaba. Fue asi como «normativa de los objetivos
            # minimos de biocombustibles» acabo contestada desde un PDF de
            # riesgo electrico.
            "normativa",
            "legislacion",
            "reglamento",
            "que norma",
            "que ley",
            "que directiva",
            "marco normativo",
            "marco legal",
            "obligacion legal",
            "obligaciones legales",
            "requisito legal",
            "requisitos legales",
        ]
        if any(kw in q for kw in boe_keywords):
            return "search"

        if self.rag_retriever.is_database_empty():
            return "search"

        # --- Tier 3: Query names a specific indexed document → pdf only --
        # `documento_nombrado`, no `detect_document_filter`: el anclaje solo
        # vale si las palabras Y la semantica eligen el mismo documento.
        # Anclar apaga la busqueda en el BOE, y una coincidencia de
        # vocabulario no es motivo suficiente para apagarla.
        doc_filter = self.rag_retriever.documento_nombrado(query)
        if doc_filter:
            return "pdf"

        # --- Tier 4: RAG probe — do docs have relevant content? ----------
        if self.rag_retriever.has_relevant_content(query):
            return "hybrid"

        return "search"

    # ------------------------------------------------------------------
    # Query processors
    # ------------------------------------------------------------------

    def process_sumario_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, list]:
        """Process BOE sumario query."""
        import re as _re
        from datetime import datetime as _dt, timedelta as _td

        query_norm = query.lower()

        # Detectar consultas de rango temporal
        _MULTI_DAY_PATTERNS = [
            (r"ultimo\s*mes|ultimos\s*30\s*d", 30),
            (r"ultima\s*semana|ultimos\s*7\s*d|esta\s*semana", 7),
            (r"ultimos\s*(\d+)\s*d", None),  # ultimos N dias
        ]
        days_range = None
        for pattern, days in _MULTI_DAY_PATTERNS:
            m = _re.search(pattern, query_norm)
            if m:
                if days is None and m.lastindex:
                    try:
                        days = int(m.group(1))
                    except (IndexError, ValueError):
                        days = 7
                days_range = min(days or 7, 30)  # máximo 30 días
                break

        if days_range:
            # Recoger múltiples sumarios y concatenar contexto
            contexts = []
            sources = []
            today = _dt.now()
            for offset in range(days_range):
                d = (today - _td(days=offset)).strftime("%Y%m%d")
                data = self.sumario_client.fetch_sumario(d)
                if data:
                    ctx = self._extract_sumario_context(data)
                    if ctx.strip():
                        fecha_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                        contexts.append(f"=== BOE {fecha_fmt} ===\n{ctx}")
                        sources.append(
                            {"display": f"BOE Sumario {fecha_fmt}", "type": "boe"}
                        )
            if not contexts:
                return "No he podido recuperar sumarios del BOE para ese período.", []
            combined = "\n\n".join(contexts[:10])  # límite de contexto
            response = self.llm_handler.generate_response(
                query, combined, conversation_history
            )
            return response, sources[:5]

        # Consulta de fecha específica o más reciente
        date = self.param_extractor.extract_date(query)
        boe_data: Optional[Dict[str, Any]] = None
        source_date: Optional[str] = None

        if date:
            formatted_date = format_date(date)
            if len(formatted_date) != 8:
                return "Formato de fecha inválido. Usa YYYY-MM-DD o YYYYMMDD.", []
            boe_data = self.sumario_client.fetch_sumario(formatted_date)
            source_date = formatted_date
            if not boe_data:
                return (
                    f"No he podido recuperar el sumario del BOE para la fecha {formatted_date}. "
                    "Es posible que no haya publicación ese día (fin de semana o festivo).",
                    [],
                )
        else:
            boe_data, source_date = self.sumario_client.fetch_latest_sumario(
                days_back=7
            )
            if not boe_data or not source_date:
                return "No he podido recuperar el sumario más reciente del BOE.", []

        context = self._extract_sumario_context(boe_data)
        response = self.llm_handler.generate_response(
            query, context, conversation_history
        )
        fecha_formateada = (
            f"{source_date[:4]}-{source_date[4:6]}-{source_date[6:]}"
            if source_date and len(source_date) == 8
            else source_date or ""
        )
        return response, [
            {"display": f"BOE Sumario — {fecha_formateada}", "type": "boe"}
        ]

    def process_search_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        display_query: Optional[str] = None,
    ) -> Tuple[str, list]:
        """Process BOE legislación consolidada search query.

        display_query: if provided, this is the query shown to the LLM (original user text).
        query may be enriched with a norm reference for retrieval purposes.
        """
        from .boe_search import _NORM_REF_RE, build_full_ref

        items: List[Dict[str, str]] = []
        is_specific_norm = False

        # Prioridad 0: referencia a norma específica (ej: "RDL 7/2026", "Ley 24/2013")
        norm_match = _NORM_REF_RE.search(query)
        if norm_match:
            is_specific_norm = True
            norm_prefix = norm_match.group(1)  # e.g. "rd", "Real Decreto"
            norm_number = norm_match.group(2)  # e.g. "36/2023"
            # Build "real decreto 36/2023" — avoids false positives like "Orden TMA/36/2023"
            full_ref = build_full_ref(norm_prefix, norm_number)

            # 0a: API de legislación consolidada (normas con texto consolidado reciente)
            raw = self.search_client.search_by_norm_ref(full_ref)
            items = BOESearchClient.extract_items(raw)

            # 0b: escanea sumarios del año de publicación de la norma (cubre normas
            # antiguas no enmendadas que no aparecen en la API consolidada)
            if not items:
                items = self._search_norm_in_sumarios(norm_number, full_ref)

        # Prioridad 1: búsqueda con parámetros extraídos (limit ampliado para cobertura temática)
        if not items:
            params = self.param_extractor.extract_search_params(query)
            raw = self.search_client.search_regulations(
                texto=params.get("texto"),
                materia=params.get("materia"),
                limit=params.get("limit", 20),
                offset=params.get("offset", 0),
            )
            items = BOESearchClient.extract_items(raw)

        # Prioridad 2: búsqueda directa con la query completa
        if not items:
            raw = self.search_client.search_regulations(texto=query, limit=20)
            items = BOESearchClient.extract_items(raw)

        # Para consultas de norma específica: enriquece con el texto completo del
        # documento BOE, independientemente del camino que encontró el ítem.
        if is_specific_norm and items:
            self._enrich_items_with_content(items)

        # Sin resultados: no usar fallback de recientes (generaría contexto irrelevante)
        if not items:
            topic = self._extract_topic_from_query(query)
            return (
                f"No he encontrado normativa específica sobre **{topic}** en la legislación "
                f"consolidada del BOE (búsqueda sobre ~1.000 normas actualizadas recientemente).\n\n"
                f"Puedes intentar:\n"
                f"- Buscar por una norma concreta: por ejemplo *«Real Decreto 661/2007»*\n"
                f"- Consultar publicaciones recientes: *«¿Qué publicó el BOE esta semana sobre {topic}?»*\n"
                f"- Usar un término más específico o un sinónimo relacionado"
            ), []

        context = self._format_search_context(items)
        # For specific-norm queries the budget scales with query depth so that
        # surface questions get a quick response and deep technical questions
        # receive enough article text to answer precisely.
        compress_budget = (
            self._query_depth_budget(display_query or query, conversation_history)
            if is_specific_norm
            else None
        )
        response = self.llm_handler.generate_response(
            display_query if display_query else query,
            context,
            conversation_history,
            compress_budget=compress_budget,
        )
        return response, [{"display": "BOE — Legislación Consolidada", "type": "boe"}]

    @staticmethod
    def _query_depth_budget(
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> int:
        """Return a compressor token budget scaled to the depth of the query.

        Depth 1 (surface) → 1 800 tok  — e.g. "¿qué es el RD 36/2023?"
        Depth 2 (medium)  → 2 800 tok  — e.g. "¿qué métodos contempla?"
        Depth 3 (deep)    → 4 000 tok  — e.g. "¿la Opción A del IPMVP
                                          es válida según el artículo 7 del CAE?"

        Scoring signals (additive):
          - Query length (words)
          - Technical / legal vocabulary hits
          - Explicit article / annex references
          - Analytical phrasing ("es válido", "cumple con"…)
          - Number of prior user turns (conversation depth)
        """
        q = (
            unicodedata.normalize("NFKD", query.lower())
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        words = q.split()
        score = 0

        # Length — longer queries tend to be more specific
        if len(words) >= 20:
            score += 2
        elif len(words) >= 10:
            score += 1

        # Technical vocabulary (capped so a single jargon-heavy query maxes at +3)
        tech_hits = sum(1 for t in _DEPTH_TECH_TERMS if t in q)
        score += min(tech_hits, 3)

        # Explicit structural reference ("artículo 7", "anexo II", "apartado 3"…)
        if _re.search(
            r"\bart[íi]?culo\s+\d+|\banexo\s+[ivx\d]+|\bapartado\s+\d+"
            r"|\bdisposici[óo]n\s+(adicional|transitoria|derogatoria|final)",
            q,
        ):
            score += 2

        # Analytical / verification phrasing
        if _re.search(
            r"es\s+v[áa]lid|cumple\s+con|diferencia\s+(entre|con)|compatib"
            r"|literal|exactamente|seg[úu]n\s+el\s+texto|c[óo]mo\s+se\s+calcula",
            q,
        ):
            score += 1

        # Conversation depth: more user turns → deeper into the topic
        if history:
            user_turns = sum(1 for m in history if m.get("role") == "user")
            if user_turns >= 4:
                score += 2
            elif user_turns >= 2:
                score += 1

        depth = 3 if score >= 6 else (2 if score >= 3 else 1)
        return _DEPTH_BUDGETS[depth]

    def _enrich_with_history_norm(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]],
        lookback: int = 10,
    ) -> str:
        """If query has no norm reference but recent history mentions one, prepend it.

        This handles follow-up questions like "¿es válido en el CAE?" after a
        prior turn about "RD 36/2023" — without this, the routing misses Tier 0
        and the norm is never re-fetched.
        """
        from .boe_search import _NORM_REF_RE

        if _NORM_REF_RE.search(query):
            return query  # already has a norm ref
        if not history:
            return query
        # Scan recent messages (both user and assistant) for the most recent norm reference
        for msg in reversed(history[-lookback:]):
            content = msg.get("content", "")
            m = _NORM_REF_RE.search(content)
            if m:
                return f"{m.group(0)} — {query}"
        return query

    def _enrich_query(
        self, query: str, conversation_history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Prepend last user message to query if it adds context (avoids ambiguous follow-ups)."""
        if not conversation_history:
            return query
        last_user = next(
            (
                m["content"]
                for m in reversed(conversation_history)
                if m.get("role") == "user"
            ),
            None,
        )
        if last_user and last_user.strip().lower() != query.strip().lower():
            return f"{last_user} {query}"
        return query

    def process_pdf_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        doc_filter: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, list]:
        """Process query using local PDF documents."""
        enriched = self._enrich_query(query, conversation_history)
        context, sources = self.rag_retriever.retrieve(enriched, doc_filter=doc_filter)
        if not context:
            return (
                "No he encontrado información relevante en los documentos cargados. "
                "Asegúrate de que los documentos están indexados.",
                [],
            )
        response = self.llm_handler.generate_response(
            query, context, conversation_history
        )
        return response, sources

    def process_hybrid_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, list]:
        """Process query using both BOE API and local PDFs."""
        raw = self.search_client.search_regulations(
            texto=BOESearchClient.sanitize_texto(query), limit=5
        )
        boe_items = BOESearchClient.extract_items(raw)
        enriched = self._enrich_query(query, conversation_history)
        pdf_context, pdf_sources = self.rag_retriever.retrieve(enriched)

        combined_context = ""
        sources: list = []

        if boe_items:
            combined_context += "=== BOE — Legislación Consolidada ===\n"
            combined_context += self._format_search_context(boe_items)
            combined_context += "\n\n"
            sources.append({"display": "BOE — Legislación Consolidada", "type": "boe"})

        if pdf_context:
            combined_context += "=== Documentos PDF Cargados ===\n"
            combined_context += pdf_context
            sources.extend(pdf_sources)

        if not combined_context:
            return "No he encontrado información relevante para tu consulta.", []

        response = self.llm_handler.generate_response(
            query, combined_context, conversation_history
        )
        return response, sources

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, list]:
        """
        Process user query through the complete workflow.

        Args:
            query: User's question
            conversation_history: Previous conversation messages

        Returns:
            Tuple of (response, sources)
        """
        q = self._normalize(query)

        # Tier 0a: explicit norm reference in current query (RD X/Y, Ley X/Y…)
        from .boe_search import _NORM_REF_RE as _NR

        if _NR.search(query):
            return self.process_search_query(query, conversation_history)

        # Tier 1 & 2 fast-path (no RAG calls needed)
        sumario_kw = [
            "sumario",
            "boe del",
            "boletin del",
            "ultimo boe",
            "boe de hoy",
            "boe de ayer",
            "que salio hoy en el boe",
            "que publico el boe",
        ]
        if any(kw in q for kw in sumario_kw):
            return self.process_sumario_query(query, conversation_history)

        boe_kw = [
            "real decreto",
            "orden ministerial",
            "ley organica",
            "convenio colectivo",
            "resolucion de",
            "instruccion de",
            "circular de",
            "boe num",
            "boletin oficial",
        ]
        if any(kw in q for kw in boe_kw):
            return self.process_search_query(query, conversation_history)

        # Tier 0b: follow-up about a norm mentioned earlier in the conversation.
        # Enriches query with the norm ref so Tier 0a logic (search + full-text fetch)
        # applies even when the user omits the norm name in the follow-up message.
        effective_query = self._enrich_with_history_norm(query, conversation_history)
        if effective_query != query:
            return self.process_search_query(
                effective_query, conversation_history, display_query=query
            )

        if self.rag_retriever.is_database_empty():
            return self.process_search_query(query, conversation_history)

        # Tier 3: named document → pdf filtered (confirmado, ver classify_intent)
        doc_filter = self.rag_retriever.documento_nombrado(query)
        if doc_filter:
            return self.process_pdf_query(
                query, conversation_history, doc_filter=doc_filter
            )

        # Tier 4: RAG probe
        if self.rag_retriever.has_relevant_content(query):
            return self.process_hybrid_query(query, conversation_history)

        return self.process_search_query(query, conversation_history)

    # ------------------------------------------------------------------
    # Context extraction helpers
    # ------------------------------------------------------------------

    def _extract_sumario_context(self, data: Dict[str, Any]) -> str:
        """
        Extract a concise, LLM-friendly summary from the BOE sumario JSON.
        Avoids passing the entire raw response (can be hundreds of KB).
        """
        try:
            sumario = data.get("data", {}).get("sumario", data.get("sumario", {}))
            if not sumario:
                # Fallback: return a truncated JSON dump
                return json.dumps(data, ensure_ascii=False)[:8000]

            meta = sumario.get("metadatos", {})
            fecha = meta.get("publicacion", "")
            lines: List[str] = []
            if fecha:
                lines.append(f"BOE del {fecha}\n")

            diario = sumario.get("diario", [])
            if isinstance(diario, dict):
                diario = [diario]

            for entry in diario:
                secciones = entry.get("seccion", [])
                if isinstance(secciones, dict):
                    secciones = [secciones]

                for seccion in secciones:
                    sec_nombre = seccion.get("@nombre", seccion.get("nombre", ""))
                    if sec_nombre:
                        lines.append(f"\n## {sec_nombre}")

                    departamentos = seccion.get("departamento", [])
                    if isinstance(departamentos, dict):
                        departamentos = [departamentos]

                    for dept in departamentos:
                        epigrafes = dept.get("epigrafe", [])
                        if isinstance(epigrafes, dict):
                            epigrafes = [epigrafes]

                        for epigrafe in epigrafes:
                            items = epigrafe.get("item", [])
                            if isinstance(items, dict):
                                items = [items]

                            for item in items:
                                titulo = item.get("titulo", "")
                                url_html = item.get("urlHtml", "")
                                item_id = item.get("@id", item.get("id", ""))
                                line = f"- {titulo}"
                                if url_html:
                                    line += f" — {url_html}"
                                elif item_id:
                                    line += f" ({item_id})"
                                lines.append(line)

            context = "\n".join(lines)
            # Trim to avoid exceeding LLM context
            return context[:12000] if len(context) > 12000 else context

        except Exception as exc:
            print(f"Error extracting sumario context: {exc}")
            raw = json.dumps(data, ensure_ascii=False)
            return raw[:8000]

    @staticmethod
    def _format_search_context(items: List[Dict[str, str]]) -> str:
        """Format legislation search results as a numbered list for easy follow-up."""
        if not items:
            return "Sin resultados."
        lines: List[str] = [f"Se han encontrado {len(items)} normas relacionadas:\n"]
        for i, item in enumerate(items, 1):
            titulo = item.get("titulo", "Sin título")
            url = item.get("url", "")
            fecha = item.get("fecha_actualizacion", "")
            rango = item.get("rango", "")
            dept = item.get("departamento", "")
            contenido = item.get("contenido", "")
            line = f"{i}. **{titulo}**"
            if rango:
                line += f" [{rango}]"
            if dept:
                line += f" — {dept}"
            if url:
                line += f" — {url}"
            if fecha:
                line += f" (actualizado: {fecha[:10]})"
            lines.append(line)
            if contenido:
                lines.append(f"\n{contenido}\n")
        return "\n".join(lines)

    @staticmethod
    def _extract_topic_from_query(query: str) -> str:
        """Extract the main topic from a search query for use in no-results messages."""
        import re as _re

        stopwords = {
            "normativa",
            "legislacion",
            "legislación",
            "norma",
            "normas",
            "busca",
            "dame",
            "muestra",
            "sobre",
            "acerca",
            "de",
            "del",
            "hablame",
            "háblame",
            "dime",
            "que",
            "qué",
            "hay",
            "existe",
            "relacionada",
            "relacionado",
            "con",
            "en",
            "el",
            "la",
            "los",
            "las",
        }
        words = _re.sub(r"[¿?¡!.,;:]", "", query.lower()).split()
        topic_words = [w for w in words if w not in stopwords and len(w) > 2]
        return " ".join(topic_words[:4]) if topic_words else query.strip()

    # ------------------------------------------------------------------
    # Document enrichment
    # ------------------------------------------------------------------

    def _enrich_items_with_content(
        self, items: List[Dict[str, str]], max_docs: int = 3
    ) -> None:
        """
        Fetch the full BOE document text for items that have a BOE-A identifier
        and attach it as ``contenido``.  Mutates items in place.

        The document text is pre-chunked into ~1 400-char segments separated by
        double newlines so that ContextCompressor can score and select the most
        relevant fragments rather than discarding the entire block (which would
        happen if it were one continuous string longer than the token budget).

        Limited to ``max_docs`` fetches to avoid excessive API calls and
        oversized LLM context.  Only called for specific-norm queries.
        """
        fetched = 0
        for item in items:
            if fetched >= max_docs:
                break
            if item.get("contenido"):
                fetched += 1
                continue
            doc_id = item.get("id", "")
            if doc_id.startswith("BOE-"):
                doc_text = self.sumario_client.fetch_document_text(doc_id)
                if doc_text:
                    item["contenido"] = self._chunk_text(doc_text)
                    fetched += 1

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1400) -> str:
        """Split *text* into ~chunk_size-char segments separated by double newlines.

        Splitting at word boundaries ensures the ContextCompressor sees
        individually-sized blocks it can score and fit into its token budget,
        rather than one monolithic block it would drop entirely.
        """
        if len(text) <= chunk_size:
            return text
        chunks: List[str] = []
        while text:
            if len(text) <= chunk_size:
                chunks.append(text)
                break
            split_at = chunk_size
            while split_at > 0 and not text[split_at].isspace():
                split_at -= 1
            if split_at == 0:
                split_at = chunk_size
            chunks.append(text[:split_at].strip())
            text = text[split_at:].strip()
        return "\n\n".join(chunks)

    # ------------------------------------------------------------------
    # Norm-in-sumario helpers
    # ------------------------------------------------------------------

    def _search_norm_in_sumarios(
        self, norm_ref: str, full_norm: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Scan BOE sumarios for the publication year of a norm reference.

        The consolidated legislation API is ordered by update date and only
        covers recently-amended norms.  For older or never-amended norms
        (e.g. "RD 36/2023") we scan the BOE daily sumarios of the relevant
        year instead, which always contains the original publication.

        norm_ref:  just the number, e.g. "36/2023"
        full_norm: full normalised reference, e.g. "real decreto 36/2023".
                   When provided, used as the match string so we don't
                   accidentally pick up "Orden TMA/36/2023".
        """
        from datetime import date, timedelta

        try:
            _num_str, year_str = norm_ref.split("/")
            year = int(year_str)
            number = int(_num_str)
        except (ValueError, AttributeError):
            return []

        days_to_scan = min(120, max(40, number * 2))
        search_norm = (
            self._normalize(full_norm) if full_norm else self._normalize(norm_ref)
        )
        start = date(year, 1, 1)

        for offset in range(days_to_scan):
            d = start + timedelta(days=offset)
            boe_data = self.sumario_client.fetch_sumario(d.strftime("%Y%m%d"))
            if not boe_data:
                continue
            found = self._find_in_sumario(boe_data, search_norm)
            if found:
                return found

        return []

    def _find_in_sumario(
        self, data: Dict[str, Any], norm_norm: str
    ) -> List[Dict[str, str]]:
        """Return items from a sumario whose normalised title contains norm_norm."""
        results: List[Dict[str, str]] = []
        sumario = data.get("data", {}).get("sumario", data.get("sumario", {}))
        if not sumario:
            return results

        diario = sumario.get("diario", [])
        if isinstance(diario, dict):
            diario = [diario]

        for entry in diario:
            secciones = entry.get("seccion", [])
            if isinstance(secciones, dict):
                secciones = [secciones]
            for seccion in secciones:
                departamentos = seccion.get("departamento", [])
                if isinstance(departamentos, dict):
                    departamentos = [departamentos]
                for dept in departamentos:
                    dept_nombre = dept.get("@nombre", dept.get("nombre", ""))
                    epigrafes = dept.get("epigrafe", [])
                    if isinstance(epigrafes, dict):
                        epigrafes = [epigrafes]
                    for epigrafe in epigrafes:
                        ep_nombre = epigrafe.get("@nombre", epigrafe.get("nombre", ""))
                        items_list = epigrafe.get("item", [])
                        if isinstance(items_list, dict):
                            items_list = [items_list]
                        for item in items_list:
                            titulo = item.get("titulo", "")
                            if norm_norm in self._normalize(titulo):
                                entry_id = str(
                                    item.get(
                                        "identificador",
                                        item.get("@id", item.get("id", "")),
                                    )
                                )
                                url_html = item.get("url_html", item.get("urlHtml", ""))
                                results.append(
                                    {
                                        "id": entry_id,
                                        "titulo": titulo,
                                        "url": str(url_html),
                                        "rango": ep_nombre,
                                        "departamento": dept_nombre,
                                    }
                                )
        return results

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return normalized.encode("ascii", "ignore").decode("ascii")
