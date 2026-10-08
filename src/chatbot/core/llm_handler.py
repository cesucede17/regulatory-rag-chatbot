"""LLM handler using Anthropic Claude.

v1.4.0 improvements:
- Dual-block system prompt: identity (always cached) + format (always cached)
- Optional third block for team context (cached per-team via cache_control)
- Context compression via ContextCompressor before LLM call
- history_max_messages driven by config, not hardcoded
"""

from typing import Dict, List, Optional

import anthropic

from .context_compressor import ContextCompressor
from .context_manager import get_team_context_text
from shared.anthropic_provider import (
    get_async_client,
    get_sync_client,
    sampling_kwargs,
    texto_de,
)
from shared.llm_usage import record, usage_from_response
from ..config import settings

# ── System prompt — Block 1: Identity (immutable, always cached) ──────────────
_IDENTITY_BLOCK = (
    "Eres Tambora, un asistente experto en legislación española, el BOE y eficiencia energética.\n\n"
    "Dispones de DOS fuentes de información. Úsalas según el tipo de pregunta:\n\n"
    "FUENTE A — CONTEXTO BOE (proporcionado en cada consulta):\n"
    "  Obligatoria para: citas de artículos, requisitos legales concretos, fechas de entrada en vigor,\n"
    "  URLs de documentos BOE, rangos normativos y cualquier afirmación sobre qué dice una norma.\n"
    "  Reglas estrictas:\n"
    "  · No inventes artículos, títulos, reales decretos, leyes ni URLs.\n"
    "  · Reproduce títulos y URLs exactamente como aparecen en el contexto, sin modificarlos.\n"
    "  · Si el contexto no cubre un aspecto legal que el usuario pregunta, indícalo:\n"
    "    «El contexto recuperado no incluye información sobre [aspecto].»\n\n"
    "FUENTE B — CONOCIMIENTO TÉCNICO (tu formación):\n"
    "  Permitida para: explicar protocolos técnicos (IPMVP, ISO 50001/50006, UNE), metodologías\n"
    "  de medición y verificación (M&V), conceptos de eficiencia energética, estándares\n"
    "  internacionales y definiciones técnicas generales.\n"
    "  Reglas:\n"
    "  · Marca estas respuestas con: (Fuente: conocimiento técnico)\n"
    "  · NO uses esta fuente para afirmar qué dice o exige una norma española concreta.\n\n"
    "Cuando una pregunta mezcle ambas (ej: «¿el IPMVP es válido según el RD X?»):\n"
    "  1. Explica el concepto técnico usando Fuente B.\n"
    "  2. Cita lo que dice la norma usando Fuente A.\n"
    "  3. Si la norma no lo menciona explícitamente, indícalo con claridad."
)

# ── System prompt — Block 2: Format instructions (stable, cached per session) ─
_FORMAT_BLOCK = (
    "Instrucciones de formato de respuesta:\n"
    "- Usa formato Markdown: listas con viñetas (-), negrita para títulos legales, "
    "cursiva para fechas.\n"
    "- Sé conciso pero completo: incluye todos los resultados relevantes del contexto.\n"
    "- Para cada resultado usa la estructura: **Título** [Rango] — Departamento — URL (fecha).\n"
    "- Si el contexto proviene de PDFs: cita el nombre del documento y la página "
    "entre corchetes [Documento: nombre, p. N].\n"
    "- Si el usuario pide una sección concreta (ej: solo Ministerio de Hacienda), "
    "filtra y muestra solo los resultados de esa sección; si no es posible filtrarlo, "
    "indícalo explícitamente.\n"
    "- Si hay más de 10 resultados, agrupa por departamento o rango normativo.\n"
    "- Para sumarios BOE: muestra primero las Secciones I y II (disposiciones y personal), "
    "luego el resto si el usuario lo solicita.\n"
    "- Atribución de fuentes al final de la respuesta:\n"
    "  · Contenido del BOE: (Fuente: BOE — [título o sumario YYYY-MM-DD])\n"
    "  · Conocimiento técnico general: (Fuente: conocimiento técnico)\n"
    "  · Respuesta mixta: indica cada parte con su fuente correspondiente."
)


class LLMHandler:
    """Handle LLM interactions using Claude."""

    def __init__(self) -> None:
        self.client = get_sync_client()
        self._async_client = get_async_client()
        self.model = settings.claude_model
        self._compressor = ContextCompressor(settings.context_max_tokens)

    def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        active_contexts: Optional[List[str]] = None,
        compress_budget: Optional[int] = None,
    ) -> str:
        """
        Generate a response using Claude with the provided context.

        Args:
            query: User's question
            context: Contextual data (BOE JSON excerpt, PDF chunks, etc.)
            conversation_history: Previous conversation messages (role/content only)
            active_contexts: Optional list of team context keys (e.g. ["sge"])
            compress_budget: Token budget override for the compressor. When set,
                replaces settings.context_max_tokens for this call only. Useful
                for specific-norm queries that need more context than general ones.

        Returns:
            Generated response string
        """
        # Resolve active contexts: explicit param takes precedence, then instance attribute
        resolved = (
            active_contexts
            if active_contexts is not None
            else getattr(self, "_active_contexts", [])
        )

        # Compress context — use per-call budget override if provided
        if compress_budget is not None:
            original_max = self._compressor._max_chars
            self._compressor._max_chars = compress_budget * 4  # 4 chars per token
            context = self._compressor.compress(context, query)
            self._compressor._max_chars = original_max
        else:
            context = self._compressor.compress(context, query)

        user_prompt = f"Pregunta: {query}\n\nCONTEXTO:\n{context}"

        # Build messages: history + current query
        messages: List[Dict[str, str]] = []
        if conversation_history:
            for msg in conversation_history[-settings.history_max_messages :]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Claude requires messages to start with a user turn
        if messages and messages[0]["role"] != "user":
            messages = []

        messages.append({"role": "user", "content": user_prompt})

        # Build dual-block system prompt with optional team context as third block.
        # Block 1 (identity) and Block 2 (format) are immutable → highest cache hit rate.
        # Block 3 (team context) varies per team but caches within 5-min sessions.
        system: List[Dict] = [
            {
                "type": "text",
                "text": _IDENTITY_BLOCK,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": _FORMAT_BLOCK,
                "cache_control": {"type": "ephemeral"},
            },
        ]

        team_text = get_team_context_text(resolved) if resolved else ""
        if team_text:
            system.append(
                {
                    "type": "text",
                    "text": team_text,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=messages,  # type: ignore[arg-type]
                # Encauzado por `sampling_kwargs` (2026-09-22): los modelos
                # sin muestreo --Sonnet 5 entre ellos-- rechazan temperature
                # con un 400, y esto la mandaba a mano.
                **sampling_kwargs(self.model),
            )
            u = response.usage
            record("chat_response", usage_from_response(self.model, u))
            cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
            cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
            total_in = u.input_tokens + cache_read + cache_write
            print(
                f"[TOKENS] input={u.input_tokens} | output={u.output_tokens} | "
                f"cache_read={cache_read} | cache_write={cache_write} | "
                f"total_in={total_in} | "
                f"equipo={'|'.join(resolved) if resolved else '-'}"
            )

            return (
                texto_de(response)
                or "No he podido generar una respuesta para esa consulta."
            )

        except anthropic.AuthenticationError:
            return (
                "⚠️ **API key de Anthropic inválida.** "
                "Comprueba que `ANTHROPIC_API_KEY` en el fichero `.env` es correcta.\n\n"
                "Puedes obtener tu clave en: https://console.anthropic.com/settings/api-keys"
            )
        except anthropic.RateLimitError:
            return (
                "⚠️ **Límite de uso alcanzado.** Se ha superado el límite de peticiones. "
                "Espera unos segundos y vuelve a intentarlo."
            )
        except anthropic.APIStatusError as exc:
            print(f"Anthropic API error: {exc}")
            return (
                f"Error al generar respuesta (código {exc.status_code}): {exc.message}"
            )
        except Exception as exc:
            print(f"Error generating LLM response: {exc}")
            return f"Error al generar respuesta: {exc}"

    def classify_intent(
        self,
        query: str,
        has_documents: bool,
    ) -> str:
        """
        Use Claude to classify query intent.

        Returns one of: 'sumario', 'search', 'pdf', 'hybrid'
        """
        system_prompt = (
            "Clasifica la intención de la consulta para un chatbot sobre el BOE español.\n\n"
            "Responde SOLO con una palabra:\n"
            "- sumario → el usuario quiere saber qué salió publicado en el BOE un día concreto\n"
            "- search → el usuario busca un Real Decreto, Ley Orgánica, Orden Ministerial u otra norma publicada en el BOE\n"
            "- hybrid → cualquier otra consulta (normas ISO, UNE, guías técnicas, auditorías, preguntas generales)\n\n"
            + (
                "IMPORTANTE: Hay documentos PDF cargados (normas ISO, UNE, guías técnicas). "
                "Si la consulta no es claramente sobre el BOE, responde 'hybrid'.\n"
                if has_documents
                else "No hay PDFs cargados. Solo responde 'sumario' o 'search'.\n"
            )
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                system=system_prompt,
                messages=[{"role": "user", "content": query}],  # type: ignore[arg-type]
                # `temperature=0` explicita: clasificar la ruta tiene que dar
                # siempre lo mismo para la misma consulta. Se conserva donde
                # el modelo la acepta; en los que no, no hay grado de
                # libertad que ajustar.
                **sampling_kwargs(self.model, temperature=0),
            )
            result = texto_de(response).strip().lower()
            if result in ("sumario", "search", "pdf", "hybrid"):
                return result
            return "search"

        except Exception as exc:
            print(f"Error classifying intent with LLM: {exc}")
            return "search"
