"""RAG retrieval system with embedding cache and post-retrieval reranking."""

import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from .reranker import Reranker
from .vector_store import VectorStore
from ..config import settings


class RAGRetriever:
    """Retrieve relevant document chunks using RAG with reranking."""

    # Distance below which a doc chunk is considered a good match (probe check)
    _PROBE_THRESHOLD = 0.8

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()
        self.top_k = settings.top_k_results
        self.similarity_threshold = settings.similarity_threshold
        self._reranker = Reranker()
        # Simple TTL cache: query_text → (embedding, timestamp)
        self._embed_cache: Dict[str, Tuple[List[float], float]] = {}
        self._cached_filenames: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text.lower())
        return normalized.encode("ascii", "ignore").decode("ascii")

    def _get_filenames(self) -> List[str]:
        if self._cached_filenames is None:
            self._cached_filenames = self.vector_store.get_distinct_filenames()
        return self._cached_filenames

    def invalidate_filename_cache(self) -> None:
        self._cached_filenames = None

    def _get_embedding_cached(self, query: str) -> List[float]:
        """Return a cached embedding or generate and cache a new one (TTL 5 min, max 256)."""
        now = time.monotonic()
        cached = self._embed_cache.get(query)
        if cached is not None:
            embedding, ts = cached
            if now - ts < settings.embedding_cache_ttl:
                return embedding

        embedding = self.vector_store.generate_embedding(query)

        if embedding:
            self._embed_cache[query] = (embedding, now)
            # Evict oldest entry to keep memory bounded
            if len(self._embed_cache) > 256:
                oldest_key = next(iter(self._embed_cache))
                del self._embed_cache[oldest_key]

        return embedding

    # Palabras que NO identifican un documento aunque salgan en su nombre:
    # preposiciones, conjunciones y determinantes de 4+ letras, que es lo que
    # colaba el filtro `len(t) > 3`.
    #
    # El fallo que esto arregla (2026-09-22): «Guia tecnica **para** la
    # evaluacion...» contiene «para», y basta UN token para anclar la consulta
    # entera a ese documento. Con la intencion en `pdf` la busqueda en el BOE
    # NO se ejecuta, asi que «normativa de objetivos minimos de biocombustibles
    # **para** las comercializadoras» se contestaba con «el contexto no
    # incluye esta informacion» -- mirando dentro de un PDF de riesgo
    # electrico.
    _PALABRAS_VACIAS = frozenset(
        {
            "para",
            "como",
            "desde",
            "entre",
            "sobre",
            "contra",
            "hasta",
            "segun",
            "donde",
            "cuando",
            "cual",
            "cuales",
            "cuyo",
            "cuya",
            "esta",
            "este",
            "esto",
            "esos",
            "esas",
            "aquel",
            "otro",
            "otra",
            "otros",
            "otras",
            "todo",
            "toda",
            "todos",
            "todas",
            "mismo",
            "misma",
            "cada",
            "algun",
            "alguna",
            "ningun",
            "pero",
            "aunque",
            "porque",
            "tambien",
            "solo",
            "mas",
            "menos",
            "muy",
            "sino",
            "ademas",
            "pues",
            "ante",
            "bajo",
            "tras",
            "durante",
            "mediante",
            "salvo",
            "excepto",
            # "frente" entra aqui porque en castellano casi siempre es la
            # locucion prepositiva "frente a" ("frente al riesgo"), no contenido.
            "frente",
            # Sufijos de nombre de fichero, no contenido.
            "pdf",
            "docx",
            "xlsx",
            "final",
            "copia",
            "borrador",
            "version",
        }
    )

    # Con dos tokens coincidentes ya no es casualidad. Con uno solo hace falta
    # que sea un IDENTIFICADOR: un numero de cinco cifras o mas --50001,
    # 16247, 20251219--. Los numeros de cuatro cifras quedan fuera a proposito
    # porque casi siempre son anos, y «normativa de 2023» no nombra ningun
    # documento.
    _MIN_TOKENS_PARA_ANCLAR = 2
    _MIN_CIFRAS_IDENTIFICADOR = 5

    def _tokens_significativos(self, filename: str) -> List[str]:
        """Los tokens del nombre que de verdad identifican al documento."""
        tokens = re.split(r"[\s_\-=\.,;/()\[\]]+", self._normalize(filename))
        return [t for t in tokens if len(t) > 3 and t not in self._PALABRAS_VACIAS]

    def _es_identificador(self, token: str) -> bool:
        return token.isdigit() and len(token) >= self._MIN_CIFRAS_IDENTIFICADOR

    def detect_document_filter(self, query: str) -> Optional[Dict[str, str]]:
        """Return a ChromaDB where-filter if query explicitly names an indexed document.

        «Explicitly names» quiere decir: la consulta trae **dos** tokens
        significativos del nombre, o **un identificador** (un codigo numerico
        como ISO 50001 o UNE-EN 16247). Hablar del TEMA de un documento no
        basta: anclar apaga la busqueda en el BOE, asi que en la duda es mejor
        no anclar y dejar que el nivel 4 del flujo decida `hybrid`, que mira
        los documentos Y el BOE.

        Se elige el documento con MAS coincidencias, no el primero de la
        lista: antes el resultado dependia del orden de
        `get_distinct_filenames()`, y «que dice la ISO 50001...» devolvia la
        Directiva porque casaba «energetica».
        """
        q = self._normalize(query)
        mejor: Optional[str] = None
        mejor_n = 0

        for filename in self._get_filenames():
            coincidentes = [
                t
                for t in self._tokens_significativos(filename)
                if re.search(r"\b" + re.escape(t) + r"\b", q)
            ]
            if not coincidentes:
                continue
            if len(coincidentes) < self._MIN_TOKENS_PARA_ANCLAR and not any(
                self._es_identificador(t) for t in coincidentes
            ):
                continue
            if len(coincidentes) > mejor_n:
                mejor, mejor_n = filename, len(coincidentes)

        if mejor is None:
            return None
        print(f"[RAG] Filtro por documento: {mejor} ({mejor_n} coincidencias)")
        return {"filename": mejor}

    def _mejor_documento_semantico(self, query: str) -> Optional[str]:
        """Nombre de fichero del trozo mas cercano a la consulta, o None.

        Es el `probe` mirando el metadato en vez de la distancia: usa la
        misma consulta al indice y el mismo embedding cacheado.
        """
        embedding = self._get_embedding_cached(query)
        if not embedding:
            return None
        try:
            results = self.vector_store.query_by_embedding(embedding, n_results=1)
        except Exception as e:
            print(f"[RAG] Confirmacion semantica: error consultando -- {e}")
            return None
        metadatas = (results.get("metadatas") or [[]])[0]
        if not metadatas:
            return None
        return (metadatas[0] or {}).get("filename")

    def documento_nombrado(self, query: str) -> Optional[Dict[str, str]]:
        """El filtro por documento, confirmado con la busqueda semantica.

        `detect_document_filter` decide con las palabras del NOMBRE del
        fichero, y eso tiene un techo conocido: «eficiencia energetica» esta
        en el nombre de la Directiva, asi que cualquier pregunta del dominio
        se le pega. La lista de palabras de `classify_intent` tapaba los
        casos previstos; lo que no esta en la lista sigue pasando --«que
        ayudas hay para la mejora de la eficiencia energetica»--, y anclar
        apaga la busqueda en el BOE.

        Aqui se pide una segunda firma: el documento que eligieron las
        palabras tiene que ser **ademas** el mas cercano semanticamente. Dos
        senales independientes de acuerdo es una decision; una sola es una
        coincidencia de vocabulario.

        **En la duda, no anclar.** Si no hay respuesta semantica --indice
        vacio, embedding que falla-- no se ancla: la pregunta sigue al nivel
        4, que mira los documentos Y el BOE. No anclar cuesta una busqueda de
        mas; anclar mal deja al usuario sin respuesta.
        """
        candidato = self.detect_document_filter(query)
        if candidato is None:
            # Nada que confirmar, y asi no se paga la consulta al indice en
            # las preguntas que no iban a anclarse de todas formas.
            return None

        mejor = self._mejor_documento_semantico(query)
        if mejor is None:
            print("[RAG] Anclaje descartado: sin confirmacion semantica")
            return None
        if mejor != candidato["filename"]:
            print(
                f"[RAG] Anclaje descartado: las palabras dicen "
                f"{candidato['filename']!r} y lo mas cercano es {mejor!r}"
            )
            return None
        return candidato

    # ------------------------------------------------------------------
    # Probe: lightweight relevance check (no full retrieval)
    # ------------------------------------------------------------------

    def probe(self, query: str) -> float:
        """Return best (lowest) distance for query across all docs. Returns 2.0 if none."""
        if self.is_database_empty():
            return 2.0
        embedding = self._get_embedding_cached(query)
        if not embedding:
            return 2.0
        try:
            results = self.vector_store.query_by_embedding(embedding, n_results=1)
            distances = (results.get("distances") or [[]])[0]
            return float(distances[0]) if distances else 2.0
        except Exception as e:
            print(f"[RAG] Probe error: {e}")
            return 2.0

    def has_relevant_content(self, query: str) -> bool:
        """True if the best matching chunk is closer than the probe threshold."""
        return self.probe(query) < self._PROBE_THRESHOLD

    # ------------------------------------------------------------------
    # Main retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, doc_filter: Optional[Dict[str, str]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant document chunks for a query, then rerank.

        Args:
            query:      User's question (or enriched query).
            doc_filter: Optional pre-computed ChromaDB where-filter.
                        If None, auto-detects from query.

        Returns:
            Tuple of (formatted_context, source_citations)
        """
        db_count = self.vector_store.get_collection_count()
        print(f"[RAG] Documentos indexados en BD: {db_count}")
        if db_count == 0:
            print("[RAG] BD vacía — no hay documentos indexados.")
            return "", []

        query_embedding = self._get_embedding_cached(query)
        if not query_embedding:
            return "", []

        # Auto-detect document filter if not provided. Confirmado: si no lo
        # estuviera, un anclaje descartado en el nivel 3 volveria a colarse
        # aqui dentro, en la mitad `rag` de una consulta `hybrid`.
        if doc_filter is None:
            doc_filter = self.documento_nombrado(query)

        # With a document filter: fetch more candidates, relax distance cutoff
        n_fetch = (self.top_k * 3) if doc_filter else (self.top_k * 2)
        effective_threshold = 2.0 if doc_filter else self.similarity_threshold

        try:
            results = self.vector_store.query_by_embedding(
                embedding=query_embedding,
                n_results=n_fetch,
                where=doc_filter,
            )

            raw_documents = (
                results.get("documents") if isinstance(results, dict) else None
            )
            if not raw_documents or not isinstance(raw_documents, list):
                return "", []

            first_documents = raw_documents[0] if raw_documents else []
            documents: List[str] = (
                [str(doc) for doc in first_documents]
                if isinstance(first_documents, list)
                else []
            )

            raw_metadatas = (
                results.get("metadatas") if isinstance(results, dict) else None
            )
            first_metadatas = (
                raw_metadatas[0]
                if isinstance(raw_metadatas, list) and raw_metadatas
                else []
            )
            metadatas: List[Dict[str, Any]] = (
                [m for m in first_metadatas if isinstance(m, dict)]
                if isinstance(first_metadatas, list)
                else []
            )

            raw_distances = (
                results.get("distances") if isinstance(results, dict) else None
            )
            first_distances = (
                raw_distances[0]
                if isinstance(raw_distances, list) and raw_distances
                else []
            )
            distances: List[float] = (
                [float(d) for d in first_distances]
                if isinstance(first_distances, list)
                else []
            )

            print(
                f"[RAG] Chunks recuperados: {len(documents)}, umbral: {effective_threshold}, filtro: {doc_filter}"
            )

            # Pre-filter by similarity threshold
            pre_filtered = [
                (doc, meta, dist)
                for doc, meta, dist in zip(documents, metadatas, distances)
                if dist < effective_threshold
            ]

            if not pre_filtered:
                print("[RAG] Ningún chunk superó el umbral de similitud.")
                return "", []

            # Rerank: combined semantic (60%) + lexical BM25 (40%)
            docs_f, metas_f, dists_f = zip(*pre_filtered)
            reranked = self._reranker.rerank(
                list(docs_f), list(metas_f), list(dists_f), query
            )

            # Take top_k after reranking
            top_chunks = reranked[: self.top_k]

            context_parts: List[str] = []
            sources: List[Dict[str, Any]] = []
            seen_sources: Set[str] = set()

            for doc, meta, _score in top_chunks:
                filename = meta.get("filename", "Unknown")
                page_raw = meta.get("page", 0)
                # PyPDF2 uses 0-indexed pages; convert to 1-indexed for display and PDF viewer
                page = (page_raw + 1) if isinstance(page_raw, int) else page_raw
                context_parts.append(
                    f"[Documento: {filename}, Página: {page}]\n{doc}\n"
                )
                src_key = f"{filename}_{page}"
                if src_key not in seen_sources:
                    seen_sources.add(src_key)
                    sources.append(
                        {
                            "display": f"{filename} (página {page})",
                            "filename": filename,
                            "page": page,
                            "type": "pdf",
                            "snippet": doc[:200].strip(),
                        }
                    )

            print(
                f"[RAG] Chunks válidos devueltos: {len(context_parts)} (pre-filtro: {len(pre_filtered)})"
            )
            return "\n---\n".join(context_parts), sources

        except Exception as e:
            print(f"[RAG] Error recuperando documentos: {e}")
            return "", []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_database_empty(self) -> bool:
        return self.vector_store.get_collection_count() == 0
