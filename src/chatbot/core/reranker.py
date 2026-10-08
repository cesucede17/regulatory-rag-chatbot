"""Post-retrieval reranker combining semantic distance and lexical overlap.

Improves over pure cosine-similarity ordering for legal queries where
exact terms like "artículo 35" or "7/2026" must be promoted regardless
of their vector neighbourhood.

No additional dependencies — uses only stdlib.
"""

import re
import unicodedata
from typing import Any, Dict, List, Tuple


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return nfkd.encode("ascii", "ignore").decode("ascii")


def _tokenize(text: str) -> List[str]:
    return [w for w in re.split(r"\W+", _normalize(text)) if len(w) > 1]


def _lexical_score(query_tokens: List[str], text: str) -> float:
    """BM25-lite: fraction of query tokens present in *text*."""
    if not query_tokens:
        return 0.0
    text_set = set(_tokenize(text))
    return sum(1 for t in query_tokens if t in text_set) / len(query_tokens)


class Reranker:
    """
    Rerank retrieved chunks by a combined semantic + lexical score.

    Score formula:
        combined = SEMANTIC_W * (1 - cosine_distance) + LEXICAL_W * lexical_overlap

    Both weights sum to 1.0.  Adjust via subclassing if needed.
    """

    SEMANTIC_W: float = 0.6
    LEXICAL_W: float = 0.4

    def rerank(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[float],
        query: str,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Return *(document, metadata, combined_score)* tuples sorted by score desc.

        Args:
            documents:  Chunk texts from ChromaDB.
            metadatas:  Parallel metadata dicts.
            distances:  Cosine distances (0 = identical, 1 = orthogonal).
            query:      User query to score lexical relevance against.
        """
        query_tokens = _tokenize(query)
        scored: List[Tuple[str, Dict[str, Any], float]] = []

        for doc, meta, dist in zip(documents, metadatas, distances):
            semantic = max(0.0, 1.0 - float(dist))
            lexical = _lexical_score(query_tokens, doc)
            combined = self.SEMANTIC_W * semantic + self.LEXICAL_W * lexical
            scored.append((doc, meta, combined))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored
