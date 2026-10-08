"""Context compressor to reduce token usage before LLM calls.

Uses BM25-lite lexical scoring (zero extra dependencies) to keep only
the most query-relevant sections of a large context string.

Typical savings:
  - BOE sumario (raw ~12 000 chars / ~3 000 tokens) → ~4 800 chars / ~1 200 tokens
  - Search results: limits output to a configurable item count
  - PDF chunks: already filtered upstream; compressor just cleans whitespace

Token budget approximation: 1 token ≈ 4 characters (conservative).
"""

import re
import unicodedata
from typing import List

_CHARS_PER_TOKEN: int = 4


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return nfkd.encode("ascii", "ignore").decode("ascii")


def _tokenize(text: str) -> List[str]:
    return [w for w in re.split(r"\W+", _normalize(text)) if len(w) > 2]


def _bm25_lite(query_tokens: List[str], text: str) -> float:
    """
    Simplified BM25: fraction of distinct query tokens present in *text*.
    Returns 0.0 if query_tokens is empty.
    """
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    matches = sum(1 for t in query_tokens if t in text_tokens)
    return matches / len(query_tokens)


class ContextCompressor:
    """
    Compress a context string to stay within a token budget.

    The compressor splits context into logical blocks, scores each block
    by relevance to the query, and returns the highest-scoring blocks up
    to *max_tokens*.  Section headers (lines starting with ``##``) are
    always kept for navigational structure but don't count heavily against
    the budget.
    """

    def __init__(self, max_tokens: int = 1200) -> None:
        self._max_chars = max_tokens * _CHARS_PER_TOKEN

    def compress(self, context: str, query: str) -> str:
        """
        Return a compressed version of *context* relevant to *query*.

        If the context already fits within the budget it is returned unchanged.
        """
        if len(context) <= self._max_chars:
            return context

        query_tokens = _tokenize(query)
        blocks = self._split_blocks(context)

        if not blocks:
            return context[: self._max_chars]

        headers = []
        content = []
        for i, block in enumerate(blocks):
            if block.startswith("##") or block.startswith("==="):
                headers.append((i, block))
            else:
                score = _bm25_lite(query_tokens, block)
                content.append((i, block, score))

        # Sort content blocks by descending relevance
        content.sort(key=lambda x: (-x[2], x[0]))

        selected: set[int] = set()
        budget = self._max_chars

        # Reserve space for headers (lightweight navigation)
        header_chars = sum(len(b) + 2 for _, b in headers)
        remaining = budget - min(header_chars, budget // 4)

        for _, block in headers:
            selected.add(headers[[h[1] for h in headers].index(block)][0])

        # Fill remaining budget with best-scored blocks (preserve original order)
        for i, block, _ in content:
            if remaining <= 0:
                break
            if len(block) <= remaining:
                selected.add(i)
                remaining -= len(block) + 2

        if not selected:
            return context[: self._max_chars]

        result = "\n\n".join(blocks[i] for i in sorted(selected))
        return result

    @staticmethod
    def _split_blocks(context: str) -> List[str]:
        """Split context on natural block boundaries."""
        raw = re.split(r"\n\n+|\n---\n|\n(?=##\s|\n===)", context)
        return [b.strip() for b in raw if b.strip()]
