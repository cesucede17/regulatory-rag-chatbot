"""Persistencia y agregados del consumo de IA del chatbot BOE
(`chat_llm_usage`) — mismo patrón que
`auditorias.repository.{record del audit_llm_usage inline, get_admin_usage_summary}`,
pero sin project_id/clause_id (el chatbot no tiene proyectos)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiosqlite

from shared.llm_usage import LlmUsage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rows(cur_rows) -> list[dict]:
    return [dict(r) for r in cur_rows]


async def record(
    db: aiosqlite.Connection, *, user_id: int, purpose: str, usage: LlmUsage
) -> None:
    await db.execute(
        """INSERT INTO chat_llm_usage
               (user_id, purpose, model, input_tokens, output_tokens,
                cache_creation_tokens, cache_read_tokens, cost_usd, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            purpose,
            usage.model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_tokens,
            usage.cache_read_tokens,
            usage.cost_usd,
            _now(),
        ),
    )
    await db.commit()


def _totals_from_rows(rows: list[dict]) -> dict:
    return {
        "calls": sum(r["calls"] for r in rows),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "cache_creation_tokens": sum(r["cache_creation_tokens"] or 0 for r in rows),
        "cache_read_tokens": sum(r["cache_read_tokens"] or 0 for r in rows),
        "cost_usd": round(sum(r["cost_usd"] or 0.0 for r in rows), 6),
    }


async def get_admin_usage_summary(
    db: aiosqlite.Connection,
    *,
    since: str | None,
    until: str | None,
    user_id: int | None,
) -> dict:
    clauses: list[str] = []
    params: list[Any] = []
    if since:
        clauses.append("l.created_at >= ?")
        params.append(since)
    if until:
        clauses.append("l.created_at <= ?")
        params.append(until)
    if user_id is not None:
        clauses.append("l.user_id = ?")
        params.append(user_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    cur = await db.execute(
        f"""SELECT u.username AS username, l.model AS model, l.purpose AS purpose, COUNT(*) AS calls,
                   SUM(l.input_tokens) AS input_tokens, SUM(l.output_tokens) AS output_tokens,
                   SUM(l.cache_creation_tokens) AS cache_creation_tokens,
                   SUM(l.cache_read_tokens) AS cache_read_tokens, SUM(l.cost_usd) AS cost_usd
            FROM chat_llm_usage l LEFT JOIN users u ON u.id = l.user_id
            {where}
            GROUP BY u.username, l.model, l.purpose
            ORDER BY u.username, l.model, l.purpose""",
        params,
    )
    rows = _rows(await cur.fetchall())
    return {"rows": rows, "totals": _totals_from_rows(rows)}
