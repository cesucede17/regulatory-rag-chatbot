"""Runner de migraciones SQL genérico y agnóstico de módulo.

Hoy el esquema de `auth.py` se aplica con `executescript` en cada arranque,
lo cual es idempotente solo para `CREATE TABLE IF NOT EXISTS` — no migra
columnas nuevas en tablas existentes. Este runner resuelve eso con una tabla
de registro (`schema_migrations`), en vez de `PRAGMA user_version`: un único
entero no puede repartirse entre módulos independientes (chatbot, auditorías,
futuros) que no se coordinan entre sí.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import aiosqlite

logger = logging.getLogger(__name__)

# (name, sql) — sql es un bloque de texto con el DDL completo de esa migración.
Migration = tuple[str, str]

_REGISTRY_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name       TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


async def ensure_registry(db: aiosqlite.Connection) -> None:
    """Crea `schema_migrations` si no existe. Idempotente."""
    await db.executescript(_REGISTRY_DDL)
    await db.commit()


async def applied_names(db: aiosqlite.Connection) -> set[str]:
    """Nombres de las migraciones ya registradas como aplicadas."""
    cursor = await db.execute("SELECT name FROM schema_migrations")
    rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def apply_migrations(
    db_path: str | Path, migrations: Sequence[Migration]
) -> list[str]:
    """Abre su propia conexión aiosqlite a `db_path` y aplica en orden las
    migraciones cuyo `name` no esté ya en `schema_migrations`.

    Cada migración se aplica en una única transacción real y atómica: se
    antepone un `BEGIN;` explícito al SQL de la migración antes de pasarlo a
    `executescript` (por sí solo, `executescript` de sqlite3/aiosqlite NO es
    transaccional — ejecuta el script sentencia a sentencia en modo
    autocommit y confirma en disco cada `CREATE TABLE` según se ejecuta, así
    que una migración de varias sentencias donde la segunda falla dejaría la
    primera persistida y huérfana sin este `BEGIN` explícito). Si el script
    se ejecuta entero sin error, la conexión queda con una transacción
    abierta (el script nunca emite su propio `COMMIT`); el registro en
    `schema_migrations` se inserta dentro de esa misma transacción y
    `db.commit()` confirma DDL + registro juntos, de forma atómica. Si
    cualquier sentencia del script falla, se hace `db.rollback()` explícito
    (deshaciendo TODAS las sentencias de esa migración, incluidas las que sí
    se habían ejecutado antes de la que falló) y la excepción se propaga sin
    reintentar — las migraciones de iteraciones anteriores del bucle, ya
    comiteadas, permanecen aplicadas.

    IMPORTANTE para quien escriba el SQL de una migración: no incluyas tú
    mismo `BEGIN`/`COMMIT`/`ROLLBACK` en el texto — el runner ya envuelve la
    transacción; anidarlos rompería el mecanismo anterior.

    Devuelve la lista de nombres efectivamente aplicados en esta llamada
    (vacía si ya estaban todas aplicadas).
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    applied: list[str] = []
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await ensure_registry(db)
        already_applied = await applied_names(db)

        for name, sql in migrations:
            if name in already_applied:
                continue
            try:
                # Deshabilita FK justo antes de la migración (fuera de la transacción)
                # para permitir DDL como DROP TABLE de tablas referenciadas.
                # PRAGMA fuera de transacción afecta a la conexión; dentro sería un no-op.
                await db.execute("PRAGMA foreign_keys=OFF")
                await db.executescript(f"BEGIN;\n{sql}")
                applied_at = datetime.now(timezone.utc).isoformat()
                await db.execute(
                    "INSERT INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                    (name, applied_at),
                )
                await db.commit()
            except Exception:
                if db.in_transaction:
                    await db.rollback()
                logger.error("Migración fallida, revertida por completo: %s", name)
                raise
            finally:
                # Restaura FK siempre tras la migración (exitosa o fallida)
                await db.execute("PRAGMA foreign_keys=ON")
            applied.append(name)
            logger.info("Migración aplicada: %s", name)

    return applied
