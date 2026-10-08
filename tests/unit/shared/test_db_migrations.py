"""Tests de common/db_migrations.py: aplicación en orden, idempotencia
(no reaplica lo ya registrado), y que un fallo a mitad no deshace las
migraciones ya comiteadas antes de ella.

pytest-asyncio no es una dependencia del proyecto — igual que el resto de la
suite (ver tests/test_report_db.py), cada test envuelve su cuerpo async con
asyncio.run()."""

import asyncio

import aiosqlite
import pytest

from shared import db_migrations as m


def test_apply_migrations_runs_in_order_and_records_names(tmp_path):
    db_path = tmp_path / "test.db"
    migrations = [
        ("0001_initial", "CREATE TABLE foo (id INTEGER PRIMARY KEY);"),
        ("0002_add_bar", "CREATE TABLE bar (id INTEGER PRIMARY KEY);"),
    ]

    async def _run():
        applied = await m.apply_migrations(db_path, migrations)
        assert applied == ["0001_initial", "0002_add_bar"]

        async with aiosqlite.connect(str(db_path)) as db:
            names = await m.applied_names(db)
            assert names == {"0001_initial", "0002_add_bar"}

            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('foo','bar')"
            )
            rows = {row[0] for row in await cursor.fetchall()}
            assert rows == {"foo", "bar"}

    asyncio.run(_run())


def test_apply_migrations_is_idempotent_on_second_call(tmp_path):
    db_path = tmp_path / "test.db"
    migrations = [("0001_initial", "CREATE TABLE foo (id INTEGER PRIMARY KEY);")]

    async def _run():
        first = await m.apply_migrations(db_path, migrations)
        assert first == ["0001_initial"]

        second = await m.apply_migrations(db_path, migrations)
        assert second == []

    asyncio.run(_run())


def test_apply_migrations_only_applies_new_ones_on_extended_list(tmp_path):
    db_path = tmp_path / "test.db"

    async def _run():
        await m.apply_migrations(
            db_path, [("0001_initial", "CREATE TABLE foo (id INTEGER PRIMARY KEY);")]
        )
        applied = await m.apply_migrations(
            db_path,
            [
                ("0001_initial", "CREATE TABLE foo (id INTEGER PRIMARY KEY);"),
                ("0002_add_bar", "CREATE TABLE bar (id INTEGER PRIMARY KEY);"),
            ],
        )
        assert applied == ["0002_add_bar"]

    asyncio.run(_run())


def test_failed_migration_leaves_previously_applied_ones_intact(tmp_path):
    db_path = tmp_path / "test.db"
    migrations = [
        ("0001_initial", "CREATE TABLE foo (id INTEGER PRIMARY KEY);"),
        ("0002_broken", "THIS IS NOT VALID SQL;"),
        ("0003_never_reached", "CREATE TABLE baz (id INTEGER PRIMARY KEY);"),
    ]

    async def _run():
        with pytest.raises(Exception):
            await m.apply_migrations(db_path, migrations)

        async with aiosqlite.connect(str(db_path)) as db:
            names = await m.applied_names(db)
            # 0001 quedó comiteada; 0002 falló y no se registró; 0003 nunca
            # se intentó (la excepción se propaga sin reintentar).
            assert names == {"0001_initial"}

            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('foo','baz')"
            )
            rows = {row[0] for row in await cursor.fetchall()}
            assert rows == {"foo"}

    asyncio.run(_run())


def test_multi_statement_migration_rolls_back_completely_on_mid_script_failure(
    tmp_path,
):
    """Regresión: una migración de VARIAS sentencias donde la segunda falla
    no debe dejar la primera (ya ejecutada con éxito) persistida en disco.

    executescript() por sí solo NO es transaccional (ejecuta en autocommit
    sentencia a sentencia); sin un BEGIN explícito envolviendo el script
    completo, la tabla `primera_tabla_ok` de este test quedaría creada y
    huérfana aunque la migración nunca se registre en schema_migrations.
    """
    db_path = tmp_path / "test.db"
    migrations = [
        (
            "0001_multi_statement_fails_midway",
            "CREATE TABLE primera_tabla_ok (id INTEGER PRIMARY KEY);\n"
            "ESTO NO ES SQL VALIDO EN ABSOLUTO;\n"
            "CREATE TABLE nunca_alcanzada (id INTEGER PRIMARY KEY);",
        )
    ]

    async def _run():
        with pytest.raises(Exception):
            await m.apply_migrations(db_path, migrations)

        async with aiosqlite.connect(str(db_path)) as db:
            names = await m.applied_names(db)
            assert names == set(), "la migración fallida no debe quedar registrada"

            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('primera_tabla_ok', 'nunca_alcanzada')"
            )
            rows = {row[0] for row in await cursor.fetchall()}
            assert rows == set(), (
                "ninguna sentencia de la migración fallida debe sobrevivir, "
                "ni siquiera la que se ejecutó con éxito antes del fallo"
            )

    asyncio.run(_run())


def test_multi_statement_migration_commits_all_statements_together_on_success(tmp_path):
    """Complemento del test anterior: si el script entero tiene éxito, todas
    sus sentencias (y el registro en schema_migrations) se confirman juntas."""
    db_path = tmp_path / "test.db"
    migrations = [
        (
            "0001_multi_statement_ok",
            "CREATE TABLE tabla_a (id INTEGER PRIMARY KEY);\n"
            "CREATE TABLE tabla_b (id INTEGER PRIMARY KEY);",
        )
    ]

    async def _run():
        applied = await m.apply_migrations(db_path, migrations)
        assert applied == ["0001_multi_statement_ok"]

        async with aiosqlite.connect(str(db_path)) as db:
            names = await m.applied_names(db)
            assert names == {"0001_multi_statement_ok"}

            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name IN ('tabla_a', 'tabla_b')"
            )
            rows = {row[0] for row in await cursor.fetchall()}
            assert rows == {"tabla_a", "tabla_b"}

    asyncio.run(_run())


def test_ensure_registry_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"

    async def _run():
        async with aiosqlite.connect(str(db_path)) as db:
            await m.ensure_registry(db)
            await m.ensure_registry(db)  # no debe lanzar ni duplicar nada
            names = await m.applied_names(db)
            assert names == set()

    asyncio.run(_run())
