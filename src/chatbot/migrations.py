"""Migraciones SQL del Chatbot BOE, aplicadas con el runner generico de
`shared/db_migrations.py`.

`auth.py` sigue creando el esquema base con `CREATE TABLE IF NOT EXISTS`
(`_DB_SCHEMA`, en cada arranque) — correcto e idempotente para bases de
datos NUEVAS, pero no migra columnas nuevas en una base ya existente, que es
justo lo que resuelve este modulo.

Los nombres de las migraciones se conservan tal cual eran cuando las dos
herramientas compartian `data/tambora.db`: `scripts/split_db.py` copia la
tabla `schema_migrations` entera a las dos bases nuevas, asi que cambiarlos
haria que se reaplicaran sobre datos ya migrados.
"""

from shared.db_migrations import Migration, apply_migrations

# SQLite no permite modificar un CHECK con ALTER TABLE — se reconstruyen las
# tablas relacionadas. Es seguro con FK desactivadas por el runner durante la
# migracion: el runner maneja el PRAGMA para permitir DDL como DROP TABLE de
# tablas con referencias entrantes.
_SQL_AUTH_0001 = """
CREATE TABLE users_new (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    first_name    TEXT NOT NULL DEFAULT '',
    last_name     TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL CHECK(role IN ('admin','user','sge')),
    is_active     INTEGER DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE TABLE chats_new (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE daily_usage_new (
    user_id INTEGER NOT NULL,
    date    TEXT NOT NULL,
    count   INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, date),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

INSERT INTO users_new (id, username, password_hash, role, is_active, created_at)
    SELECT id, username, password_hash, role, is_active, created_at FROM users;

INSERT INTO chats_new (id, user_id, title, created_at, updated_at)
    SELECT id, user_id, title, created_at, updated_at FROM chats;

INSERT INTO daily_usage_new (user_id, date, count)
    SELECT user_id, date, count FROM daily_usage;

DROP TABLE chats;
DROP TABLE daily_usage;
DROP TABLE users;

ALTER TABLE users_new RENAME TO users;
ALTER TABLE chats_new RENAME TO chats;
ALTER TABLE daily_usage_new RENAME TO daily_usage;

CREATE INDEX IF NOT EXISTS idx_chats_user    ON chats(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage         ON daily_usage(user_id, date);
"""

_SQL_CHAT_0001 = """
CREATE TABLE chat_llm_usage (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               INTEGER REFERENCES users(id) ON DELETE SET NULL,
    purpose               TEXT NOT NULL CHECK(purpose IN ('chat_response','parameter_extraction')),
    model                 TEXT NOT NULL,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd              REAL NOT NULL DEFAULT 0.0,
    created_at            TEXT NOT NULL
);
CREATE INDEX idx_chat_llm_usage ON chat_llm_usage(created_at DESC);
CREATE INDEX idx_chat_llm_usage_model ON chat_llm_usage(model, created_at DESC);
"""

MIGRATIONS: list[Migration] = [
    ("auth_0001_users_name_and_role", _SQL_AUTH_0001),
    ("chat_0001_llm_usage", _SQL_CHAT_0001),
]


async def run(db_path) -> list[str]:
    return await apply_migrations(db_path, MIGRATIONS)
