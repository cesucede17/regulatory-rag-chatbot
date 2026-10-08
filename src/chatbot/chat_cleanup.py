"""
chat_cleanup.py — Elimina chats y mensajes con más de CHAT_RETENTION_DAYS días.

Uso manual:  python src/chatbot/chat_cleanup.py (desde la raiz del modulo)
Programado:  schtasks (ver instrucciones al final del archivo)
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

# Este fichero vive en src/chatbot/, asi que parents[2] es la raiz del modulo.
# Se anade src/ al sys.path para poder ejecutarlo tambien como script suelto
# (python src/chatbot/<este fichero>.py), no solo importado por la app.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from chatbot.config import settings  # noqa: E402 (tras preparar sys.path)


async def run_cleanup() -> None:
    db_path = Path(settings.db_path)
    if not db_path.exists():
        print(f"[Cleanup] Base de datos no encontrada en {db_path}. Nada que hacer.")
        return

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=settings.chat_retention_days)
    ).isoformat()
    print(
        f"[Cleanup] Eliminando chats anteriores a {cutoff[:10]} (retención: {settings.chat_retention_days} días)"
    )

    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("PRAGMA foreign_keys=ON")
        # ON DELETE CASCADE en messages → se eliminan automáticamente
        cursor = await db.execute("DELETE FROM chats WHERE updated_at < ?", (cutoff,))
        deleted = cursor.rowcount
        await db.commit()
        print(f"[Cleanup] Eliminados {deleted} chat(s) y sus mensajes.")

    # Report remaining
    async with aiosqlite.connect(str(db_path)) as db:
        row = await db.execute("SELECT COUNT(*) FROM chats")
        remaining = (await row.fetchone())[0]
        print(f"[Cleanup] Conversaciones restantes en BD: {remaining}")


if __name__ == "__main__":
    asyncio.run(run_cleanup())


# ---------------------------------------------------------------------------
# Windows Task Scheduler setup (run once as admin in PowerShell)
# ---------------------------------------------------------------------------
# schtasks /Create /TN "Tambora\ChatCleanup" ^
#   /TR "\"C:\Users\CesarSuelaCedenilla\Desktop\JARVIS\Tambora_servidor\.venv\Scripts\python.exe\" \"C:\Users\CesarSuelaCedenilla\Desktop\JARVIS\Tambora_servidor\apps\chatbot\chat_cleanup.py\"" ^
#   /SC DAILY /ST 02:00 /F
#
# If using uv:
#   /TR "uv run \"C:\...\apps\chatbot\chat_cleanup.py\""
