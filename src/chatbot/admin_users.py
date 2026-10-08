"""Gestion de usuarios del Chatbot BOE — prefijo `/api/admin/users`.

Cada herramienta tiene su propia base de datos y su propia tabla `users`,
asi que tambien su propia copia de este router: los usuarios de una no
existen en la otra y se crean por separado. Es codigo duplicado a
proposito — el precio de que las dos aplicaciones sean independientes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import auth

router = APIRouter(prefix="/api/admin/users")


class UserCreateIn(BaseModel):
    first_name: str = ""
    last_name: str = ""
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=200)
    role: str = Field(pattern="^(admin|user)$")


class UserPatchIn(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    username: str | None = Field(default=None, min_length=1, max_length=100)
    role: str | None = Field(default=None, pattern="^(admin|user)$")


class PasswordResetIn(BaseModel):
    new_password: str = Field(min_length=8, max_length=200)


_USER_COLS = "id, username, first_name, last_name, role, is_active"


@router.get("")
async def list_users(
    user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = await db.execute(
        """SELECT u.id, u.username, u.first_name, u.last_name, u.role, u.is_active,
                  COALESCE(d.count, 0) AS today_count
           FROM users u
           LEFT JOIN daily_usage d ON d.user_id = u.id AND d.date = ?
           ORDER BY u.id""",
        (today,),
    )
    return JSONResponse([dict(r) for r in await rows.fetchall()])


@router.post("", status_code=201)
async def create_user(
    body: UserCreateIn,
    user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    now = datetime.now(timezone.utc).isoformat()
    hashed = auth.hash_password(body.password)
    try:
        cur = await db.execute(
            """INSERT INTO users (username, password_hash, first_name, last_name, role, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (body.username, hashed, body.first_name, body.last_name, body.role, now),
        )
        await db.commit()
    except aiosqlite.IntegrityError:
        return JSONResponse(
            {"error": "Ese nombre de usuario ya existe."}, status_code=409
        )

    row = await db.execute(
        f"SELECT {_USER_COLS} FROM users WHERE id = ?", (cur.lastrowid,)
    )
    return JSONResponse(dict(await row.fetchone()), status_code=201)


@router.patch("/{user_id}")
async def patch_user(
    user_id: int,
    body: UserPatchIn,
    current_user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    if user_id == current_user["id"] and body.role is not None and body.role != "admin":
        return JSONResponse(
            {"error": "No puedes quitarte a ti mismo el rol de admin."}, status_code=400
        )

    fields = body.model_dump(exclude_unset=True)
    if not fields:
        row = await db.execute(
            f"SELECT {_USER_COLS} FROM users WHERE id = ?", (user_id,)
        )
        return JSONResponse(dict(await row.fetchone()))

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    try:
        await db.execute(
            f"UPDATE users SET {set_clause} WHERE id = ?", (*fields.values(), user_id)
        )
        await db.commit()
    except aiosqlite.IntegrityError:
        return JSONResponse(
            {"error": "Ese nombre de usuario ya existe."}, status_code=409
        )

    row = await db.execute(f"SELECT {_USER_COLS} FROM users WHERE id = ?", (user_id,))
    return JSONResponse(dict(await row.fetchone()))


@router.patch("/{user_id}/password")
async def reset_user_password(
    user_id: int,
    body: PasswordResetIn,
    user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    hashed = auth.hash_password(body.new_password)
    await db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?", (hashed, user_id)
    )
    await db.commit()
    return JSONResponse({"success": True})


@router.patch("/{user_id}/toggle")
async def toggle_user(
    user_id: int,
    current_user: dict = Depends(auth.require_admin),
    db: aiosqlite.Connection = Depends(auth.get_db),
):
    if user_id == current_user["id"]:
        return JSONResponse(
            {"error": "No puedes desactivar tu propia cuenta."}, status_code=400
        )
    await db.execute(
        "UPDATE users SET is_active = CASE WHEN is_active = 1 THEN 0 ELSE 1 END WHERE id = ?",
        (user_id,),
    )
    await db.commit()
    return JSONResponse({"success": True})
