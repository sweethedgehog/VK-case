"""Small SQLite-backed store for short per-session chat histories."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "chat_history.sqlite3"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_FILE, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_store() -> None:
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                session_id TEXT PRIMARY KEY,
                history_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )"""
        )


def load_history(session_id: str) -> list[tuple[str, str]]:
    initialize_store()
    with _connect() as conn:
        row = conn.execute(
            "SELECT history_json FROM conversations WHERE session_id = ?", (session_id,)
        ).fetchone()
    if not row:
        return []
    try:
        payload = json.loads(row[0])
        if not isinstance(payload, list):
            return []
        history: list[tuple[str, str]] = []
        for item in payload[-8:]:
            if isinstance(item, list) and len(item) == 2:
                history.append((str(item[0])[:40], str(item[1])[:6000]))
        return history
    except (ValueError, TypeError):
        return []


def save_history(session_id: str, history: Iterable[tuple[str, str]]) -> None:
    initialize_store()
    limited = [(str(role)[:40], str(content)[:6000]) for role, content in list(history)[-8:]]
    serialized = json.dumps(limited, ensure_ascii=False)
    with _connect() as conn:
        conn.execute(
            """INSERT INTO conversations(session_id, history_json, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(session_id) DO UPDATE SET
                 history_json = excluded.history_json,
                 updated_at = CURRENT_TIMESTAMP""",
            (session_id, serialized),
        )
