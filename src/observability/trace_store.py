from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from src.observability.context import current_run_id, current_thread_id

DEFAULT_TRACE_DB_PATH = Path(__file__).resolve().parents[2] / "traces.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_loads(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


class TraceStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or DEFAULT_TRACE_DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT,
                    status TEXT,
                    created_at TEXT,
                    finished_at TEXT,
                    title TEXT,
                    metadata_json TEXT,
                    input_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    ts TEXT,
                    event_type TEXT,
                    agent TEXT,
                    node TEXT,
                    step TEXT,
                    duration_ms REAL,
                    token_json TEXT,
                    payload_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_run_id ON events(run_id)"
            )
            conn.commit()

    def create_run(
        self,
        thread_id: str,
        title: Optional[str] = None,
        input_payload: Any | None = None,
        metadata: Any | None = None,
        status: str = "running",
    ) -> str:
        run_id = uuid4().hex
        created_at = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, thread_id, status, created_at, finished_at,
                    title, metadata_json, input_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    thread_id,
                    status,
                    created_at,
                    None,
                    title,
                    _json_dumps(metadata) if metadata is not None else None,
                    _json_dumps(input_payload) if input_payload is not None else None,
                ),
            )
            conn.commit()
        return run_id

    def update_run_status(self, run_id: str, status: str) -> None:
        finished_at = _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                (status, finished_at, run_id),
            )
            conn.commit()

    def add_event(
        self,
        run_id: str,
        event_type: str,
        payload: Any | None = None,
        agent: str | None = None,
        node: str | None = None,
        step: str | None = None,
        duration_ms: float | None = None,
        token_usage: Any | None = None,
    ) -> int:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    run_id, ts, event_type, agent, node, step,
                    duration_ms, token_json, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _utc_now(),
                    event_type,
                    agent,
                    node,
                    step,
                    duration_ms,
                    _json_dumps(token_usage) if token_usage is not None else None,
                    _json_dumps(payload) if payload is not None else None,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_runs(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT run_id, thread_id, status, created_at, finished_at, title, metadata_json
                FROM runs
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
        return [
            {
                "run_id": row["run_id"],
                "thread_id": row["thread_id"],
                "status": row["status"],
                "created_at": row["created_at"],
                "finished_at": row["finished_at"],
                "title": row["title"],
                "metadata": _json_loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def get_run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT run_id, thread_id, status, created_at, finished_at,
                       title, metadata_json, input_json
                FROM runs
                WHERE run_id = ?
                """,
                (run_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
            "title": row["title"],
            "metadata": _json_loads(row["metadata_json"]),
            "input": _json_loads(row["input_json"]),
        }

    def get_events(
        self, run_id: str, since_id: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, ts, event_type, agent, node, step, duration_ms,
                       token_json, payload_json
                FROM events
                WHERE run_id = ? AND id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (run_id, since_id, limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "id": row["id"],
                "ts": row["ts"],
                "event_type": row["event_type"],
                "agent": row["agent"],
                "node": row["node"],
                "step": row["step"],
                "duration_ms": row["duration_ms"],
                "token_usage": _json_loads(row["token_json"]),
                "payload": _json_loads(row["payload_json"]),
            }
            for row in rows
        ]


trace_store = TraceStore()


def get_current_run_id() -> str | None:
    return current_run_id.get()


def get_current_thread_id() -> str | None:
    return current_thread_id.get()
