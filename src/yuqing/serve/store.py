"""SQLite persistence for M7 demo tickets and human decisions."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class TicketStore:
    """Persist ticket state and atomically claim one approval per case."""

    def __init__(self, path: Path) -> None:
        """Bind this store to one D-drive project database path."""
        self.path = path

    def initialize(self, expiry_minutes: int) -> int:
        """Create tables and expire stale processing rows; return the expired count."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                "CREATE TABLE IF NOT EXISTS tickets ("
                "case_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL, "
                "submitted_at REAL NOT NULL, updated_at REAL NOT NULL, claimed_at REAL, "
                "input_json TEXT NOT NULL, result_json TEXT, failure_reason TEXT, "
                "checkpoint_mode TEXT NOT NULL DEFAULT 'memory', graph_contract TEXT, "
                "config_sha256 TEXT, input_sha256 TEXT, pending_decision_id TEXT);"
                "CREATE TABLE IF NOT EXISTS decisions ("
                "case_id TEXT PRIMARY KEY REFERENCES tickets(case_id), decision TEXT NOT NULL, "
                "reason TEXT NOT NULL, reviewer TEXT NOT NULL, final_disposition TEXT, decided_at REAL NOT NULL);"
            )
            self._migrate_ticket_columns(db)
            cutoff = time.time() - expiry_minutes * 60
            cursor = db.execute(
                "UPDATE tickets SET status='expired', updated_at=?, failure_reason='processing_timeout' "
                "WHERE status='processing' AND updated_at < ?", (time.time(), cutoff))
            return cursor.rowcount

    def create_ticket(self, case_id: str, thread_id: str, ticket: dict[str, Any], *,
                      checkpoint_mode: str = "memory", graph_contract: str | None = None,
                      config_sha256: str | None = None, input_sha256: str | None = None) -> None:
        """Persist the submitted ticket before the graph starts."""
        now = time.time()
        with self._connect() as db:
            db.execute("INSERT INTO tickets(case_id, thread_id, status, submitted_at, updated_at, input_json, "
                "checkpoint_mode, graph_contract, config_sha256, input_sha256) "
                "VALUES(?, ?, 'processing', ?, ?, ?, ?, ?, ?, ?)",
                (case_id, thread_id, now, now, _json(ticket), checkpoint_mode,
                 graph_contract, config_sha256, input_sha256))

    def get_ticket(self, case_id: str) -> dict[str, Any] | None:
        """Return one decoded ticket row, or None when the identity is unknown."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM tickets WHERE case_id=?", (case_id,)).fetchone()
        return _decode_row(row) if row else None

    def claim_approval(self, case_id: str) -> bool:
        """Atomically let exactly one requester move an awaiting ticket to processing."""
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("UPDATE tickets SET status='processing', claimed_at=?, updated_at=? "
                "WHERE case_id=? AND status='awaiting_approval'", (now, now, case_id))
            db.commit()
            return cursor.rowcount == 1

    def save_waiting(self, case_id: str, result: dict[str, Any],
                     pending_decision_id: str | None = None) -> None:
        """Save the paused graph result and make the ticket available for review."""
        with self._connect() as db:
            cursor = db.execute("UPDATE tickets SET status='awaiting_approval', updated_at=?, result_json=?, "
                "pending_decision_id=? WHERE case_id=? AND status='processing'",
                (time.time(), _json(result), pending_decision_id, case_id))
            if cursor.rowcount != 1:
                raise RuntimeError("ticket_waiting_transition_invalid")

    def expire_waiting(self, case_id: str, reason: str) -> bool:
        """Expire only a still-waiting ticket; never overwrite a concurrent decision."""
        with self._connect() as db:
            cursor = db.execute("UPDATE tickets SET status='expired', updated_at=?, failure_reason=? "
                "WHERE case_id=? AND status='awaiting_approval'",
                (time.time(), reason, case_id))
            return cursor.rowcount == 1

    def finish_without_decision(self, case_id: str, result: dict[str, Any]) -> None:
        """Complete a case that did not enter the human-review gate."""
        self._update(case_id, "completed", result=result)

    def finish(self, case_id: str, review: dict[str, Any], result: dict[str, Any]) -> None:
        """Insert the unique decision and complete the claimed ticket in one transaction."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO decisions(case_id, decision, reason, reviewer, final_disposition, decided_at) "
                "VALUES(?, ?, ?, ?, ?, ?)", (case_id, review["decision"], review["reason"],
                review["reviewer"], review.get("final_disposition"), time.time()))
            cursor = db.execute("UPDATE tickets SET status='completed', updated_at=?, result_json=? "
                "WHERE case_id=? AND status='processing'", (time.time(), _json(result), case_id))
            if cursor.rowcount != 1:
                db.rollback()
                raise RuntimeError("ticket_completion_claim_lost")
            db.commit()

    def fail(self, case_id: str, status: str, reason: str) -> None:
        """Record a terminal failure without making the graph eligible for replay."""
        if status not in {"failed", "resume_failed", "expired"}:
            raise ValueError("ticket_failure_status_invalid")
        self._update(case_id, status, failure_reason=reason)

    def _update(self, case_id: str, status: str, *, result: dict[str, Any] | None = None,
                failure_reason: str | None = None) -> None:
        """Update only the supplied ticket fields and refresh its timestamp."""
        with self._connect() as db:
            db.execute("UPDATE tickets SET status=?, updated_at=?, result_json=COALESCE(?, result_json), "
                "failure_reason=COALESCE(?, failure_reason) WHERE case_id=?",
                (status, time.time(), _json(result) if result is not None else None,
                 failure_reason, case_id))

    def _connect(self) -> sqlite3.Connection:
        """Open a row-aware connection with foreign-key checks enabled."""
        db = sqlite3.connect(self.path, timeout=5.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @staticmethod
    def _migrate_ticket_columns(db: sqlite3.Connection) -> None:
        """Add X5 recovery metadata to existing M7 databases without rewriting rows."""
        existing = {row[1] for row in db.execute("PRAGMA table_info(tickets)")}
        additions = {"checkpoint_mode": "TEXT NOT NULL DEFAULT 'memory'",
            "graph_contract": "TEXT", "config_sha256": "TEXT", "input_sha256": "TEXT",
            "pending_decision_id": "TEXT"}
        for name, declaration in additions.items():
            if name not in existing:
                db.execute(f"ALTER TABLE tickets ADD COLUMN {name} {declaration}")


def _json(value: dict[str, Any]) -> str:
    """Serialize an API payload deterministically for SQLite storage."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    """Decode the two JSON columns without exposing SQLite implementation details."""
    value = dict(row)
    value["input"] = json.loads(value.pop("input_json"))
    raw_result = value.pop("result_json")
    value["result"] = json.loads(raw_result) if raw_result else None
    return value
