"""SQLite append store for the experiment cost ledger; recovery is a later card."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Mapping
from urllib.parse import quote
from uuid import UUID

_FORMAT = "1"
_GENESIS = "genesis"
_META = {"format", "manifest", "manifest_hash", "revision", "head_hash"}
_MANIFEST = {"experiment_id", "mode", "max_cost_cny", "price_fingerprint", "conditions_fingerprint", "format_version"}
_EVENTS = ("prepared", "reserved", "dispatched", "settled", "cancelled", "unknown", "halted")
_EVENT_FIELDS = {"event_type", "request_id", "sample_id", "group_id", "run_id", "variant", "amount_cny", "actual_cost_cny", "reason"}


@dataclass(frozen=True)
class LedgerHistory:
    manifest: Mapping[str, object]
    revision: int
    events: tuple[Mapping[str, object], ...]
    event_hashes: tuple[str, ...]


class ExperimentBudgetStoreError(RuntimeError): pass


class StaleRevisionError(ExperimentBudgetStoreError): pass


class LedgerIntegrityError(ExperimentBudgetStoreError): pass


class LedgerWriteError(ExperimentBudgetStoreError): pass


def _money_text(value: object, *, positive: bool) -> str:
    """Validate Decimal or decimal text and return the exact stored text."""
    if not isinstance(value, (Decimal, str)) or isinstance(value, bool):
        raise ValueError("money_must_be_decimal_or_text")
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid_money") from exc
    if not amount.is_finite() or amount < 0 or (positive and amount == 0):
        raise ValueError("invalid_money")
    return str(amount)


def _manifest_data(manifest: Mapping[str, object]) -> dict[str, object]:
    """Validate fixed experiment metadata and normalize its cap to decimal text."""
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST:
        raise ValueError("invalid_experiment_manifest_fields")
    data = dict(manifest)
    if not isinstance(data["experiment_id"], str) or re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", data["experiment_id"]) is None:
        raise ValueError("invalid_experiment_id")
    if data["mode"] not in ("full", "pilot") or type(data["format_version"]) is not int or data["format_version"] != 1:
        raise ValueError("invalid_experiment_mode_or_format")
    data["max_cost_cny"] = _money_text(data["max_cost_cny"], positive=True)
    for name in ("price_fingerprint", "conditions_fingerprint"):
        if not isinstance(data[name], str) or re.fullmatch(r"[0-9a-f]{64}", data[name]) is None:
            raise ValueError(f"invalid_{name}")
    return data


def _event_data(event: Mapping[str, object]) -> dict[str, object]:
    """Validate an allowlisted event so prompts and evidence cannot be stored."""
    if not isinstance(event, Mapping) or not set(event).issubset(_EVENT_FIELDS) or "event_type" not in event:
        raise ValueError("invalid_ledger_event_fields")
    data = dict(event)
    if data["event_type"] not in _EVENTS:
        raise ValueError("invalid_ledger_event_type")
    request_id = data.get("request_id")
    if data["event_type"] != "halted" and request_id is None:
        raise ValueError("request_id_required")
    if request_id is not None and (not isinstance(request_id, str) or UUID(request_id).hex != request_id):
        raise ValueError("invalid_request_id")
    for name in _EVENT_FIELDS - {"event_type", "amount_cny", "actual_cost_cny", "reason"}:
        if name in data and (not isinstance(data[name], str) or re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", data[name]) is None):
            raise ValueError(f"invalid_{name}")
    for name in ("amount_cny", "actual_cost_cny"):
        if name in data:
            data[name] = _money_text(data[name], positive=False)
    reason = data.get("reason")
    if reason is not None and (not isinstance(reason, str) or re.fullmatch(r"[a-z][a-z0-9_.-]{0,79}", reason) is None):
        raise ValueError("reason_must_be_short_code")
    return data


def _json(value: object) -> str:
    """Serialize values into stable JSON text with no floating-point NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _connect(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file in rw mode, never silently creating one."""
    uri = quote(path.resolve().as_posix(), safe="/:")
    return sqlite3.connect(f"file:{uri}?mode=rw", uri=True, timeout=30, isolation_level=None, check_same_thread=False)


def _digest(revision: int, previous: str, event_json: str) -> str:
    """Hash the sequence, previous digest, and canonical event as one chain link."""
    body = _json({"revision": revision, "previous": previous, "event": json.loads(event_json)})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class ExperimentBudgetStore:
    """Append-only SQLite storage with transaction and revision checks."""

    def __init__(self, path: Path, connection: sqlite3.Connection) -> None:
        """Keep one connection and serialize access from this store instance."""
        self._path, self._connection, self._lock, self._closed = path, connection, RLock(), False

    @property
    def path(self) -> Path:
        """Return the canonical database path as the persistent ledger identity."""
        return self._path.resolve()

    @classmethod
    def create(cls, path: str | Path, manifest: Mapping[str, object]) -> ExperimentBudgetStore:
        """Create a new database without overwriting any existing file."""
        target, clean = Path(path), _manifest_data(manifest)
        if target.exists():
            raise FileExistsError("ledger_file_already_exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        connection: sqlite3.Connection | None = None
        try:
            target.touch(exist_ok=False)
            connection = _connect(target)
            raw = _json(clean)
            metadata = {"format": _FORMAT, "manifest": raw, "manifest_hash": hashlib.sha256(raw.encode()).hexdigest(), "revision": "0", "head_hash": _GENESIS}
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE events (revision INTEGER PRIMARY KEY, event_json TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL)")
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
            connection.commit()
            return cls(target, connection)
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise LedgerWriteError("ledger_initialization_failed") from exc

    @classmethod
    def open_existing(cls, path: str | Path, expected_manifest: Mapping[str, object]) -> ExperimentBudgetStore:
        """Open and fully verify an existing database and its expected conditions."""
        target, expected = Path(path), _manifest_data(expected_manifest)
        if not target.is_file():
            raise FileNotFoundError("ledger_file_missing")
        store: ExperimentBudgetStore | None = None
        try:
            store = cls(target, _connect(target))
            history = store.load()
        except ExperimentBudgetStoreError:
            if store is not None:
                store.close()
            raise
        except sqlite3.Error as exc:
            if store is not None:
                store.close()
            raise LedgerIntegrityError("ledger_database_unreadable") from exc
        if dict(history.manifest) != expected:
            store.close()
            raise ValueError("ledger_manifest_mismatch")
        return store

    def append(self, event: Mapping[str, object], expected_revision: int) -> int:
        """Atomically append one validated event if its expected revision is current."""
        clean = _event_data(event)
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool) or expected_revision < 0:
            raise ValueError("invalid_expected_revision")
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                history = self._load_unlocked()
                if history.revision != expected_revision:
                    raise StaleRevisionError("stale_ledger_revision")
                revision = history.revision + 1
                previous = history.event_hashes[-1] if history.event_hashes else _GENESIS
                raw, digest = _json(clean), ""
                digest = _digest(revision, previous, raw)
                self._connection.execute("INSERT INTO events VALUES (?, ?, ?, ?)", (revision, raw, previous, digest))
                self._connection.execute("UPDATE metadata SET value = ? WHERE key = 'revision'", (str(revision),))
                self._connection.execute("UPDATE metadata SET value = ? WHERE key = 'head_hash'", (digest,))
                self._connection.commit()
                return revision
            except ExperimentBudgetStoreError:
                self._rollback()
                raise
            except sqlite3.Error as exc:
                self._rollback()
                raise LedgerWriteError("ledger_event_write_failed") from exc

    def load(self) -> LedgerHistory:
        """Return history only after validating metadata and every event digest."""
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN")
                history = self._load_unlocked()
                self._connection.commit()
                return history
            except (ExperimentBudgetStoreError, sqlite3.Error):
                self._rollback()
                raise

    def close(self) -> None:
        """Close the connection; repeated calls are safe."""
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def _ensure_open(self) -> None:
        """Reject operations after the database connection has been closed."""
        if self._closed:
            raise ExperimentBudgetStoreError("ledger_store_closed")

    def _rollback(self) -> None:
        """Rollback an append transaction when a write or revision check fails."""
        if self._connection.in_transaction:
            self._connection.rollback()

    def _load_unlocked(self) -> LedgerHistory:
        """Verify exact metadata, contiguous revisions, canonical JSON, and hash chain."""
        try:
            meta = dict(self._connection.execute("SELECT key, value FROM metadata"))
            if set(meta) != _META or meta["format"] != _FORMAT:
                raise LedgerIntegrityError("ledger_metadata_invalid")
            raw_manifest = meta["manifest"]
            if hashlib.sha256(raw_manifest.encode()).hexdigest() != meta["manifest_hash"]:
                raise LedgerIntegrityError("ledger_manifest_digest_mismatch")
            manifest = _manifest_data(json.loads(raw_manifest))
            if _json(manifest) != raw_manifest:
                raise LedgerIntegrityError("ledger_manifest_not_canonical")
            rows = self._connection.execute("SELECT revision, event_json, previous_hash, event_hash FROM events ORDER BY revision")
            events, hashes, previous = [], [], _GENESIS
            for expected, (revision, raw, prior, digest) in enumerate(rows, start=1):
                event = _event_data(json.loads(raw))
                if revision != expected or prior != previous or _json(event) != raw or _digest(revision, prior, raw) != digest:
                    raise LedgerIntegrityError("ledger_event_chain_invalid")
                events.append(MappingProxyType(event))
                hashes.append(digest)
                previous = digest
            revision = int(meta["revision"])
            if revision != len(events) or meta["head_hash"] != previous:
                raise LedgerIntegrityError("ledger_revision_mismatch")
            return LedgerHistory(MappingProxyType(manifest), revision, tuple(events), tuple(hashes))
        except ExperimentBudgetStoreError:
            raise
        except (sqlite3.Error, ValueError, TypeError, json.JSONDecodeError, InvalidOperation) as exc:
            raise LedgerIntegrityError("ledger_contents_invalid") from exc
