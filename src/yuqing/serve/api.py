"""Three minimal HTTP endpoints for demo case submission and human review."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Literal, NoReturn
from uuid import uuid4

from fastapi import APIRouter, FastAPI, HTTPException, Request
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.yuqing.graph.build import resume_case_graph
from src.yuqing.graph.runtime import (CaseRuntimeAssembly, _case_decision_id,
    _review_choices, _validate_human_review)
from src.yuqing.serve.cases import CaseRegistry, build_demo_assembly
from src.yuqing.serve.store import TicketStore
from src.yuqing.settings import get_graph_config, get_settings


class TicketRequest(BaseModel):
    """Validated content submitted to the explicitly Mock-backed M7 demo."""

    model_config = ConfigDict(extra="forbid")
    content_id: str = Field(min_length=1)
    text: str = Field(min_length=1, pattern=r"\S")
    source_platform: str | None = None
    report_count: int | None = Field(default=None, ge=0)
    sample_id: str | None = None
    scenario: Literal["general", "content_platform"] | None = None

    @field_validator("content_id", "text")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Reject strings that contain only whitespace."""
        if not value.strip():
            raise ValueError("must_not_be_blank")
        return value.strip()


class ApprovalRequest(BaseModel):
    """A human decision attached to the current case and decision snapshot."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject", "modify"]
    final_disposition: str | None = None
    reason: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)

    @field_validator("reason", "reviewer")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Require a visible reason and reviewer label in the demo API."""
        if not value.strip():
            raise ValueError("must_not_be_blank")
        return value.strip()


AssemblyFactory = Callable[[str, dict[str, Any], BaseCheckpointSaver | None], CaseRuntimeAssembly]
_MOCK_GRAPH_CONTRACT = "x5-mock-human-gate-v1"
router = APIRouter()


def create_app(database_path: Path | None = None, *, registry: CaseRegistry | None = None,
               assembly_factory: AssemblyFactory = build_demo_assembly,
               checkpointer_path: Path | None = None) -> FastAPI:
    """Build an offline demo API with injectable SQLite path and graph factory."""
    store = TicketStore(database_path or get_settings().review_db_path)
    cases = registry or CaseRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Initialize ticket storage and hold the optional checkpoint connection."""
        store.initialize(get_settings().case_processing_expiry_minutes)
        graph_config = get_graph_config()
        _app.state.checkpointer_mode = graph_config.checkpointer
        _app.state.checkpointer = None
        _app.state.recovery_config_sha256 = _config_fingerprint()
        connection: sqlite3.Connection | None = None
        try:
            if graph_config.checkpointer == "sqlite":
                path = checkpointer_path or get_settings().checkpoints_db_path
                if checkpointer_path is None and path.resolve().drive != get_settings().project_root.resolve().drive:
                    raise ValueError("checkpoints_path_must_use_project_drive")
                path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(path, check_same_thread=False)
                saver = SqliteSaver(connection, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
                saver.setup()
                _app.state.checkpointer = saver
            yield
        finally:
            if connection is not None:
                connection.close()

    app = FastAPI(title="项目 A 工单演示 API（Mock）",
        description="仅用离线 Mock 演示工单和人工复核流程；reviewer 是演示字段，未实现身份认证。",
        lifespan=lifespan)
    app.state.ticket_store, app.state.case_registry = store, cases
    app.state.assembly_factory = assembly_factory
    app.include_router(router)
    return app


@router.post("/tickets", status_code=201)
def submit_ticket(request: TicketRequest, app_request: Request) -> dict[str, Any]:
    """Persist a unique ticket, execute the Mock graph, and record its terminal state."""
    store, cases = app_request.app.state.ticket_store, app_request.app.state.case_registry
    case_id, payload = uuid4().hex, request.model_dump(mode="json")
    try:
        assembly = app_request.app.state.assembly_factory(
            case_id, payload, app_request.app.state.checkpointer)
        cases.add(case_id, assembly)
        store.create_ticket(case_id, assembly.context.thread_id, payload,
            checkpoint_mode=app_request.app.state.checkpointer_mode,
            graph_contract=_MOCK_GRAPH_CONTRACT,
            config_sha256=app_request.app.state.recovery_config_sha256,
            input_sha256=_input_fingerprint(payload))
        result = assembly.invoke()
        status, data = _initial_result(assembly, result)
        _persist_initial_result(case_id, status, data, result, assembly, store)
        return {"case_id": case_id, "thread_id": assembly.context.thread_id,
            "status": status, "mode": "mock_demo", **data}
    except Exception as exc:
        if store.get_ticket(case_id) is not None:
            store.fail(case_id, "failed", type(exc).__name__)
        raise HTTPException(status_code=500, detail="case_execution_failed") from exc


@router.get("/cases/{case_id}")
def get_case(case_id: str, request: Request) -> dict[str, Any]:
    """Return persisted case dimensions, ledger, decision state, and brief."""
    row = request.app.state.ticket_store.get_ticket(case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="case_not_found")
    return _case_response(row)


@router.post("/cases/{case_id}/approval")
def approve_case(case_id: str, body: ApprovalRequest, request: Request) -> dict[str, Any]:
    """Validate a real pause, atomically claim it, then resume exactly one graph call."""
    store, cases = request.app.state.ticket_store, request.app.state.case_registry
    row = store.get_ticket(case_id)
    if row is None:
        raise HTTPException(status_code=404, detail="case_not_found")
    if row["status"] != "awaiting_approval":
        raise HTTPException(status_code=409, detail={"status": row["status"]})
    assembly = cases.get(case_id)
    if assembly is None:
        if row["checkpoint_mode"] != "sqlite":
            _expire_waiting_or_report_current(store, case_id, "memory_checkpoint_unavailable")
        _validate_recovery_row(row, request.app.state)
        assembly = cases.get_or_create(case_id, lambda: _build_restored_assembly(
            request, row))
    _validate_recovery_row(row, request.app.state)
    _validate_recovery_assembly(row, assembly)
    expected_decision = row.get("pending_decision_id") if row["checkpoint_mode"] == "sqlite" else None
    snapshot, interrupt_data = _checked_pending_review(assembly, case_id, store,
        expected_decision, row["input"] if expected_decision else None)
    review = _validated_review(body, snapshot.values, interrupt_data, case_id)
    if not store.claim_approval(case_id):
        raise HTTPException(status_code=409, detail="approval_already_claimed")
    return _resume_claimed(case_id, assembly, review, store)


def _persist_initial_result(case_id: str, status: str, data: dict[str, Any],
                            result: dict[str, Any], assembly: CaseRuntimeAssembly,
                            store: TicketStore) -> None:
    """Save one initial graph outcome under its matching ticket status."""
    if status == "awaiting_approval":
        config = {"configurable": {"thread_id": assembly.context.thread_id}}
        snapshot = assembly.graph.get_state(config)
        store.save_waiting(case_id, data, _case_decision_id(snapshot.values))
    elif status == "completed":
        store.finish_without_decision(case_id, data)
    else:
        store.fail(case_id, "failed", _failure_type(result))


def _checked_pending_review(assembly: CaseRuntimeAssembly, case_id: str,
                            store: TicketStore,
                            expected_decision_id: str | None = None,
                            expected_input: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    """Expire a missing checkpoint and reject mismatched graph identity before claim."""
    try:
        snapshot, interrupt_data = _pending_review(assembly)
    except HTTPException as exc:
        if exc.detail == {"status": "expired"}:
            _expire_waiting_or_report_current(store, case_id, "checkpoint_unavailable")
        raise
    except ValueError:
        _expire_waiting_or_report_current(store, case_id, "checkpoint_corrupt")
    except sqlite3.DatabaseError as exc:
        if _is_sqlite_corruption(exc):
            _expire_waiting_or_report_current(store, case_id, "checkpoint_corrupt")
        raise HTTPException(status_code=503, detail="checkpoint_storage_unavailable") from exc
    if interrupt_data.get("case_id") != case_id:
        raise HTTPException(status_code=409, detail="checkpoint_identity_mismatch")
    if interrupt_data.get("decision_id") != _case_decision_id(snapshot.values):
        raise HTTPException(status_code=409, detail="checkpoint_decision_mismatch")
    if expected_decision_id is not None and interrupt_data.get("decision_id") != expected_decision_id:
        raise HTTPException(status_code=409, detail="checkpoint_decision_mismatch")
    if expected_input is not None and (snapshot.values.get("case_id") != case_id
            or snapshot.values.get("ticket") != expected_input
            or snapshot.values.get("thread_id") != assembly.context.thread_id):
        raise HTTPException(status_code=409, detail="checkpoint_input_identity_mismatch")
    return snapshot, interrupt_data


def _input_fingerprint(ticket: dict[str, Any]) -> str:
    """Hash the exact normalized ticket fields used to build the Mock graph."""
    encoded = json.dumps(ticket, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _config_fingerprint() -> str:
    """Bind restoration to the exact non-secret project weights configuration."""
    return sha256(get_settings().weights_path.read_bytes()).hexdigest()


def _validate_recovery_row(row: dict[str, Any], app_state: Any) -> None:
    """Reject stale or cross-mode tickets before rebuilding or resuming graph state."""
    if row["checkpoint_mode"] != app_state.checkpointer_mode:
        raise HTTPException(status_code=409, detail="checkpoint_mode_mismatch")
    if row["checkpoint_mode"] == "sqlite" and (
            app_state.checkpointer is None
            or row["graph_contract"] != _MOCK_GRAPH_CONTRACT
            or row["config_sha256"] != app_state.recovery_config_sha256
            or row["input_sha256"] != _input_fingerprint(row["input"])
            or not row.get("pending_decision_id")):
        raise HTTPException(status_code=409, detail="checkpoint_recovery_metadata_mismatch")


def _validate_recovery_assembly(row: dict[str, Any], assembly: CaseRuntimeAssembly) -> None:
    """Require the rebuilt sidecar to use the ticket's original case and thread IDs."""
    if (assembly.context.case_id != row["case_id"]
            or assembly.context.thread_id != row["thread_id"]):
        raise HTTPException(status_code=409, detail="checkpoint_identity_mismatch")


def _is_sqlite_corruption(error: sqlite3.DatabaseError) -> bool:
    """Identify persistent database corruption without expiring tickets on temporary locks."""
    code = getattr(error, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}


def _expire_waiting_or_report_current(store: TicketStore, case_id: str, reason: str) -> NoReturn:
    """Use a conditional transition and report concurrent completion without overwriting it."""
    if store.expire_waiting(case_id, reason):
        raise HTTPException(status_code=409, detail={"status": "expired"})
    current = store.get_ticket(case_id)
    raise HTTPException(status_code=409, detail={"status": current["status"] if current else "missing"})


def _build_restored_assembly(request: Request, row: dict[str, Any]) -> CaseRuntimeAssembly:
    """Rebuild only the sidecar around the persisted thread; never invoke the first graph turn."""
    return request.app.state.assembly_factory(row["case_id"],
        row["input"] | {"thread_id": row["thread_id"]}, request.app.state.checkpointer)


def _validated_review(body: ApprovalRequest, state: dict[str, Any],
                      interrupt_data: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Bind the API choice to program-owned case and decision identities."""
    decision_id = interrupt_data.get("decision_id")
    try:
        return _validate_human_review(body.model_dump(mode="json") | {"case_id": case_id,
            "decision_id": decision_id}, state, str(decision_id),
            _review_choices(state.get("risk") or {}))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _pending_review(assembly: CaseRuntimeAssembly) -> tuple[Any, dict[str, Any]]:
    """Require the stored graph to be paused at its actual human gate."""
    config = {"configurable": {"thread_id": assembly.context.thread_id}}
    snapshot = assembly.graph.get_state(config)
    interrupts = [item for task in snapshot.tasks for item in task.interrupts]
    if snapshot.next != ("human_gate",) or len(interrupts) != 1:
        raise HTTPException(status_code=409, detail={"status": "expired"})
    value = interrupts[0].value
    if not isinstance(value, dict) or value.get("case_id") != assembly.context.case_id:
        raise HTTPException(status_code=409, detail="checkpoint_identity_mismatch")
    return snapshot, value


def _resume_claimed(case_id: str, assembly: CaseRuntimeAssembly, review: dict[str, Any],
                    store: TicketStore) -> dict[str, Any]:
    """Resume one claimed graph and persist either its decision or a non-replayable failure."""
    try:
        result = resume_case_graph(assembly.graph, assembly.context.thread_id, review)
        final = assembly.graph.get_state({"configurable": {"thread_id": assembly.context.thread_id}})
        if final.next or result.get("approval") not in {"approved", "rejected"}:
            raise RuntimeError("resume_did_not_reach_valid_terminal_state")
        data = _result_data(result, "completed")
        store.finish(case_id, review, data)
        return {"case_id": case_id, "thread_id": assembly.context.thread_id,
            "status": "completed", "mode": "mock_demo", **data}
    except Exception as exc:
        store.fail(case_id, "resume_failed", type(exc).__name__)
        raise HTTPException(status_code=500, detail={"status": "resume_failed"}) from exc


def _initial_result(assembly: CaseRuntimeAssembly, result: dict[str, Any]
                    ) -> tuple[str, dict[str, Any]]:
    """Classify a graph result as paused, completed, or failed from its checkpoint."""
    config = {"configurable": {"thread_id": assembly.context.thread_id}}
    snapshot = assembly.graph.get_state(config)
    if snapshot.next == ("human_gate",):
        return "awaiting_approval", _result_data(result, "awaiting_approval")
    if not snapshot.next and result.get("brief") is not None and not result.get("stop_reason"):
        return "completed", _result_data(result, "completed")
    return "failed", _result_data(result, "failed")


def _result_data(state: dict[str, Any], status: str) -> dict[str, Any]:
    """Expose the specified case summary without any evidence body text."""
    risk = state.get("risk") or {}
    return {"status": status, "risk": risk, "evidence_sufficiency": risk.get("evidence_sufficiency"),
        "issues": state.get("issues", []), "suggested_action": risk.get("disposition"),
        "approval": state.get("approval"), "review": state.get("review"),
        "alert": state.get("alert"), "brief": state.get("brief"),
        "failures": state.get("failures", [])}


def _failure_type(state: dict[str, Any]) -> str:
    """Record a stable stop reason without persisting exception text or secrets."""
    return str(state.get("stop_reason") or "initial_graph_failed")


def _case_response(row: dict[str, Any]) -> dict[str, Any]:
    """Format a stored case response and retain its explicit Mock marker."""
    return {**(row["result"] or {}), "case_id": row["case_id"], "thread_id": row["thread_id"],
        "status": row["status"], "mode": "mock_demo", "ticket": row["input"],
        "failure_reason": row["failure_reason"]}


app = create_app()
