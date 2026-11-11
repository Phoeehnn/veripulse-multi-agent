"""Run the frozen M6b slots through an explicitly offline executor."""

from collections.abc import Callable, Mapping
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from experiments.archive.m6b_readiness import m6b_preflight
from experiments.archive import joint_demo
from experiments.archive.m6b_readiness import PROMPT, SELECTION, _catalog

MockExecutor = Callable[[dict[str, Any]], Mapping[str, Any]]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish the current mock-run summary."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _run_slot(root: Path, selection: dict[str, Any], item: dict[str, Any],
              preflight: dict[str, Any], execute: MockExecutor) -> dict[str, Any]:
    """Run one mock slot only while the captured conditions remain unchanged."""
    executor_called = False
    try:
        if not _conditions_unchanged(root, selection, preflight["conditions"]):
            return {"status": "failed", "reason": "m6b_conditions_changed_before_slot", "executor_called": False}
        executor_called = True
        slot = dict(execute(item))
        if slot.get("status") != "mock_slot_scheduled":
            slot = {"status": "failed", "reason": slot.get("reason", "mock_executor_status_invalid"),
                "executor_called": True}
        if not _conditions_unchanged(root, selection, preflight["conditions"]):
            return {"status": "failed", "reason": "m6b_conditions_changed_after_slot", "executor_called": True}
        slot["executor_called"] = True
        return slot
    except Exception as exc:
        return {"status": "failed", "reason": type(exc).__name__, "executor_called": executor_called}


def _conditions_unchanged(root: Path, selection: dict[str, Any],
                          expected: dict[str, str]) -> bool:
    """Recheck frozen file, source, model and runtime fingerprints without rescanning history."""
    from src.yuqing.settings import get_cost_config, get_settings

    runtime = get_settings()
    current: dict[str, str] = {}
    for name in expected:
        if name == "ced_catalog_sha256":
            current[name] = joint_demo._digest(_catalog(root / "data/raw/CED_Dataset"))
        elif name == SELECTION:
            path = root / SELECTION
            if not path.is_file():
                return False
            current[name] = joint_demo._digest(json.loads(path.read_bytes()))
        elif name == PROMPT:
            current[name] = joint_demo._sha(root / name)
        elif name == "model":
            current[name] = runtime.deepseek_model
        elif name == "base_url_sha256":
            current[name] = sha256(runtime.deepseek_base_url.encode()).hexdigest()
        elif name == "prices_sha256":
            current[name] = joint_demo._digest(get_cost_config().model_dump())
        elif name == "timeout":
            current[name] = str(runtime.llm_timeout_s)
        elif name == "prompt_dev_permission":
            current[name] = "author_approved_exact_sha256"
        elif name == "input_archive_sha256":
            continue
        else:
            path = root / name
            if not path.is_file():
                return False
            current[name] = joint_demo._sha(path)
    return joint_demo._digest(current) == joint_demo._digest(expected)


def run_m6b_mock(root: Path, selection: dict[str, Any], execute: MockExecutor,
                 *, output_dir: Path | None = None) -> dict[str, Any]:
    """Verify fixed-slot scheduling; injected execution must remain offline."""
    preflight = m6b_preflight(root, selection)
    if preflight["provider_calls"] or preflight["formal_wallets_created"] or preflight["claims_created"]:
        raise ValueError("m6b_mock_preflight_has_side_effects")
    target = None
    if output_dir is not None:
        target = output_dir.resolve()
        traces = (root / "traces").resolve()
        if not target.is_relative_to(traces) or target == traces:
            raise ValueError("m6b_mock_output_outside_traces")
        target.mkdir(parents=True, exist_ok=False)
    run_id = uuid4().hex
    result: dict[str, Any] = {"run_id": run_id, "mode": "mock_only",
        "execution_scope": "scheduler_only", "selection_sha256": preflight["selection_sha256"],
        "conditions_sha256": preflight["conditions_sha256"], "planned": 10, "executed": 0,
        "stopped": False, "paid_ready": False, "provider_calls": 0,
        "formal_wallets_created": 0, "claims_created": 0, "rows": []}
    if target is not None:
        _write_json(target / "run.json", result)
    for item in selection["samples"]:
        slot = _run_slot(root, selection, item, preflight, execute)
        slot.update(order=item["order"], sample_id=item["sample_id"])
        slot.setdefault("graph_invoked", False)
        slot.setdefault("case_completed", False)
        result["rows"].append(slot)
        result["executed"] += int(slot["executor_called"])
        failed = slot["status"] == "failed"
        result["stopped"] = failed
        if target is not None:
            with (target / "rows.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(slot, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
            _write_json(target / "run.json", result)
        if failed:
            break
    result["rows_sha256"] = sha256(json.dumps(result["rows"], ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if target is not None:
        _write_json(target / "run.json", result)
    return result
