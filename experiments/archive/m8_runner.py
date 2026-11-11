"""Build isolated M8 experiment groups over one shared wallet."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from src.yuqing.agents.propagation import PropagationResult
from experiments.archive.m8_cache import (M8CacheIdentity, M8RunBinding, M8VerifiedSnapshot,
    _trace_root_valid, _wallet_identity, _wallet_ready, save_shared_decision)
from src.yuqing.graph.fork import fork_case_state
from src.yuqing.graph.runtime import CaseRuntimeContext
from src.yuqing.settings import PROJECT_ROOT


class M8GroupBranch:
    """一个独立组的图状态、运行上下文、Trace 根和共享总钱包引用。"""

    def __init__(self, group_id: Literal["B4-nf", "B4-fixed"], state: dict[str, Any],
                 context: CaseRuntimeContext, wallet: object, trace_root: Path,
                 snapshot: M8VerifiedSnapshot) -> None:
        """保留隔离对象，钱包仅共享唯一实验实例。"""
        self.group_id, self.state, self.context = group_id, state, context
        self.wallet, self.trace_root = wallet, trace_root
        self.snapshot = snapshot


def fork_m8_group(snapshot: M8VerifiedSnapshot, group_id: Literal["B4-nf", "B4-fixed"],
                  wallet: object, trace_root: Path) -> M8GroupBranch:
    """从相同已核验首轮状态深拷贝组状态，隔离上下文并共享总钱包。"""
    status = wallet.snapshot() if _wallet_ready(wallet) else None
    if (status is None or status.halted or status.pending_requests or status.unknown_requests
            or not _trace_root_valid(trace_root, group_id)
            or _wallet_identity(wallet) != (snapshot.wallet_experiment_id, snapshot.wallet_path)):
        raise ValueError("m8_group_resource_invalid")
    state = fork_case_state(snapshot.state)
    context = CaseRuntimeContext(snapshot.identity.case_id, snapshot.identity.sample_id,
                                 thread_id=state["thread_id"])
    if snapshot.fact_artifact is not None:
        context.store_fact(1, snapshot.fact_artifact)
    if snapshot.propagation_result is not None:
        context.store_propagation(PropagationResult.model_validate_json(
            snapshot.propagation_result.model_dump_json(), strict=True))
    return M8GroupBranch(group_id, state, context, wallet, trace_root.resolve(), snapshot)


def assemble_m8_group(composition: object, branch: M8GroupBranch, *,
                      external_calls_authorized: bool = False) -> object:
    """接入已有案件组合根；只跳过已缓存首轮，其他生产节点照常运行。"""
    inputs = getattr(composition, "inputs", None)
    if (inputs is None or inputs.group_id != branch.group_id or inputs.variant != branch.group_id
            or inputs.wallet is not branch.wallet
            or inputs.trace_root.resolve() != branch.trace_root
            or branch.state.get("thread_id") != branch.context.thread_id):
        raise ValueError("m8_group_composition_mismatch")
    composition.context = branch.context
    return composition.assemble(initial_state=deepcopy(branch.state),
        m8_cached_first_round=branch, external_calls_authorized=external_calls_authorized)


def assemble_m8_shared_prefix(composition: object, *,
                               external_calls_authorized: bool = False) -> object:
    """Build the one shared expert→merge→arbiter prefix used only by the M8 paired experiment."""
    inputs = getattr(composition, "inputs", None)
    if (inputs is None or inputs.group_id != "shared-first-round"
            or inputs.variant != "shared-first-round"
            or not _trace_root_valid(inputs.trace_root, "shared-first-round")):
        raise ValueError("m8_shared_prefix_composition_invalid")
    state = {"case_id": inputs.case_id,
        "sample": inputs.sample.model_dump(mode="json"),
        "ticket": {"claim": inputs.sample.claim},
        "_m8_capture_shared_decision": True}
    return composition.assemble(initial_state=state,
        external_calls_authorized=external_calls_authorized)


def save_m8_shared_snapshot(path: Path, identity: M8CacheIdentity, state: dict[str, Any],
                            composition: object) -> str:
    """将实际共同前缀结果和同一钱包请求绑定后写入只读缓存。"""
    from src.yuqing.collab.arbiter_budget_v2 import _request_states

    inputs, context = getattr(composition, "inputs", None), getattr(composition, "context", None)
    runs, wallet = getattr(composition, "runs", None), getattr(inputs, "wallet", None)
    if (inputs is None or context is None or not isinstance(runs, dict)
            or inputs.group_id != "shared-first-round" or inputs.variant != "shared-first-round"
            or inputs.sample.sample_id != identity.sample_id or inputs.case_id != identity.case_id):
        raise ValueError("m8_shared_prefix_identity_mismatch")
    recovered, _ = _request_states(wallet)
    rows = [(request_id, item) for request_id, item in recovered.requests.items()
        if item.identity.get("sample_id") == identity.sample_id
        and item.identity.get("group_id") == "shared-first-round"
        and item.identity.get("variant") == "shared-first-round"]
    bindings = []
    trace_base = (PROJECT_ROOT / "traces").resolve(strict=True)
    for run_id, binding in runs.items():
        request_ids = tuple(sorted(key for key, item in recovered.requests.items()
            if item.identity.get("run_id") == run_id))
        matches = tuple(inputs.trace_root.rglob(f"{run_id}.jsonl"))
        if len(matches) != 1:
            raise ValueError("m8_shared_prefix_trace_missing")
        trace_path = matches[0].resolve(strict=True)
        if not trace_path.is_relative_to(trace_base):
            raise ValueError("m8_shared_prefix_trace_outside_root")
        bindings.append(M8RunBinding(run_id=run_id,
            trace_path=trace_path.relative_to(trace_base).as_posix(), role=binding.role,
            decision_round=binding.decision_round, request_ids=request_ids,
            target=binding.target, shared_prefix=True))
    bindings = tuple(bindings)
    request_ids = tuple(sorted(request_id for request_id, _item in rows))
    snapshot_state = deepcopy(state)
    snapshot_state.pop("_m8_capture_shared_decision", None)
    if any(not item.shared_prefix for item in bindings):
        raise ValueError("m8_shared_prefix_binding_invalid")
    return save_shared_decision(path, identity, snapshot_state, context, wallet,
                                request_ids, bindings)


def map_ced_shared_result(state: dict[str, Any], execution_id: str) -> dict[str, Any]:
    """保存CED唯一结果，并只用组引用表达两组相同，不复制计分观察。"""
    if not execution_id or state.get("coverage") != "none":
        raise ValueError("m8_ced_shared_result_invalid")
    return {"execution_id": execution_id, "shared_result": deepcopy(state),
        "logical_groups": ("B4-nf", "B4-fixed"), "metric_observation_count": 1}


def group_wallet_costs(wallet: object, group_ids: tuple[str, ...]) -> dict[str, dict[str, str | int]]:
    """按组汇总已知费用与保留额，共享首轮作为独立一组只计一次。"""
    from src.yuqing.collab.arbiter_budget_v2 import _request_states

    recovered, _ = _request_states(wallet)
    names = (*group_ids, "shared-first-round")
    if len(set(names)) != len(names):
        raise ValueError("m8_wallet_group_ids_duplicate")
    totals = {group: {"known_cost_cny": Decimal(0), "pending_reserved_cny": Decimal(0),
        "unknown_reserved_cny": Decimal(0), "request_count": 0} for group in names}
    for request in recovered.requests.values():
        group = request.identity.get("group_id")
        if group not in totals:
            raise ValueError("m8_wallet_group_identity_unknown")
        totals[group]["request_count"] += 1
        if request.state == "settled" and request.actual is not None:
            totals[group]["known_cost_cny"] += request.actual
        elif request.state in {"reserved", "dispatched"}:
            totals[group]["pending_reserved_cny"] += request.reserved
        elif request.state == "unknown":
            totals[group]["unknown_reserved_cny"] += request.reserved
    return {group: {key: str(value) if isinstance(value, Decimal) else value
        for key, value in totals[group].items()} for group in names}

