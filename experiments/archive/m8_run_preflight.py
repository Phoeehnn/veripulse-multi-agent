"""Read-only preflight for the fixed M8 paired experiment."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from experiments.archive.m8_inputs import M8Inputs, load_m8_inputs
from src.yuqing.settings import PROJECT_ROOT

M8_CAP_CNY = Decimal("20")
_RUNTIME_FILES = (
    "experiments/archive/m8_cache.py", "experiments/archive/m8_cache_validation.py",
    "experiments/archive/m8_runner.py", "experiments/archive/m8_inputs.py",
    "experiments/archive/m8_run_preflight.py",
    "experiments/archive/m8_run_restart.py",
    "experiments/archive/m8_run_journal.py",
    "experiments/archive/m8_run_audit.py",
    "experiments/archive/m8_run_executor.py",
    "scripts/run_m8.py",
    "src/yuqing/graph/composition.py", "src/yuqing/graph/build.py",
    "src/yuqing/graph/runtime.py", "src/yuqing/graph/fact_adapter.py",
    "src/yuqing/graph/nodes.py", "src/yuqing/graph/state.py",
    "src/yuqing/graph/fact_selection.py", "src/yuqing/graph/fork.py",
    "src/yuqing/agents/arbiter_loop.py", "src/yuqing/agents/compliance.py",
    "src/yuqing/agents/arbiter.py", "src/yuqing/agents/fact_check.py",
    "src/yuqing/agents/propagation.py",
    "src/yuqing/collab/arbiter_budget_v2.py", "src/yuqing/collab/arbiter_material_v2.py",
    "src/yuqing/collab/dispatchability.py", "src/yuqing/collab/progress.py",
    "src/yuqing/collab/validate.py", "src/yuqing/collab/board.py",
    "src/yuqing/collab/fact_material.py", "src/yuqing/collab/fact_archive.py",
    "src/yuqing/collab/fact_delivery.py", "src/yuqing/collab/arbiter_view.py",
    "src/yuqing/harness/initial_evidence.py", "src/yuqing/harness/inherited_evidence.py",
    "src/yuqing/harness/fact_v2_contract.py", "src/yuqing/harness/loop.py",
    "src/yuqing/harness/policy.py", "src/yuqing/harness/tools.py",
    "src/yuqing/harness/verifier.py", "src/yuqing/harness/feedback.py",
    "src/yuqing/harness/ledger.py", "src/yuqing/harness/context.py",
    "src/yuqing/harness/budget.py", "src/yuqing/harness/trace.py",
    "src/yuqing/harness/fact_v2_messages.py", "src/yuqing/retrieval/bm25.py",
    "src/yuqing/retrieval/fusion.py", "src/yuqing/retrieval/rerank.py",
    "src/yuqing/retrieval/vector_store.py", "src/yuqing/retrieval/embed.py",
    "src/yuqing/retrieval/corpus.py", "src/yuqing/retrieval/search.py",
    "src/yuqing/report/brief.py", "src/yuqing/schemas.py", "src/yuqing/settings.py",
    "src/yuqing/eval/dual_budget.py", "src/yuqing/eval/experiment_budget.py",
    "src/yuqing/eval/experiment_budget_store.py",
    "src/yuqing/eval/experiment_budget_recovery.py",
    "src/yuqing/eval/persistent_experiment_budget.py",
    "src/yuqing/eval/experiment_session.py", "src/yuqing/eval/fact_v2_experiment.py",
    "src/yuqing/eval/fact_v2_conditions.py", "src/yuqing/eval/preflight.py",
    "experiments/archive/m5_checkpoint.py", "src/yuqing/features/propagation.py",
    "src/yuqing/ingest/chef.py", "src/yuqing/llm.py",
    "config/weights.yaml", "config/rules_compliance.yaml",
    "prompts/arbiter/contracts.json", "prompts/fact_check/v2_manifest.json",
    "prompts/arbiter/v2.md", "prompts/propagation/v1.md",
    "prompts/fact_check/investigator_v2.md",
)


@dataclass(frozen=True)
class M8ExecutionSlot:
    """One actual model execution; CED appears once and maps to both logical groups."""

    order: int
    sample_id: str
    source: str
    execution_group: Literal["B4-nf", "B4-fixed", "CED-shared"]
    logical_groups: tuple[str, ...]


@dataclass(frozen=True)
class M8PreflightReport:
    """Read-only readiness receipt; it never grants execution permission."""

    summary_path: str
    summary_sha256: str
    archive_sha256: str
    conditions_sha256: str
    source_counts: dict[str, int]
    actual_slot_count: int
    slot_plan_sha256: str
    max_run_cost_cny: str
    provider_calls: int = 0
    wallet_created: bool = False
    claim_written: bool = False
    execution_authorized: bool = False


def _canonical(value: object) -> bytes:
    """Encode stable JSON bytes for experiment condition fingerprints."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(raw: bytes) -> str:
    """Return lowercase SHA-256 for exact bytes."""
    return sha256(raw).hexdigest()


def _runtime_file_hashes(root: Path) -> dict[str, str]:
    """Fingerprint each explicitly listed file that can affect M8 execution."""
    return {name: _digest((root / name).read_bytes()) for name in _RUNTIME_FILES}


def build_m8_slot_plan(inputs: M8Inputs) -> tuple[M8ExecutionSlot, ...]:
    """Expand fixed inputs into the preregistered CHEF pairs and single CED runs."""
    if len(inputs.cases) != 100 or len({item.sample_id for item in inputs.cases}) != 100:
        raise ValueError("m8_run_input_identity_invalid")
    slots: list[M8ExecutionSlot] = []
    for case in inputs.cases:
        if case.source == "chef" and case.coverage == "full":
            groups = ("B4-nf", "B4-fixed") if case.order % 2 else ("B4-fixed", "B4-nf")
            slots.extend(M8ExecutionSlot(case.order, case.sample_id, case.source, group,
                (group,)) for group in groups)
        elif case.source == "ced" and case.coverage == "none":
            slots.append(M8ExecutionSlot(case.order, case.sample_id, case.source,
                "CED-shared", ("B4-nf", "B4-fixed")))
        else:
            raise ValueError("m8_run_source_coverage_invalid")
    if sum(item.source == "chef" for item in inputs.cases) != 60 or len(slots) != 160:
        raise ValueError("m8_run_slot_count_invalid")
    return tuple(slots)


def _runtime_conditions(root: Path, inputs: M8Inputs, slots: tuple[M8ExecutionSlot, ...]) -> dict[str, object]:
    """Validate licensed frozen runtime components and hash every run-affecting input."""
    from experiments.archive.m5_checkpoint import _validate_arbiter_permission
    from src.yuqing.eval.preflight import validate_propagation_startup, validate_runtime_config
    from src.yuqing.graph.fact_adapter import FactNodeAdapter
    from src.yuqing.settings import (get_collab_config, get_cost_config, get_graph_config,
        get_m5_case_budget_config, get_settings)

    validate_runtime_config()
    config = get_settings()
    if config.project_root.resolve() != root.resolve() or root.drive.upper() != "D:":
        raise ValueError("m8_run_project_root_invalid")
    graph = get_graph_config()
    collaboration = get_collab_config()
    if (graph.fact_node_version != "v2" or collaboration.max_actions != 1
            or collaboration.targets != ["fact"] or collaboration.review_enabled
            or collaboration.generic_followup):
        raise ValueError("m8_run_frozen_graph_config_mismatch")
    if get_cost_config().input_per_1m is None or get_cost_config().output_per_1m is None:
        raise ValueError("m8_run_prices_unknown")
    FactNodeAdapter(graph.fact_node_version, root=root)
    validate_propagation_startup(root / "prompts/propagation/v1.md", offline=True)
    _validate_arbiter_permission(root)
    files = _runtime_file_hashes(root)
    return {"input_archive_sha256": inputs.archive_sha256,
        "input_sources": inputs.source_sha256, "runtime_files": files,
        "model": config.deepseek_model,
        "base_url_sha256": _digest(config.deepseek_base_url.encode("utf-8")),
        "timeout_s": config.llm_timeout_s, "thinking": config.llm_thinking,
        "graph": graph.model_dump(mode="json"),
        "collab": collaboration.model_dump(mode="json"),
        "cost": get_cost_config().model_dump(),
        "case_budget": get_m5_case_budget_config().model_dump(),
        "slot_plan_sha256": _digest(_canonical([slot.__dict__ for slot in slots]))}


def preflight_m8_run(root: Path, summary_path: Path, summary_sha256: str,
                     max_run_cost_cny: Decimal = M8_CAP_CNY) -> M8PreflightReport:
    """Verify the fixed M8 archive and runtime without creating experiment side effects."""
    if max_run_cost_cny <= 0 or max_run_cost_cny > M8_CAP_CNY:
        raise ValueError("m8_run_cap_exceeds_authorized_ceiling")
    inputs = load_m8_inputs(root, summary_path, summary_sha256)
    slots = build_m8_slot_plan(inputs)
    conditions = _runtime_conditions(root, inputs, slots)
    counts = {source: sum(case.source == source for case in inputs.cases)
        for source in ("chef", "ced")}
    return M8PreflightReport(summary_path=summary_path.resolve().relative_to(root.resolve()).as_posix(),
        summary_sha256=summary_sha256, archive_sha256=inputs.archive_sha256,
        conditions_sha256=_digest(_canonical(conditions)), source_counts=counts,
        actual_slot_count=len(slots), slot_plan_sha256=conditions["slot_plan_sha256"],
        max_run_cost_cny=str(max_run_cost_cny))
