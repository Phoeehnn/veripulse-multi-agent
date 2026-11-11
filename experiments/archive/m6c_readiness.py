"""M6c固定八案的离线选择、来源闭包与启动预检。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from experiments.archive import joint_demo as joint
from experiments.archive.m5_inputs import _initial_rows
from src.yuqing.schemas import AgentSpec, Sample
from src.yuqing.settings import (PROJECT_ROOT, EvaluationConfig, get_cost_config, get_evaluation_config,
    get_m6c_run_budget_config, get_propagation_tools_config, get_settings, yaml)

SELECTION_PATH = "docs/cards/M6c-selection-v1.json"
CLAIM_PATH = "traces/m6c-fixed-eight.claim"
JOINT_CASES = ("407", "2508", "658")
DEV_PATHS = ("eval/dev_extra_60.jsonl",)
INITIAL_PATH = "eval/dev90_initial_evidence_v2.jsonl"
EXCLUDE_PATHS = ("eval/test_60.jsonl", "eval/label_pool_100.jsonl",
    "docs/review_history/风险补充50条_清单_2026-10-18.jsonl",
    "docs/cards/M6b-C_selection.json", "docs/cards/M5-B-selection-v1.json")
M9_DEMO_IDS = frozenset({"chef:dev:680", "chef:test:598", "ced:rumor:1905",
    "chef:test:658", "chef:test:26", "chef:dev:826", "ced:rumor:334"})
CASE_CAP, RUN_CAP, SEED = Decimal("2.06"), Decimal("16.48"), 42
M6B_PROMPT = "experiments/archive/prompts/propagation/v1_m6b.draft.md"
M6B_PROMPT_SHA = "86156d7a8d0ef4a5eeadf5774bd40c404cfca0821cd7f107f1f43c9f849dcf49"
M6C_PROMPT = "prompts/propagation/v1_m6c.md"


@dataclass(frozen=True)
class PreparedM6cJoint:
    """Joint source data validated independently from the unused v1 default prompt."""

    root: Path
    sample: Sample
    chunks: tuple[Any, ...]
    store: Path
    conditions: dict[str, str]
    case: str


@dataclass(frozen=True)
class M6cCase:
    """A selected slot with source identity and an optional verified joint bundle."""

    order: int
    group: str
    sample: Sample
    joint_input: PreparedM6cJoint | None = None


def _rank(sample_id: str) -> str:
    """Rank candidate identities reproducibly without inspecting labels or outputs."""
    return sha256((str(SEED) + "|" + sample_id).encode()).hexdigest()


def _sha(path: Path) -> str:
    """Hash exact source bytes."""
    return sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    """Hash canonical JSON independent of insertion order."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()
    return sha256(raw).hexdigest()


def _ids(path: Path) -> set[str]:
    """Read sample identities from a JSONL exclusion source."""
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(raw)
        rows = value.get("samples", value.get("selected", [])) if isinstance(value, dict) else value
        return {row["sample_id"] for row in rows if row.get("sample_id")}
    return {row["sample_id"] for line in raw.splitlines()
        if line.strip() and (row := json.loads(line)).get("sample_id")}


def _joint_input(root: Path, case: str) -> tuple[dict[str, Any], PreparedM6cJoint]:
    """Validate joint documents, author-reviewed sources, raw data, and the stored evidence."""
    material, sample = joint.load_joint_material(root, case)
    chunks = joint.material_chunks(material)
    store = joint._validate_joint_store(root, material, chunks, case)
    files = joint._case_files(case)
    inventory = json.loads((root / files.inventory).read_bytes())
    paths = {files.material, files.inventory, inventory["original_path"],
        inventory["propagation_path"], "config/weights.yaml", "experiments/archive/joint_demo.py"}
    if case != "658":
        paths.add(inventory["shortlist_path"])
        paths.update(item["path"] for item in inventory["protected_pool_checks"])
    conditions = {name: _sha(root / name) for name in sorted(paths)}
    conditions.update({f"store/{case}/{name}": _sha(store / name)
        for name in ("READY", "corpus.json", "initial.json")})
    safe_sample = sample.model_copy(update={"gold_veracity": None, "gold_risk": None})
    row = {"sample_id": safe_sample.sample_id, "group": "joint_reuse",
        "joint_case": case, "coverage": "demo",
        "sample_sha256": _digest(safe_sample.model_dump(mode="json"))}
    prepared = PreparedM6cJoint(root, safe_sample, tuple(chunks), store, conditions, case)
    return row, prepared


def _candidate_rows(root: Path) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]], dict[str, str]]:
    """Audit every CHEF dev candidate and retain only eligible, evidenced rows."""
    excluded = set().union(*(_ids(root / path) for path in EXCLUDE_PATHS)) | M9_DEMO_IDS
    sources = {path: _sha(root / path) for path in (*DEV_PATHS, INITIAL_PATH, *EXCLUDE_PATHS)}
    initials = _initial_rows(root)
    questions = {row["sample_id"]: row for name in DEV_PATHS
        for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()
        for row in [json.loads(line)]}
    candidates, eligible = [], []
    joint_ids = {joint._case_files(case).sample_id for case in JOINT_CASES}
    for sample_id, row in questions.items():
        reasons = [reason for match, reason in ((not sample_id.startswith("chef:dev:"), "not_chef_dev"),
            (sample_id in excluded, "test_label_pool_manual_or_demo"),
            (sample_id in joint_ids, "already_in_joint_set"),
            (sample_id not in initials, "initial_evidence_manifest_missing")) if match]
        record = {"sample_id": sample_id, "rank_sha256": _rank(sample_id),
            "question_sha256": _digest(row), "reasons": reasons}
        if sample_id in initials:
            record["initial_manifest_sha256"] = _digest(initials[sample_id].model_dump(mode="json"))
        candidates.append(record)
        if not reasons:
            eligible.append((record["rank_sha256"], sample_id, row))
    return candidates, sorted(eligible), sources


def build_m6c_selection(root: Path) -> dict[str, Any]:
    """Build three fixed joint cases and five hash-ranked CHEF development cases."""
    candidates, eligible, sources = _candidate_rows(root)
    if len(eligible) < 5:
        raise ValueError("m6c_chef_dev_sample_shortage")
    chosen = {sample_id for _, sample_id, _ in eligible[:5]}
    for candidate in candidates:
        if candidate["sample_id"] in chosen:
            candidate["selected"] = True
        elif not candidate["reasons"]:
            candidate["reasons"] = ["eligible_hash_rank_outside_top_five"]
    samples, prepared_joint = [], []
    for case in JOINT_CASES:
        row, prepared = _joint_input(root, case)
        row["order"] = len(samples) + 1
        samples.append(row)
        prepared_joint.append(prepared)
        sources.update(prepared.conditions)
    by_id = {row["sample_id"]: row for row in candidates}
    for _, sample_id, question in eligible[:5]:
        sample = Sample(sample_id=sample_id, source="chef", claim=question["claim"])
        samples.append({"order": len(samples) + 1, "sample_id": sample_id,
            "group": "chef_dev", "coverage": "full", "question_sha256": _digest(question),
            "sample_sha256": _digest(sample.model_dump(mode="json")),
            "initial_manifest_sha256": by_id[sample_id]["initial_manifest_sha256"]})
    sources["config/weights.yaml"] = _sha(root / "config/weights.yaml")
    return {"version": "m6c-fixed-eight-v1", "seed": SEED,
        "selection_rule": "SHA256(seed|sample_id) over CHEF dev_extra_60 rows with frozen initial evidence",
        "prior_use_note": "Dev sample, not unseen: previously included in dev90; excludes M5-B/M6b collaboration samples, M9 demos, label pool, test set, and missing initial manifests.",
        "samples": samples, "candidate_audit": sorted(candidates, key=lambda row: row["sample_id"]),
        "source_sha256": dict(sorted(sources.items())), "paid_ready": False}


def prepare_m6c_cases(root: Path, selection: dict[str, Any]) -> tuple[M6cCase, ...]:
    """Rebuild the frozen set from source bytes and load its label-free inputs."""
    if _digest(build_m6c_selection(root)) != _digest(selection):
        raise ValueError("m6c_selection_or_source_drift")
    prepared_joint = {case: _joint_input(root, case)[1] for case in JOINT_CASES}
    questions = {row["sample_id"]: row for name in DEV_PATHS
        for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip()
        for row in [json.loads(line)]}
    cases = []
    for row in selection["samples"]:
        joint_input = prepared_joint.get(row.get("joint_case"))
        sample = joint_input.sample if joint_input else Sample(sample_id=row["sample_id"],
            source="chef", claim=questions[row["sample_id"]]["claim"])
        sample = sample.model_copy(update={"gold_veracity": None, "gold_risk": None})
        if sample.sample_id != row["sample_id"] or _digest(sample.model_dump(mode="json")) != row["sample_sha256"]:
            raise ValueError("m6c_loaded_sample_identity_or_content_mismatch")
        cases.append(M6cCase(row["order"], row["group"], sample, joint_input))
    return tuple(cases)


def _prompt_contracts(root: Path) -> dict[str, str]:
    """Bind the exact development-licensed M6b and M6c prompt contracts and request schemas."""
    from src.yuqing.agents.propagation import register_propagation_tools
    from experiments.archive import m6b_readiness, m6c_review_prompt
    from src.yuqing.harness.tools import ToolRegistry
    from src.yuqing import llm
    from src.yuqing.schemas import PropagationReport, ReviewResult

    base = root / m6b_readiness.PROMPT
    review = m6c_review_prompt.validate_m6c_review_prompt(offline=True)
    if (_sha(base) != m6b_readiness.APPROVED_PROMPT_SHA256
            or M6B_PROMPT_SHA != m6b_readiness.APPROVED_PROMPT_SHA256):
        raise ValueError("m6c_first_round_prompt_bytes_unapproved")
    normal_registry = ToolRegistry()
    register_propagation_tools(normal_registry, tools_config=get_propagation_tools_config(), contract_version="v1.1")
    normal = llm._tool_definitions([normal_registry.get(name) for name in normal_registry._tools], False,
        submission_tool="submit_propagation_report")
    review_registry = ToolRegistry()
    register_propagation_tools(review_registry, tools_config=get_propagation_tools_config(),
        contract_version="v1.1", review_mode=True)
    review_tools = llm._tool_definitions([review_registry.get(name) for name in review_registry._tools], False,
        submission_tool="submit_review")
    return {"first_prompt_sha256": _sha(base), "review_prompt_sha256": review.prompt_sha256,
        "first_tools_sha256": _digest(normal), "review_tools_sha256": _digest(review_tools),
        "first_schema_sha256": _digest(PropagationReport.model_json_schema()),
        "review_schema_sha256": _digest(ReviewResult.model_json_schema())}


def m6c_preflight(root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    """Reject source, prompt, runtime, and envelope drift before run side effects."""
    path = root / SELECTION_PATH
    if not path.is_file() or json.loads(path.read_bytes()) != selection:
        raise ValueError("m6c_selection_file_mismatch")
    cases = prepare_m6c_cases(root, selection)
    budget = get_m6c_run_budget_config()
    case_cap = Decimal(str(budget.case_budget.max_cost_cny))
    if case_cap != CASE_CAP or RUN_CAP != case_cap * len(cases):
        raise ValueError("m6c_budget_envelope_mismatch")
    prompts = _prompt_contracts(root)
    paths = set(joint._runtime_code_files(root)) | {"experiments/archive/m6c_readiness.py",
        "experiments/archive/m6c_live.py", "experiments/archive/m6c_execution.py",
        "scripts/prepare_m6c.py", SELECTION_PATH, M6B_PROMPT, "prompts/propagation/contracts.json",
        M6C_PROMPT, "prompts/propagation/review_contracts.json", "config/weights.yaml",
        "docs/decisions.md"}
    files = {name: _sha(root / name) for name in sorted(paths)}
    conditions = _runtime_conditions(root, selection, budget, prompts, files)
    return {"mode": "offline_preflight", "planned": len(cases), "cases": [
        {"order": item.order, "sample_id": item.sample.sample_id, "group": item.group,
         "case_cap_cny": str(case_cap)} for item in cases],
        "selection_sha256": _digest(selection), "conditions": conditions,
        "conditions_sha256": _digest(conditions), "case_cap_cny": str(case_cap),
        "run_cap_cny": str(RUN_CAP), "theoretical_worst_case_cny": str(RUN_CAP),
        "provider_calls": 0, "wallets_created": 0, "claims_created": 0,
        "paid_ready": False, "blockers": ["Sol_paid_preflight_review_required",
            "author_paid_material_send_and_run_authorization_required"]}


def m6c_wallet_evaluation_config() -> EvaluationConfig:
    """Give only the fixed-eight wallet the author-approved envelope, leaving global E2 limits intact."""
    config = get_evaluation_config()
    if RUN_CAP != CASE_CAP * 8:
        raise ValueError("m6c_wallet_envelope_invalid")
    return config.model_copy(update={"max_run_cost_cny": float(RUN_CAP)})


def _runtime_conditions(root: Path, selection: dict[str, Any], budget: Any,
                        prompts: dict[str, str], files: dict[str, str]) -> dict[str, str]:
    """Bind source set, prompt/schema/tools, runtime configuration, and all code bytes."""
    settings = get_settings()
    return {"selection_sha256": _digest(selection),
        "selection_file_sha256": _sha(root / SELECTION_PATH), **prompts,
        "review_contract_sha256": _sha(root / "prompts/propagation/review_contracts.json"),
        "development_license_record_sha256": _sha(root / "docs/decisions.md"),
        "case_cap_cny": str(CASE_CAP), "run_cap_cny": str(RUN_CAP),
        "model": settings.deepseek_model,
        "base_url_sha256": sha256(settings.deepseek_base_url.encode()).hexdigest(),
        "prices_sha256": _digest(get_cost_config().model_dump()),
        "evaluation_sha256": _digest(get_evaluation_config().model_dump(mode="json")),
        "wallet_evaluation_sha256": _digest(m6c_wallet_evaluation_config().model_dump(mode="json")),
        "wallet_run_cap_cny": str(RUN_CAP),
        "m6c_budget_sha256": _digest(budget.model_dump()),
        "files_sha256": _digest(files)}
