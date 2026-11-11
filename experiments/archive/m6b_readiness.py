"""M6b离线资格与显式装配；不授予Provider、钱包或实验认领权限。"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from experiments.archive import joint_demo as joint
from src.yuqing.eval.propagation_v11 import _catalog, _prior_use, _load_id_file
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.schemas import AgentSpec
from src.yuqing.settings import get_m5_case_budget_config, get_propagation_harness_config

PROMPT = "experiments/archive/prompts/propagation/v1_m6b.draft.md"
SELECTION = "docs/cards/M6b-C_selection.json"
JOINT_CASES = ("658", "2508", "407")
APPROVED_PROMPT_SHA256 = "86156d7a8d0ef4a5eeadf5774bd40c404cfca0821cd7f107f1f43c9f849dcf49"
PROTECTED = ("eval/label_pool_100.jsonl", "eval/test_60.jsonl", "eval/dev_val_120_v2.jsonl",
    "docs/review_history/风险补充50条_清单_2026-10-18.jsonl")
POST_SELECTION_DECISION = "docs/decisions.md"
M6B_CONDITION_PATHS = ("experiments/archive/m6b_readiness.py", "experiments/archive/m6b_run.py",
    "experiments/archive/m6b_execution.py", "src/yuqing/eval/propagation_v11.py",
    "src/yuqing/agents/propagation_v11.py", "src/yuqing/collab/propagation_delivery.py",
    "src/yuqing/collab/propagation_followup.py", "src/yuqing/graph/propagation_collab.py",
    "experiments/archive/m6b_runner.py", "experiments/archive/m6b_live.py",
    "scripts/prepare_m6b.py", "docs/cards/M6b-C_plan.md",
    "docs/cards/M6b-C-live-readiness.md", POST_SELECTION_DECISION)


def _rank(sample_id: str) -> str:
    """使用登记的种子42按身份排序，与内容和金标签无关。"""
    return sha256(("42|" + sample_id).encode()).hexdigest()


def _sample_record(sample: Any, group: str) -> dict[str, Any]:
    """只保存模型输入字段，金标签不进入运行材料。"""
    safe = sample.model_copy(update={"gold_veracity": None, "gold_risk": None})
    return {"sample_id": safe.sample_id, "group": group,
        "sample_sha256": joint._digest(safe.model_dump(mode="json")),
        "coverage": "demo" if group == "joint_reuse" else "none" if group == "ced_new" else "full"}


def build_m6b_selection(root: Path) -> dict[str, Any]:
    """先登记身份排除，再按固定顺序生成5CED+3联合+2CHEF清单。"""
    used, sources, scope = _prior_use(root)
    protected: set[str] = set()
    for name in PROTECTED:
        sources[name] = joint._sha(root / name)
        protected.update(_load_id_file(root / name))
    previous = "docs/cards/A-v1.1-1_selection.json"
    sources[previous] = joint._sha(root / previous)
    ced20 = {row["sample_id"] for row in json.loads((root / previous).read_bytes())["selected"]}
    originals = root / "data/raw/CED_Dataset"
    catalog = _catalog(originals)
    joint_ids = {joint._case_files(case).sample_id for case in JOINT_CASES}
    selected, audit = _select_ced(originals, catalog, used, protected | ced20 | joint_ids, sources)
    for case in JOINT_CASES:
        prepared = joint._validated_joint_demo(root, case)
        selected.append({**_sample_record(prepared.sample, "joint_reuse"), "joint_case": case})
        sources.update({name: digest for name,digest in prepared.conditions.items() if (root / name).is_file()})
        sources.update({str((prepared.store / name).relative_to(root).as_posix()):joint._sha(prepared.store / name)
            for name in ("READY", "corpus.json", "initial.json")})
    questions = []
    for name in ("eval/questions_30.jsonl", "eval/dev_extra_60.jsonl"):
        sources[name] = joint._sha(root / name)
        questions.extend(json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line.strip())
    eligible = sorted((q for q in questions if q["sample_id"] not in protected), key=lambda q:_rank(q["sample_id"]))
    if len(eligible) < 2:
        raise ValueError("m6b_chef_eligible_shortage")
    selected.extend({"sample_id": q["sample_id"], "group": "chef_dev_reuse", "coverage": "full",
        "question_sha256": joint._digest(q)} for q in eligible[:2])
    sources["ced_catalog_sha256"] = joint._digest(catalog)
    return {"version": "m6b-ten-v1", "seed": 42, "selection_rule": "M6b-C_plan.md",
        "prior_use_scope": scope, "samples": [{"order": i+1, **row} for i,row in enumerate(selected)],
        "ced_candidates": audit, "source_sha256": sources, "paid_ready": False}


def _select_ced(dataset: Path, catalog: list, used: dict, excluded: set,
                sources: dict[str,str]) -> tuple[list, list]:
    """固定身份顺序；已使用、保护身份或解析失败都留下排除理由。"""
    selected, audit = [], []
    for sample_id, category, stem in sorted(catalog, key=lambda row:_rank(row[0])):
        reasons = sorted(used.get(sample_id, ()))
        if sample_id in excluded:
            reasons.append("protected_or_ced20_or_joint")
        row = {"sample_id": sample_id, "rank_sha256": _rank(sample_id), "reasons": reasons}
        if not reasons and len(selected) < 5:
            try:
                sample = load_ced_sample(dataset, stem+".json", rumor=category == "rumor")
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                row["reasons"] = ["parse_" + type(exc).__name__]
            else:
                selected.append(_sample_record(sample, "ced_new"))
                for folder in ("original-microblog", "rumor-repost" if category == "rumor" else "non-rumor-repost"):
                    path = dataset / folder / (stem+".json")
                    sources[path.relative_to(dataset.parents[2]).as_posix()] = joint._sha(path)
                row["selected"] = True
        audit.append(row)
    if len(selected) != 5:
        raise ValueError("m6b_ced_eligible_shortage")
    return selected, audit


def m6b_composition(prepared: joint.PreparedJointDemo, wallet: object, trace_root: Path,
                    initial_path: Path | None = None) -> Any:
    """在同案现有装配上显式开启M6b；调用者仍须获得真实运行许可。"""
    from src.yuqing.graph.composition import CaseComposition
    composition = prepared.composition(wallet, trace_root)
    try:
        spec = AgentSpec(agent="propagation_analyst",
            system_prompt=(prepared.root / PROMPT).read_text(encoding="utf-8"))
        inputs = replace(composition.inputs, m6b=True, propagation_spec=spec,
            initial_manifest_path=initial_path or composition.inputs.initial_manifest_path,
            group_id="m6b_development", variant="m6b-offline-candidate")
        return CaseComposition(inputs)
    except Exception:
        composition.inputs.fact_worker_scope.close()
        composition.inputs.propagation_worker_scope.close()
        raise


def m6b_preflight(root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    """核对输入快照与代码闭包，返回阻塞项而不创建实验或钱包。"""
    rows = selection["samples"]
    if (selection.get("version") != "m6b-ten-v1" or len(rows) != 10
            or len({row["sample_id"] for row in rows}) != 10
            or [row["order"] for row in rows] != list(range(1,len(rows)+1))
            or [row["group"] for row in rows] != ["ced_new"]*5 + ["joint_reuse"]*3 + ["chef_dev_reuse"]*2):
        raise ValueError("m6b_selection_identity_invalid")
    frozen_path = root / SELECTION
    frozen = json.loads(frozen_path.read_bytes())
    if joint._digest(frozen) != joint._digest(selection):
        raise ValueError("m6b_selection_file_mismatch")
    rebuilt = build_m6b_selection(root)
    runtime_files = set(joint._runtime_code_files(root))
    expected_sources = {name: digest for name, digest in selection["source_sha256"].items()
        if name not in runtime_files}
    current_sources = {name: digest for name, digest in rebuilt["source_sha256"].items()
        if name not in runtime_files}
    source_differences = {name for name in expected_sources.keys() | current_sources.keys()
        if expected_sources.get(name) != current_sources.get(name)}
    if (source_differences - {POST_SELECTION_DECISION}
            or {key: value for key, value in rebuilt.items() if key != "source_sha256"}
            != {key: value for key, value in selection.items() if key != "source_sha256"}):
        raise ValueError("m6b_selection_or_source_drift")
    paths = {*joint._runtime_code_files(root), *M6B_CONDITION_PATHS}
    conditions = {**selection["source_sha256"], **{name:joint._sha(root / name) for name in sorted(paths)}}
    prompt_sha = joint._sha(root / PROMPT)
    if prompt_sha != APPROVED_PROMPT_SHA256:
        raise ValueError("m6b_prompt_bytes_not_author_approved")
    conditions.update({SELECTION: joint._digest(selection), PROMPT: prompt_sha,
        "prompt_dev_permission": "author_approved_exact_sha256"})
    from src.yuqing.settings import get_settings, get_cost_config
    runtime = get_settings()
    conditions.update(model=runtime.deepseek_model, base_url_sha256=sha256(runtime.deepseek_base_url.encode()).hexdigest(),
        prices_sha256=joint._digest(get_cost_config().model_dump()), timeout=str(runtime.llm_timeout_s))
    budget, prop = get_m5_case_budget_config(), get_propagation_harness_config()
    return {"mode": "offline_only", "provider_calls": 0, "formal_wallets_created": 0,
        "claims_created": 0, "paid_ready": False, "selection_sha256": joint._digest(selection),
        "conditions": conditions, "conditions_sha256": joint._digest(conditions),
        "budget": {"case_hard_cap_cny": str(budget.max_cost_cny),
            "propagation_effective_cap_cny": str(min(Decimal(str(prop.max_cost_cny)), Decimal(str(budget.action_caps["propagation_investigate"])))),
            "ten_case_theoretical_cny": str(Decimal(str(budget.max_cost_cny))*len(rows))},
        "blockers": ["runner_is_offline_mock_only_no_provider_path",
            "real_model_run_requires_separate_authorization"]}
