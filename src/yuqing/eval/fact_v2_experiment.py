"""收集事实 V2 配对运行条件，并在任何实验副作用前做容量预检。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Literal, Mapping

from src.yuqing import llm
from src.yuqing.eval.dataset import RetrievalQuestion, load_questions
from src.yuqing.eval.fact_v2_conditions import build_fact_v2_initial_requests
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.initial_evidence import InitialEvidenceManifestV2
from src.yuqing.schemas import CitationRepairV2, FactVerdict
from src.yuqing.settings import (FactV2EvaluationConfig, HarnessConfig, PROJECT_ROOT,
    get_cost_config, get_evaluation_config, get_fact_v2_evaluation_config,
    get_harness_config, get_settings)

QUESTION_SETS = {
    "dev90": ("eval/questions_30.jsonl", "eval/dev_extra_60.jsonl"),
    "dev_val_120_v2": ("eval/dev_val_120_v2.jsonl",),
}
PROMPT_FILES = {
    "single": "prompts/fact_check/single_v4.md",
    "loop": "prompts/fact_check/investigator_v2.md",
}
PROMPT_MANIFEST = "prompts/fact_check/v2_manifest.json"
V3_PROMPT = "prompts/fact_check/v3.md"
V3_CONTRACTS = "prompts/fact_check/contracts.json"
FROZEN_FACT_COMMIT = "6224c9c0097ce456b3a45659a179bbe4f1aa0670"
R2_COST_AUDIT = "traces/review-packages/F2-dev-val-3arm-preflight-2026-10-22/readonly-audit.json"
INITIAL_MANIFESTS = {
    "dev90": "eval/dev90_initial_evidence_v2.jsonl",
    "dev_val_120_v2": "eval/dev_val_120_v2_initial_evidence.jsonl",
}
INITIAL_AUDITS = {
    "dev90": "eval/dev90_initial_evidence_v2_audit.jsonl",
    "dev_val_120_v2": "eval/dev_val_120_v2_initial_evidence_audit.jsonl",
}


@dataclass(frozen=True)
class FactV2RunConditions:
    """V2固定题目、初始证据与环境指纹；不创建工作区或钱包。"""

    dataset: str
    questions: tuple[RetrievalQuestion, ...]
    manifests: tuple[InitialEvidenceManifestV2, ...]
    prompt_texts: Mapping[str, str]
    conditions: Mapping[str, str]
    round_label: str | None = None


@dataclass(frozen=True)
class FactV2Preflight:
    """经校验的节点预算、每题首请求估算与许可状态。"""

    node_config: HarnessConfig
    maximum_initial_input: int
    prompt_approved_for_dev: bool


def _canonical(value: object) -> bytes:
    """将条件映射编码为无歧义JSON字节。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_files(root: Path, paths: list[Path]) -> dict[str, str]:
    """只哈希明确列出的项目内文件，缺失或越界均停止。"""
    result = {}
    for path in sorted(set(paths)):
        actual = path.resolve()
        if not actual.is_relative_to(root.resolve()) or not actual.is_file():
            raise ValueError("fact_v2_condition_file_missing_or_outside_project")
        result["file:" + actual.relative_to(root.resolve()).as_posix()] = sha256(actual.read_bytes()).hexdigest()
    return result


def _questions(root: Path, dataset: str) -> tuple[RetrievalQuestion, ...]:
    """按任务规格只加载冻结dev90或合格v2验证集，保持原顺序。"""
    if dataset not in QUESTION_SETS:
        raise ValueError("fact_v2_dataset_not_allowed")
    paths = [root / relative for relative in QUESTION_SETS[dataset]]
    questions = tuple(question for path in paths for question in load_questions(path))
    expected = 90 if dataset == "dev90" else 120
    ids = [item.sample_id for item in questions]
    if len(questions) != expected or len(set(ids)) != expected:
        raise ValueError("fact_v2_frozen_question_order_invalid")
    if any(not item.sample_id.startswith("chef:dev:") for item in questions):
        raise ValueError("fact_v2_non_dev_question")
    return questions


def _initial_manifests(root: Path, dataset: str,
                       questions: tuple[RetrievalQuestion, ...]) -> tuple[InitialEvidenceManifestV2, ...]:
    """要求初始证据行数、次序、题号与原声明全部对应。"""
    path = root / INITIAL_MANIFESTS[dataset]
    lines = path.read_bytes().splitlines()
    if len(lines) != len(questions):
        raise ValueError("fact_v2_initial_manifest_count_mismatch")
    manifests = tuple(InitialEvidenceManifestV2.model_validate_json(line, strict=True) for line in lines)
    identities = {(item.retrieval_sha256, item.corpus_sha256, item.limits_sha256) for item in manifests}
    if len(identities) != 1 or not (root / INITIAL_AUDITS[dataset]).is_file():
        raise ValueError("fact_v2_initial_manifest_conditions_inconsistent")
    for question, manifest in zip(questions, manifests, strict=True):
        if (manifest.sample_id != question.sample_id or
                manifest.claim_sha256 != sha256(question.claim.encode("utf-8")).hexdigest()):
            raise ValueError("fact_v2_initial_manifest_question_mismatch")
    return manifests


def _loaded_object_digests(
    manifests: tuple[InitialEvidenceManifestV2, ...], prompts: Mapping[str, str]
) -> dict[str, str]:
    """指纹化已加载对象，避免可变嵌套数据与磁盘快照脱节。"""
    return {
        "manifest_objects_sha256": sha256(_canonical(
            [item.model_dump(mode="json") for item in manifests])).hexdigest(),
        "prompt_texts_sha256": sha256(_canonical(dict(prompts))).hexdigest(),
    }


def _load_prompts(root: Path, dataset: str) -> dict[str, str]:
    """Load only registered prompt versions needed by the selected experiment."""
    prompts = {role: (root / path).read_text(encoding="utf-8") for role, path in PROMPT_FILES.items()}
    if dataset == "dev_val_120_v2":
        prompts["single_v3"] = (root / V3_PROMPT).read_text(encoding="utf-8")
        _verify_v3_prompt(root, prompts["single_v3"])
    if any(not prompt.strip() for prompt in prompts.values()):
        raise ValueError("fact_v2_prompt_empty")
    return prompts


def collect_fact_v2_conditions(root: Path = PROJECT_ROOT, *,
                               dataset: Literal["dev90", "dev_val_120_v2"],
                               round_label: Literal["R2"] | None = None) -> FactV2RunConditions:
    """固定数据、提示词、容量、语料、代码与运行环境，不写任何实验产物。"""
    questions = _questions(root, dataset)
    manifests = _initial_manifests(root, dataset, questions)
    prompts = _load_prompts(root, dataset)
    settings = get_settings()
    store = settings.dev_store_path
    _verify_prompt_manifest(root, prompts)
    paths = [root / path for path in QUESTION_SETS[dataset]]
    paths.extend(root / path for path in PROMPT_FILES.values())
    paths.extend((root / PROMPT_MANIFEST, root / INITIAL_MANIFESTS[dataset],
        root / INITIAL_AUDITS[dataset], root / "config/fact_v2_eval.yaml",
        root / "config/weights.yaml", settings.chef_dev_path, store / "corpus.json", store / "READY"))
    if dataset == "dev_val_120_v2":
        paths.extend((root / V3_PROMPT, root / V3_CONTRACTS, root / R2_COST_AUDIT))
    paths.extend(sorted((root / "src/yuqing").rglob("*.py")))
    paths.append(root / "scripts/run_fact_v2_dev90.py")
    if dataset == "dev_val_120_v2":
        paths.append(root / "scripts/run_fact_v2_dev_val_120.py")
    conditions = _hash_files(root, paths)
    conditions.update({"dataset": dataset, "question_ids_sha256": sha256(_canonical(
        [item.sample_id for item in questions])).hexdigest(),
        "question_content_sha256": sha256(_canonical([item.model_dump(mode="json")
            for item in questions])).hexdigest(),
        "initial_manifest_rows_sha256": sha256((root / INITIAL_MANIFESTS[dataset]).read_bytes()).hexdigest(),
        "model": settings.deepseek_model,
        "endpoint_sha256": sha256(settings.deepseek_base_url.encode("utf-8")).hexdigest(),
        "runtime:python": platform.python_version(),
        "runtime:openai": version("openai"), "runtime:pydantic": version("pydantic")})
    prompt_roles = dict(PROMPT_FILES)
    if dataset == "dev_val_120_v2":
        prompt_roles["single_v3"] = V3_PROMPT
    conditions["prompt_roles"] = json.dumps(prompt_roles, ensure_ascii=False, sort_keys=True)
    conditions["initial_evidence_generation_commit"] = "b0aa1cb"
    first = manifests[0]
    conditions.update({"initial_evidence_generation_retrieval_sha256": first.retrieval_sha256,
        "initial_evidence_generation_corpus_sha256": first.corpus_sha256,
        "initial_evidence_generation_limits_sha256": first.limits_sha256})
    corpus = json.loads((store / "corpus.json").read_bytes())
    source_sha = sha256(settings.chef_dev_path.read_bytes()).hexdigest()
    if (corpus.get("corpus_sha256") != first.corpus_sha256 or
            corpus.get("source_sha256") != source_sha):
        raise ValueError("fact_v2_initial_evidence_corpus_mismatch")
    conditions.update(_loaded_object_digests(manifests, prompts))
    _register_round_label(conditions, round_label)
    return FactV2RunConditions(dataset, questions, manifests,
        MappingProxyType(prompts), MappingProxyType(conditions), round_label)


def enable_fact_v2_three_arm(conditions: FactV2RunConditions) -> FactV2RunConditions:
    """Mark only the author-approved dev_val_120 three-arm execution identity."""
    if conditions.dataset != "dev_val_120_v2" or "single_v3" not in conditions.prompt_texts:
        raise ValueError("fact_v2_three_arm_dataset_required")
    values = dict(conditions.conditions)
    values.update({"evaluation:group_layout": "fact-v2-three-arm-v1",
        "fact_v2_frozen_commit": "6224c9c0097ce456b3a45659a179bbe4f1aa0670",
        "fact_v2_execution_identity": "frozen-fact-inputs-plus-reviewed-three-arm-adapter"})
    return FactV2RunConditions(conditions.dataset, conditions.questions, conditions.manifests,
        conditions.prompt_texts, MappingProxyType(values), conditions.round_label)


def _verify_prompt_manifest(root: Path, prompts: Mapping[str, str]) -> None:
    """要求已登记的提示词SHA与本次加载的完整字节一致。"""
    manifest = json.loads((root / PROMPT_MANIFEST).read_bytes())
    declared = {item.get("path"): item for item in manifest.get("prompts", []) if isinstance(item, dict)}
    for role, relative in PROMPT_FILES.items():
        digest = sha256(prompts[role].encode("utf-8")).hexdigest()
        if declared.get(relative, {}).get("prompt_sha256") != digest:
            raise ValueError("fact_v2_prompt_manifest_mismatch")


def _verify_v3_prompt(root: Path, prompt: str) -> None:
    """按已登记的V1合同核对v3提示词及两种结构Schema指纹。"""
    contracts = json.loads((root / V3_CONTRACTS).read_bytes())
    matches = [item for item in contracts if isinstance(item, dict) and item.get("path") == V3_PROMPT]
    if len(matches) != 1:
        raise ValueError("fact_v3_prompt_contract_missing_or_ambiguous")
    record = matches[0]
    prompt_sha = sha256(prompt.encode("utf-8")).hexdigest()
    schema_shas = [sha256(json.dumps(model.model_json_schema(), ensure_ascii=False).encode("utf-8")).hexdigest()
        for model in (FactVerdict, CitationRepairV2)]
    if (record.get("contract_version") != "fact-verdict-v3" or record.get("prompt_sha256") != prompt_sha
            or record.get("approved_for_dev") is not True or schema_shas != [
                record.get("verdict_schema_sha256"), record.get("repair_schema_sha256")]):
        raise ValueError("fact_v3_prompt_contract_mismatch")


def _register_round_label(conditions: dict[str, str], round_label: str | None) -> None:
    """仅为显式R2条件加入轮次身份，保留R1旧指纹内容。"""
    if round_label is not None:
        conditions["experiment_round"] = round_label


def build_fact_v2_node_config(base: HarnessConfig, capacity: FactV2EvaluationConfig) -> HarnessConfig:
    """将已确认的事实V2节点容量应用到循环预算，其他安全上限沿用原配置。"""
    values = base.model_dump()
    values.update(token_budget_per_call=capacity.token_budget_per_call,
        max_output_tokens=capacity.max_output_tokens, max_total_tokens=capacity.max_total_tokens,
        max_cost_cny=capacity.max_cost_cny, final_token_reserve=capacity.final_token_reserve,
        input_bound_margin_tokens=capacity.input_bound_margin_tokens,
        input_estimate_version=capacity.input_estimate_version)
    return HarnessConfig.model_validate(values)


def _prompt_approved(root: Path, conditions: FactV2RunConditions) -> bool:
    """核对V2正式提示词登记指纹，并要求两版都明确批准dev运行。"""
    raw = (root / PROMPT_MANIFEST).read_bytes()
    manifest = json.loads(raw)
    if conditions.round_label == "R2" and manifest.get("round") != "R2":
        return False
    if conditions.round_label is None and manifest.get("round") == "R2":
        return False
    if manifest.get("status") != "approved" or manifest.get("runtime_registered") is not True:
        return False
    if manifest.get("approved_for_dev") is not True or manifest.get("approved_for_test") is not False:
        return False
    entries = {item.get("path"): item for item in manifest.get("prompts", []) if isinstance(item, dict)}
    for role, relative in PROMPT_FILES.items():
        item = entries.get(relative)
        digest = sha256(conditions.prompt_texts[role].encode("utf-8")).hexdigest()
        if not item or item.get("prompt_sha256") != digest or item.get("approved_for_dev") is not True:
            return False
    if conditions.dataset == "dev_val_120_v2":
        try:
            _verify_v3_prompt(root, conditions.prompt_texts["single_v3"])
        except (OSError, ValueError, TypeError, KeyError):
            return False
    return True


def preflight_fact_v2(conditions: FactV2RunConditions, *, root: Path = PROJECT_ROOT,
                      require_provider_license: bool = True) -> FactV2Preflight:
    """先核价格、普通请求加收尾容量与每题真实首请求，再检查模型许可。"""
    prices = get_cost_config()
    if prices.input_per_1m is None or prices.output_per_1m is None:
        raise ValueError("fact_v2_unknown_prices")
    capacity = get_fact_v2_evaluation_config()
    node = build_fact_v2_node_config(get_harness_config(), capacity)
    tracker = BudgetTracker(node, prices)
    if not tracker.check_model_call(node.token_budget_per_call, node.max_output_tokens).allowed:
        raise ValueError("fact_v2_normal_plus_closeout_capacity_insufficient")
    maximum = 0
    for question, manifest in zip(conditions.questions, conditions.manifests, strict=True):
        requests = build_fact_v2_initial_requests(question.sample_id, question.claim,
            tuple(manifest.entries), conditions.prompt_texts["single"], conditions.prompt_texts["loop"], capacity)
        estimate = max(requests.single_input_estimate, requests.loop_input_estimate)
        if conditions.dataset == "dev_val_120_v2":
            from src.yuqing.agents.fact_check import FactCheckAgent
            v3_evidence = [item.sources[0] for item in manifest.entries]
            v3_agent = FactCheckAgent(conditions.prompt_texts["single_v3"])
            v3_system = llm.build_structured_system(conditions.prompt_texts["single_v3"], FactVerdict)
            v3_messages = [dict(role="system", content=v3_system),
                dict(role="user", content=v3_agent.build_prompt(question.claim, v3_evidence))]
            v3_capacity = get_harness_config()
            if not BudgetTracker(v3_capacity, prices).check_model_call(
                    v3_capacity.token_budget_per_call, v3_capacity.max_output_tokens).allowed:
                raise ValueError("fact_v3_normal_plus_closeout_capacity_insufficient")
            v3_estimate = llm.estimate_request_input(v3_messages, [], v3_capacity.input_bound_margin_tokens)
            if v3_estimate > v3_capacity.token_budget_per_call:
                raise ValueError("fact_v3_initial_request_over_token_limit")
            estimate = max(estimate, v3_estimate)
        if estimate > node.token_budget_per_call:
            raise ValueError("fact_v2_initial_request_over_token_limit")
        maximum = max(maximum, estimate)
    approved = _prompt_approved(root, conditions)
    if require_provider_license and not approved:
        raise ValueError("fact_v2_prompt_not_licensed")
    return FactV2Preflight(node, maximum, approved)


def require_fact_v2_provider_ready(conditions: FactV2RunConditions, *, root: Path = PROJECT_ROOT) -> None:
    """要求正式提示词和容量预检通过；不创建运行状态或触发模型请求。"""
    preflight_fact_v2(conditions, root=root, require_provider_license=True)


def assert_fact_v2_conditions_unchanged(expected: FactV2RunConditions, *,
                                        root: Path = PROJECT_ROOT) -> None:
    """重新采集条件并拒绝任何题单、清单、Prompt、代码或环境漂移。"""
    question_digest = sha256(_canonical([item.model_dump(mode="json")
        for item in expected.questions])).hexdigest()
    manifest_digest = sha256(_canonical([item.model_dump(mode="json")
        for item in expected.manifests])).hexdigest()
    prompt_digest = sha256(_canonical(dict(expected.prompt_texts))).hexdigest()
    if (question_digest != expected.conditions.get("question_content_sha256") or
            manifest_digest != expected.conditions.get("manifest_objects_sha256") or
            prompt_digest != expected.conditions.get("prompt_texts_sha256")):
        raise ValueError("fact_v2_conditions_changed:in_memory_snapshot")
    current = (collect_fact_v2_conditions(root, dataset=expected.dataset) if expected.round_label is None
        else collect_fact_v2_conditions(root, dataset=expected.dataset, round_label=expected.round_label))
    if expected.conditions.get("evaluation:group_layout") == "fact-v2-three-arm-v1":
        current = enable_fact_v2_three_arm(current)
    if current.conditions != expected.conditions:
        changed = sorted(key for key in set(current.conditions) | set(expected.conditions)
                         if current.conditions.get(key) != expected.conditions.get(key))
        raise ValueError("fact_v2_conditions_changed:" + ",".join(changed))


def fact_v2_budget_ceiling(mode: Literal["full", "pilot"], override: Decimal | None) -> Decimal:
    """复用现有整场费用策略，覆盖值只能收紧已批准配置。"""
    config = get_evaluation_config()
    ceiling = Decimal(str(config.max_run_cost_cny if mode == "full" else config.pilot_max_run_cost_cny))
    if override is not None and (not isinstance(override, Decimal) or not override.is_finite()
                                 or override <= 0 or override > ceiling):
        raise ValueError("fact_v2_experiment_budget_cannot_increase")
    return ceiling if override is None else Decimal(override)
