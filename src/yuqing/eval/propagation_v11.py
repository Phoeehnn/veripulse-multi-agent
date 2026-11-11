"""A-v1.1传播验证的确定性CED选样和排除审计。"""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.yuqing import llm
from src.yuqing.agents import propagation
from src.yuqing.agents.propagation_v11 import (submission_model, stats_receipt,
    v11_submission_feedback)
from src.yuqing.eval.preflight import PreparedPropagationPrompt, validate_propagation_startup
from src.yuqing.harness.budget import BudgetTracker
from src.yuqing.harness.tools import ToolRegistry
from src.yuqing.harness.trace import RunContext
from src.yuqing.schemas import AgentSpec, PropagationRaw, ToolRequest
from src.yuqing.eval.propagation_eval import _input_fingerprint
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.features.propagation import SpreadStats, get_spread_stats
from src.yuqing.schemas import Sample
from src.yuqing.settings import (get_cost_config, get_policy_config,
    get_propagation_harness_config, get_propagation_tools_config, get_settings,
    CostConfig, PropagationHarnessConfig, PropagationToolsConfig)

SEED = 42
SELECTION_SHA256 = "1506692bde3f0d5a1c314cea690426f1d20820b0fd89f47338157df016da4ad6"
SAMPLE_PATTERN = re.compile(r"ced:(?:rumor|non_rumor):[A-Za-z0-9_-]+")


class CandidateAudit(BaseModel):
    """记录每个CED候选的身份、稳定顺序和纳入/排除原因。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    category: Literal["rumor", "non_rumor"]
    rank_sha256: str
    disposition: Literal["selected", "excluded", "not_selected_by_rank", "parse_failed"]
    reasons: list[str]


class SelectedSample(BaseModel):
    """固定样本身份、输入指纹和对应原始文件指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    category: Literal["rumor", "non_rumor"]
    input_sha256: str
    original_sha256: str
    propagation_sha256: str


class PropagationSelection(BaseModel):
    """20个开发验证样本及全部候选的逐项审计。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["a-v1.1-ced20-v1"]
    seed: int
    selection_rule: str
    source_sha256: dict[str, str]
    prior_use_scope: str
    selected: list[SelectedSample]
    candidates: list[CandidateAudit]



def _sha(path: Path) -> str:
    """计算文件原始字节的SHA-256。"""

    return sha256(path.read_bytes()).hexdigest()


def _catalog(dataset: Path) -> list[tuple[str, str, str]]:
    """枚举CED原帖和传播文件，缺配对时拒绝生成题单。"""

    originals = {path.stem for path in (dataset / "original-microblog").glob("*.json")}
    rows = []
    seen: set[str] = set()
    for category, folder in (("non_rumor", "non-rumor-repost"), ("rumor", "rumor-repost")):
        stems = {path.stem for path in (dataset / folder).glob("*.json")}
        if stems & seen or not stems <= originals:
            raise ValueError("propagation_v11_ced_source_pair_invalid")
        seen.update(stems)
        rows.extend((f"ced:{category}:{stem}", category, stem) for stem in stems)
    if seen != originals or len(rows) != len({row[0] for row in rows}):
        raise ValueError("propagation_v11_ced_catalog_invalid")
    return sorted(rows)


def _load_id_file(path: Path) -> set[str]:
    """读取JSONL中可识别的样本编号，损坏文件即停。"""

    found: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict) and isinstance(value.get("sample_id"), str):
            found.add(value["sample_id"])
    return found


def _record(used: dict[str, set[str]], sample_id: str, reason: str) -> None:
    """将已知CED样本及其具体排除来源写入去重表。"""

    if sample_id.startswith("ced:"):
        used.setdefault(sample_id, set()).add(reason)


def _record_result_sources(root: Path, used: dict[str, set[str]], sources: dict[str, str]) -> None:
    """读取所有正式结果CSV并按样本身份排除曾运行过的CED样本。"""

    for path in sorted((root / "results").glob("*.csv")):
        sources[path.relative_to(root).as_posix()] = _sha(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                _record(used, row.get("sample_id", ""), "prior_result_csv")


def _record_frozen_sources(root: Path, used: dict[str, set[str]], sources: dict[str, str]) -> None:
    """记录标注池、测试集与补充人工标注来源，并排除其中CED题目。"""

    id_files = ("eval/label_pool_100.jsonl", "eval/test_60.jsonl",
                "docs/review_history/风险补充50条_清单_2026-10-18.jsonl")
    id_reasons = ("m8_annotation_pool", "frozen_test_split", "supplemental_annotation")
    for relative, reason in zip(id_files, id_reasons, strict=True):
        path = root / relative
        sources[relative] = _sha(path)
        for sample_id in _load_id_file(path):
            _record(used, sample_id, reason)


def _record_m8_sources(root: Path, used: dict[str, set[str]], sources: dict[str, str]) -> None:
    """从存在的M8冻结清单中收集所有已使用CED槽位。"""

    for relative in ("traces/m8-fixed-100-retry1/manifest.json",
                     "traces/m8-fixed-100-94c91dd/manifest.json"):
        path = root / relative
        if path.is_file():
            sources[relative] = _sha(path)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for row in manifest.get("slots", []):
                _record(used, row.get("sample_id", ""), "m8_paid_sample")


def _record_trace_and_history_docs(root: Path, used: dict[str, set[str]],
                                   sources: dict[str, str]) -> None:
    """固定范围核对M9 Trace和既有M3至M9记录，不扫描本卡新材料。"""

    trace_root = root / "traces/m9-demo"
    trace_count = 0
    if trace_root.is_dir():
        for path in trace_root.rglob("*.jsonl"):
            trace_count += 1
            text = path.read_text(encoding="utf-8", errors="strict")
            for sample_id in SAMPLE_PATTERN.findall(text):
                _record(used, sample_id, "m9_trace")
            sources[path.relative_to(root).as_posix()] = _sha(path)
    docs_root = root / "docs"
    historical_docs = {docs_root / "decisions.md", docs_root / "M3-retirement-path-index.md",
        docs_root / "cards/F1-CED10-readiness.md", docs_root / "cards/M9-demo-case-selection.md",
        docs_root / "cards/M4-five-sample-readiness.md", docs_root / "handoff/M9-pre-run.md"}
    historical_docs.update(docs_root / "learning" / name for name in (
        "M3-D2-R2-explained.md", "M3-D2-prep-explained.md", "M3-D2-explained.md",
        "M5-B-checkpoint-review-2026-10-23.md", "M5-A-offline-close-capacity-2026-10-23.md",
        "M8-retry1-diagnostic.md", "M5-B-prepaid-review-2026-10-23.md",
        "devlog/F1-CED10-readiness.md"))
    for path in sorted(historical_docs):
        if not path.is_file():
            raise ValueError(f"propagation_v11_history_source_missing:{path.relative_to(root).as_posix()}")
        text = path.read_text(encoding="utf-8", errors="strict")
        sources[path.relative_to(root).as_posix()] = _sha(path)
        for sample_id in SAMPLE_PATTERN.findall(text):
            _record(used, sample_id, "documented_prior_use_or_review")
    if not trace_count:
        raise ValueError("propagation_v11_m9_trace_sources_missing")


def _prior_use(root: Path) -> tuple[dict[str, set[str]], dict[str, str], str]:
    """从结果、冻结池、M8清单和已归档M9 Trace收集保守排除身份。"""

    used: dict[str, set[str]] = {}
    sources: dict[str, str] = {}
    _record_result_sources(root, used, sources)
    _record_frozen_sources(root, used, sources)
    _record_m8_sources(root, used, sources)
    _record_trace_and_history_docs(root, used, sources)
    scope = ("排除所有results CSV中的CED身份、label_pool/test/补充标注、两份M8运行清单、"
             "M9归档JSONL，以及预先固定的M3/M4/M5/M8/M9历史决策与复核文档中记录的CED身份；"
             "本卡新生成的题单和学习记录不回流为既往使用；仅测试解析未计为已调试。")
    return used, sources, scope


def _choose_category(data: Path, catalog: list[tuple[str, str, str]], category: str,
                     used: dict[str, set[str]], ranks: dict[str, str],
                     parsed: dict[str, Sample], parse_failed: set[str]) -> list[str]:
    """按固定rank取一个类别的前10条可读样本，解析失败只顺延。"""

    candidates = sorted((row for row in catalog if row[1] == category and row[0] not in used),
                        key=lambda row: ranks[row[0]])
    selected = []
    for sample_id, item_category, stem in candidates:
        try:
            parsed[sample_id] = load_ced_sample(data, f"{stem}.json",
                                                rumor=item_category == "rumor")
        except (OSError, ValueError, KeyError, TypeError):
            parse_failed.add(sample_id)
            continue
        selected.append(sample_id)
        if len(selected) == 10:
            return selected
    raise ValueError("propagation_v11_ced_candidates_insufficient")


def _selected_sources(data: Path, sample_ids: list[str], by_id: dict[str, tuple[str, str]],
                      parsed: dict[str, Sample]) -> list[SelectedSample]:
    """为选中样本固定输入、原帖与传播树指纹。"""

    result = []
    for sample_id in sample_ids:
        category, stem = by_id[sample_id]
        folder = "rumor-repost" if category == "rumor" else "non-rumor-repost"
        original = data / "original-microblog" / f"{stem}.json"
        propagation = data / folder / f"{stem}.json"
        result.append(SelectedSample(sample_id=sample_id, category=category,
            input_sha256=_input_fingerprint(parsed[sample_id]), original_sha256=_sha(original),
            propagation_sha256=_sha(propagation)))
    return result


def _candidate_audit(catalog: list[tuple[str, str, str]], ranks: dict[str, str],
                     used: dict[str, set[str]], selected: set[str],
                     parse_failed: set[str]) -> list[CandidateAudit]:
    """输出每个CED候选的纳入、排除或解析失败原因。"""

    audit = []
    for sample_id, category, _ in catalog:
        reason = sorted(used.get(sample_id, set()))
        if sample_id in parse_failed:
            disposition = "parse_failed"
            reason.append("ced_parser_rejected")
        elif sample_id in selected:
            disposition = "selected"
        elif reason:
            disposition = "excluded"
        else:
            disposition = "not_selected_by_rank"
        audit.append(CandidateAudit(sample_id=sample_id, category=category,
            rank_sha256=ranks[sample_id], disposition=disposition, reasons=reason))
    return audit


def build_selection(root: Path, *, dataset: Path | None = None) -> PropagationSelection:
    """以固定种子按谣言类别各选10条，并保留全候选排除审计。"""

    data = dataset or root / "data/raw/CED_Dataset"
    catalog = _catalog(data)
    used, sources, scope = _prior_use(root)
    ranks = {item[0]: sha256(f"{SEED}|{item[0]}".encode("utf-8")).hexdigest() for item in catalog}
    parsed: dict[str, Sample] = {}
    parse_failed: set[str] = set()
    selected_ids = [sample_id for category in ("rumor", "non_rumor")
                    for sample_id in _choose_category(data, catalog, category, used, ranks,
                                                       parsed, parse_failed)]
    selected_set = set(selected_ids)
    by_id = {sample_id: (category, stem) for sample_id, category, stem in catalog}
    selected = _selected_sources(data, selected_ids, by_id, parsed)
    audit = _candidate_audit(catalog, ranks, used, selected_set, parse_failed)
    sources["ced_catalog_sha256"] = sha256(json.dumps(catalog, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return PropagationSelection(version="a-v1.1-ced20-v1", seed=SEED,
        selection_rule="分别按SHA256(seed|sample_id)升序，排除既往请求/逐题分析身份后，各选rumor与non_rumor 10条；解析失败候选继续按固定排序向后取，不按内容或预期结果挑选。",
        source_sha256=sources, prior_use_scope=scope, selected=selected, candidates=audit)


def save_selection(root: Path) -> tuple[Path, str]:
    """首次写入固定选择清单并返回其SHA-256，禁止覆盖。"""

    selection = build_selection(root)
    target = root / "docs/cards/A-v1.1-1_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = selection.model_dump_json(indent=2) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
    return target, sha256(payload.encode("utf-8")).hexdigest()


def _pair_slot(root: Path, index: int, item: SelectedSample) -> dict[str, object]:
    """固定单题两版本顺序和本案动态提交Schema指纹。"""

    from src.yuqing.schemas import PropagationRaw

    pair_order = ("v1.0", "v1.1") if index < 10 else ("v1.1", "v1.0")
    stem = item.sample_id.rsplit(":", 1)[1] + ".json"
    sample = load_ced_sample(root / "data/raw/CED_Dataset", stem, rumor=item.category == "rumor")
    if _input_fingerprint(sample) != item.input_sha256:
        raise ValueError("propagation_v11_selected_input_drift")
    raw = PropagationRaw.model_validate(sample.propagation.model_dump()) if sample.propagation else None
    available = propagation._available_stats_fields(get_spread_stats(raw))
    schema = submission_model(available).model_json_schema()
    schema_sha = sha256(json.dumps(schema, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"pair_index": index + 1, "sample_id": item.sample_id,
        "sample_category": item.category, "input_sha256": item.input_sha256,
        "order": list(pair_order), "v1_1_allowed_spread_basis": list(available),
        "v1_1_submission_schema_sha256": schema_sha}


def build_pair_plan(root: Path, selection_path: Path) -> dict[str, object]:
    """生成不执行模型请求的20对评测条件与配对顺序登记。"""

    from src.yuqing.eval.preflight import propagation_contract_fingerprints

    selection_bytes = selection_path.read_bytes()
    selection, _ = validate_selection(root, selection_path)
    prompt = validate_propagation_startup(
        root / "prompts/propagation/v1.md", offline=True)
    node, costs = get_propagation_harness_config(), get_cost_config()
    slots = [_pair_slot(root, index, item) for index, item in enumerate(selection.selected)]
    condition = propagation_contract_fingerprints()
    condition.update(score_fingerprints(root))
    return {"version": "a-v1.1-paired-offline-plan-v2",
        "paid_execution_authorized": False,
        "selection_path": selection_path.relative_to(root).as_posix(),
        "selection_sha256": sha256(selection_bytes).hexdigest(),
        "prompt_path": prompt.path.relative_to(root).as_posix(),
        "prompt_sha256": prompt.prompt_sha256,
        "base_tools_sha256": prompt.tools_sha256,
        "base_schema_sha256": prompt.schema_sha256,
        "config_sha256": prompt.config_sha256, "code_sha256": prompt.code_sha256,
        "model": get_settings().deepseek_model,
        "input_price_per_million": str(costs.input_per_1m),
        "output_price_per_million": str(costs.output_per_1m),
        "slot_node_budget_cny": str(node.max_cost_cny),
        "shared_budget_cap_cny": "3.50", "pair_reserve_cny": "0.16",
        "max_worst_case_node_total_cny": "3.20", "unused_cap_margin_cny": "0.30",
        "historical_expected_cost_cny": "2.3688",
        "historical_expected_cost_basis": "M8 CED 31案平均估算约¥0.05922；40槽粗推，仅参考，不保证实际费用。",
        "paired_order_rule": "按固定题单顺序前10对v1.0先、后10对v1.1先；每对连续执行，结果不影响后续顺序。",
        "condition_fingerprints": condition,
        "initial_tools_sha256_by_variant": _initial_tool_fingerprints(),
        "pairs": slots}


def _initial_tool_fingerprints() -> dict[str, str]:
    """分别登记两版真实首请求工具Schema，不借用已读统计后的Schema。"""
    fingerprints = {}
    for version in ("v1.0", "v1.1"):
        registry = ToolRegistry()
        propagation.register_propagation_tools(registry, tools_config=get_propagation_tools_config(), contract_version=version)
        tools = [registry.get(name) for name in registry._tools]
        definitions = llm._tool_definitions(tools, False, submission_tool=get_propagation_harness_config().submission_tool)
        fingerprints[version] = sha256(json.dumps(definitions, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf8")).hexdigest()
    return fingerprints


def score_fingerprints(root: Path) -> dict[str, str]:
    """固定纯程序计分所读配置和实现，不把风险准确率纳入本次比较。"""
    return {"score_inputs_sha256": sha256(json.dumps({name: _sha(root / name) for name in (
        "config/weights.yaml", "config/rules_compliance.yaml", "src/yuqing/agents/arbiter.py")},
        sort_keys=True).encode("utf8")).hexdigest()}


def check_source_snapshot(root: Path, selection: PropagationSelection) -> None:
    """运行中只核验冻结来源与目录身份，不把本实验输出作为新的排除记录。"""
    for relative, digest in selection.source_sha256.items():
        if relative != "ced_catalog_sha256" and _sha(root / relative) != digest:
            raise ValueError("propagation_v11_selection_sources_drift")
    catalog = _catalog(root / "data/raw/CED_Dataset")
    digest = sha256(json.dumps(catalog, separators=(",", ":"), ensure_ascii=False).encode("utf8")).hexdigest()
    if digest != selection.source_sha256["ced_catalog_sha256"]:
        raise ValueError("propagation_v11_catalog_drift")
    recorded = {name for name in selection.source_sha256 if name.startswith("results/")}
    current = {path.relative_to(root).as_posix() for path in (root / "results").glob("*.csv")}
    if recorded != current:
        raise ValueError("propagation_v11_history_roster_drift")


def validate_selection(root: Path, path: Path) -> tuple[PropagationSelection, list[Sample]]:
    """核验固定20案、历史来源和实际输入；失败时不创建钱包或运行模型。"""
    if _sha(path) != SELECTION_SHA256:
        raise ValueError("propagation_v11_selection_sha_mismatch")
    selection = PropagationSelection.model_validate_json(path.read_bytes())
    ids = [item.sample_id for item in selection.selected]
    categories = [item.category for item in selection.selected]
    if selection.seed != SEED or len(ids) != 20 or len(set(ids)) != 20 or categories != ["rumor"] * 10 + ["non_rumor"] * 10:
        raise ValueError("propagation_v11_selection_identity_invalid")
    used, sources, _ = _prior_use(root)
    data = root / "data/raw/CED_Dataset"
    catalog = _catalog(data)
    sources["ced_catalog_sha256"] = sha256(json.dumps(catalog, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    if sources != selection.source_sha256:
        raise ValueError("propagation_v11_selection_sources_drift")
    audit = {item.sample_id: item for item in selection.candidates}
    if len(audit) != len(catalog) or set(audit) != {item[0] for item in catalog}:
        raise ValueError("propagation_v11_candidate_audit_invalid")
    samples = []
    origins: set[str] = set()
    prior_origins = {sid.rsplit(":", 1)[-1].split("_", 1)[-1] for sid in used}
    for item in selection.selected:
        row = audit[item.sample_id]
        if item.sample_id in used or row.disposition != "selected" or row.reasons or row.category != item.category:
            raise ValueError("propagation_v11_selection_exclusion_invalid")
        sample = validate_sample(root, item)
        origin = item.sample_id.rsplit(":", 1)[-1].split("_", 1)[-1]
        if origin in origins or origin in prior_origins:
            raise ValueError("propagation_v11_duplicate_original_identity")
        origins.add(origin)
        samples.append(sample)
    return selection, samples


def validate_sample(root: Path, item: SelectedSample) -> Sample:
    """重算本案原帖、传播树与完整解析输入SHA，禁止借用旧字符串。"""
    stem = item.sample_id.rsplit(":", 1)[-1]
    if not item.sample_id.startswith(f"ced:{item.category}:") or Path(stem).name != stem:
        raise ValueError("propagation_v11_sample_identity_invalid")
    data = root / "data/raw/CED_Dataset"
    folder = "rumor-repost" if item.category == "rumor" else "non-rumor-repost"
    if (_sha(data / "original-microblog" / f"{stem}.json") != item.original_sha256 or
            _sha(data / folder / f"{stem}.json") != item.propagation_sha256):
        raise ValueError("propagation_v11_selected_source_drift")
    sample = load_ced_sample(data, f"{stem}.json", rumor=item.category == "rumor")
    if sample.sample_id != item.sample_id or sample.source != "ced" or _input_fingerprint(sample) != item.input_sha256:
        raise ValueError("propagation_v11_selected_input_drift")
    return sample


class _CapacityReplay(llm.OfflineCompletionReplay):
    """在真实网关中只返回合成响应，同时捕获实际请求及保守预算检查。"""

    def __init__(self, responses: list[object], config: PropagationHarnessConfig) -> None:
        """绑定有限Mock响应，记录请求不回退到真实网络。"""
        super().__init__(responses)
        self.config = config
        self.requests: list[dict[str, object]] = []
        self.worst = BudgetTracker(config, get_cost_config())
        self.worst_reachable = True

    def create(self, **body: object) -> object:
        """捕获生产构造的完整请求，分别计算Mock与最坏预留口径。"""
        tools = body.get("tools", [])
        final = len(tools) == 1
        estimate = llm.estimate_request_input(body["messages"], tools,
                                              self.config.input_bound_margin_tokens)
        decision = self.worst.check_model_call(estimate, self.config.max_output_tokens, final=final)
        stage = "initial" if not self.requests else "forced_final" if final else (
            "after_tools" if len(self.requests) == 1 else "after_feedback")
        encoded = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        self.requests.append({"stage": stage, "estimated_input": estimate,
            "input_limit": self.config.token_budget_per_call, "output_limit": self.config.max_output_tokens,
            "request_sha256": sha256(encoded.encode("utf-8")).hexdigest(),
            "observed_on_fixed_mock_usage": True, "worst_case_budget_allowed": decision.allowed,
            "worst_case_budget_reason": decision.reason,
            "worst_case_path_reachable_before_request": self.worst_reachable})
        if decision.allowed:
            self.worst.tokens += estimate + self.config.max_output_tokens
            self.worst.cost_cny += Decimal(str(llm.compute_cost_cny(llm.TokenUsage(
                input_tokens=estimate, output_tokens=self.config.max_output_tokens), get_cost_config())))
            self.worst.steps += int(not final)
        else:
            self.worst_reachable = False
        return super().create(**body)


def _mock_completion(calls: list[tuple[str, dict[str, object]]], index: int) -> object:
    """构造固定可信用量的SDK响应对象，不创建客户端。"""
    functions = [{"id": f"capacity-{index}-{position}", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        for position, (name, args) in enumerate(calls)]
    return llm.ChatCompletion.model_validate({"id": f"capacity-{index}", "object": "chat.completion",
        "created": 1, "model": "offline-capacity", "choices": [{"index": 0,
        "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
        "tool_calls": functions}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20,
        "total_tokens": 120}})


def _capacity_responses(sample: Sample, *, final: bool) -> list[object]:
    """用完整提交参数构造纠错路径或真实工具限次触发的收尾路径。"""
    from src.yuqing.agents.propagation_v11 import available_fields

    raw = PropagationRaw.model_validate(sample.propagation.model_dump()) if sample.propagation else None
    fields = available_fields(get_spread_stats(raw))
    report = {"emotion_level": "低", "emotion_signals": [],
        "spread_level": "低" if fields else "未知", "spread_basis": list(fields[:1]),
        "narrative": "无法判断", "narrative_basis": [], "leads": [], "confidence": 0.5}
    calls = [("get_spread_stats", {}), ("sample_comments", {"order": "earliest",
        "k": get_propagation_tools_config().sample_max_k})]
    if final:
        calls.append(("get_spread_stats", {}))
        return [_mock_completion(calls, 0), _mock_completion([("submit_propagation_report", report)], 1)]
    invalid = {**report, "spread_basis": ["visible_nodes"]}
    return [_mock_completion(calls, 0), _mock_completion([("submit_propagation_report", invalid)], 1),
            _mock_completion([("submit_propagation_report", report)], 2)]


def measure_sample_capacity(sample: Sample, *, contract_version: Literal["v1.0", "v1.1"],
                             prepared_prompt: PreparedPropagationPrompt | None = None,
                             trace_root: Path | None = None) -> dict[str, object]:
    """经生产循环捕获纠错与收尾的真实Mock请求，不联网也不改变预算。"""
    from uuid import uuid4
    from src.yuqing.harness.tools import EvidenceWorkerScope

    root = get_settings().project_root
    prompt = prepared_prompt or validate_propagation_startup(
        root / "prompts/propagation/v1.md", offline=True)
    config = get_propagation_harness_config()
    scenarios = []
    for final in (False, True):
        replay = _CapacityReplay(_capacity_responses(sample, final=final), config)
        context = RunContext(run_id=uuid4().hex, sample_id=sample.sample_id,
            agent="propagation_analyst", trace_root=trace_root or root / "traces/review-packages/A-v1.1-1-capacity-v2")
        scope = EvidenceWorkerScope()
        try:
            with llm.use_offline_completion_replay(replay):
                result = propagation.run_propagation_agent(AgentSpec(agent="propagation_analyst",
                    system_prompt=prompt.text), sample, context, node_config=config,
                    worker_scope=scope, contract_version=contract_version)
        finally:
            scope.close()
        scenarios.append({"scenario": "forced_final" if final else "repair",
            "mock_success": result is not None, "trace_path": str(context.trace_path),
            "requests": replay.requests, "mock_input_tokens_each": 100, "mock_output_tokens_each": 20})
    return {"sample_id": sample.sample_id, "contract_version": contract_version,
        "scenarios": scenarios, "context_note": "真实循环请求+固定合成用量；不是提供商tokenizer或效果成绩。保守上界检查与Mock实际可达性分别报告。"}
