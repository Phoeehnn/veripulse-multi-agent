"""组装事实 V2 两组共用的首请求并做共同容量检查。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from html import escape

from src.yuqing import llm
from src.yuqing.harness.fact_v2_contract import FactRunContextV2, build_fact_run_context_v2
from src.yuqing.harness.fact_v2_messages import build_fact_v2_message_history
from src.yuqing.harness.initial_evidence import (
    InitialEvidenceItemV2, InitialEvidenceManifestV2, delivery_block,
)
from src.yuqing.harness.tools import ToolRegistry, register_evidence_tools, register_v2_submission_tool
from src.yuqing.settings import (
    FactV2EvaluationConfig, get_harness_config, get_policy_config,
)


@dataclass(frozen=True)
class FactV2InitialRequests:
    """两组真实首请求估算所需的系统消息、用户消息与工具定义。"""

    single_messages: list[dict[str, object]]
    single_tools: list[dict[str, object]]
    loop_messages: list[dict[str, object]]
    loop_tools: list[dict[str, object]]
    single_input_estimate: int
    loop_input_estimate: int


@dataclass(frozen=True)
class CommonEvidenceCapacity:
    """共同容量裁减后的完整证据条目与两组请求估算。"""

    entries: tuple[InitialEvidenceItemV2, ...]
    single_input_estimate: int
    loop_input_estimate: int
    capacity_exceeded: bool


def _manifest(sample_id: str, claim: str,
              entries: tuple[InitialEvidenceItemV2, ...]) -> InitialEvidenceManifestV2:
    """构造只用于首请求组装的有效内存清单，不写入磁盘。"""
    payload = [entry.model_dump(mode="json") for entry in entries]
    entries_sha = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()
    return InitialEvidenceManifestV2(manifest_version="fact-v2-initial-v1", sample_id=sample_id,
        claim_sha256=sha256(claim.encode("utf-8")).hexdigest(), retrieval_sha256="0" * 64,
        corpus_sha256="0" * 64, limits_sha256="0" * 64, entries_sha256=entries_sha,
        retrieval_status="success" if entries else "empty", original_candidate_count=len(entries),
        delivered_count=len(entries), entries=list(entries))


def _messages(claim: str, block: str, system: str) -> list[dict[str, object]]:
    """按 Harness 首轮的真实角色顺序构造模型消息。"""
    user = f"<claim>{escape(claim, quote=False)}</claim>\n{block}"
    return build_fact_v2_message_history(system, user)


def _tool_definitions(context: FactRunContextV2) -> list[dict[str, object]]:
    """复用循环实际注册的搜索、阅读、提交定义和权限过滤。"""
    registry = ToolRegistry()
    register_evidence_tools(registry, config=get_harness_config())
    register_v2_submission_tool(registry, context.schema, context.response_schema)
    return registry.model_tools("fact_investigator", get_policy_config())


def _loop_status_message(tools: list[dict[str, object]],
                         entries: tuple[InitialEvidenceItemV2, ...]) -> dict[str, str]:
    """生成与事实循环首轮相同的程序状态说明，不把状态伪装成证据。"""
    quotas = []
    configured = get_harness_config().max_tool_calls
    for item in tools:
        name = str(item["function"]["name"])
        if name in configured:
            limit = configured[name]
            quotas.append(f"{name}剩余{limit}/{limit}")
    text = "程序状态（不是证据）：" + ("；".join(quotas) or "当前没有调查工具额度")
    readable = "、".join(f"E{index}" for index in range(1, len(entries) + 1)) or "无"
    text += f"\n本次请求已附带且成功阅读的证据编号：{readable}。搜索预览不属于可引用正文。"
    return {"role": "user", "content": text}


def build_fact_v2_initial_requests(
    sample_id: str, claim: str, entries: tuple[InitialEvidenceItemV2, ...],
    single_prompt: str, loop_prompt: str, capacity: FactV2EvaluationConfig,
) -> FactV2InitialRequests:
    """组装单次JSON和循环工具首请求，确保两者收到相同证据数据块。"""
    manifest = _manifest(sample_id, claim, entries)
    evidence_ids = [f"E{index}" for index, _ in enumerate(entries, start=1)]
    block = delivery_block(manifest, evidence_ids)
    contract = build_fact_run_context_v2("initial", initial_manifest_path=Path("in-memory-manifest.json"))
    single_system = llm.build_structured_system(single_prompt, contract.response_schema)
    single_messages = _messages(claim, block, single_system)
    loop_tools = _tool_definitions(contract)
    loop_messages = _messages(claim, block, loop_prompt)
    loop_messages.append(_loop_status_message(loop_tools, entries))
    single_estimate = llm.estimate_request_input(single_messages, [], capacity.input_bound_margin_tokens)
    loop_estimate = llm.estimate_request_input(loop_messages, loop_tools, capacity.input_bound_margin_tokens)
    return FactV2InitialRequests(single_messages, [], loop_messages, loop_tools,
                                 single_estimate, loop_estimate)


def fit_common_initial_evidence(
    sample_id: str, claim: str, candidates: tuple[InitialEvidenceItemV2, ...],
    single_prompt: str, loop_prompt: str, capacity: FactV2EvaluationConfig,
) -> CommonEvidenceCapacity:
    """只从排名末尾移除整段，直到两组输入都在单次上限内。"""
    entries = candidates
    while True:
        requests = build_fact_v2_initial_requests(sample_id, claim, entries,
                                                 single_prompt, loop_prompt, capacity)
        fits = max(requests.single_input_estimate, requests.loop_input_estimate) <= capacity.token_budget_per_call
        if fits or not entries:
            return CommonEvidenceCapacity(entries, requests.single_input_estimate,
                requests.loop_input_estimate, not fits)
        entries = entries[:-1]
