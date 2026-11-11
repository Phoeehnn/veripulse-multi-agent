"""两阶段开发先导的冻结输入与不含标注的消息构造。"""

from dataclasses import dataclass
from hashlib import sha256
from html import escape
import json
import platform
from importlib.metadata import version
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.yuqing.eval.evidence_scope import EvidenceScopeFactAdapter
from src.yuqing.eval.fact_v2_conditions import _manifest
from src.yuqing.harness.initial_evidence import InitialEvidenceItemV2, delivery_block
from src.yuqing.settings import PROJECT_ROOT, get_settings


FROZEN_INPUTS = {
    "docs/cards/scope-pair-v1-selection.json": "acba686948bd4d58d1925a5d62141176a1a236a120acc3875693eb10ab3ed195",
    "docs/cards/scope-pair-v1-materials.jsonl": "2a6b9d93b2cb37bce8680da6bf4824cf8b61188a2c2ae7e795205b93aa767aee",
    "docs/cards/scope-pair-v1-exposure.json": "0488532f4626f0b36d2e6327547106c65fd95a48bd59191314777076f54e9215",
    "experiments/archive/prompts/fact_check/scope_pair_self_review.draft.md": "e9bd046004e47dff3f410e2a4a89c671e4cbd49fa5c132d266b4edaf3a200df7",
    "experiments/archive/prompts/fact_check/scope_pair_peer_review.draft.md": "52d1c1dd222e83a9105b84ea600be481e943be1167f495996ae40176bc6f65a9",
}


class ScopePacket(BaseModel):
    """模型可见材料的唯一入口；额外标注字段一律拒绝。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    claim: str
    entries: tuple[InitialEvidenceItemV2, ...] = Field(min_length=1, max_length=5)


@dataclass(frozen=True)
class ScopeInputs:
    """捕获一次的材料和Prompt；标注题单不进入执行对象。"""

    root: Path
    packets: tuple[ScopePacket, ...]
    system: str
    reviews: tuple[str, str]
    fingerprints: tuple[tuple[str, str], ...]


def capture_inputs(root: Path = PROJECT_ROOT) -> ScopeInputs:
    """核验批准快照、顺序和原始身份，拒绝原地重抽或标签混入。"""
    raw = {path: (root / path).read_bytes() for path in FROZEN_INPUTS}
    if any(sha256(raw[path]).hexdigest() != expected for path, expected in FROZEN_INPUTS.items()):
        raise ValueError("scope_pair_frozen_input_drift")
    roster = json.loads(raw["docs/cards/scope-pair-v1-selection.json"])
    packets = tuple(ScopePacket.model_validate_json(line) for line in
                    raw["docs/cards/scope-pair-v1-materials.jsonl"].splitlines())
    if ([p.sample_id for p in packets] != [r["sample_id"] for r in roster["rows"]]
            or len(packets) != 12 or len({p.sample_id for p in packets}) != 12):
        raise ValueError("scope_pair_roster_mismatch")
    for packet, row in zip(packets, roster["rows"], strict=True):
        if sha256(packet.claim.encode()).hexdigest() != row["claim_sha256"]:
            raise ValueError("scope_pair_claim_mismatch")
    adapter = EvidenceScopeFactAdapter(root=root)
    reviews = tuple(raw[f"experiments/archive/prompts/fact_check/scope_pair_{name}_review.draft.md"].decode()
                    for name in ("self", "peer"))
    tracked = tuple(FROZEN_INPUTS) + ("config/weights.yaml", "config/fact_v2_eval.yaml",
        "prompts/fact_check/investigator_v2.md", "experiments/archive/prompts/fact_check/evidence_scope_playbook.draft.md",
        "experiments/archive/scope_pair_inputs.py", "experiments/archive/scope_pair_runner.py",
        "experiments/archive/scope_pair_audit.py", "src/yuqing/llm.py",
        "src/yuqing/harness/budget.py", "src/yuqing/harness/fact_v2_contract.py",
        "src/yuqing/eval/dual_budget.py", "src/yuqing/eval/experiment_session.py",
        "src/yuqing/eval/persistent_experiment_budget.py", "src/yuqing/eval/experiment_budget_recovery.py",
        "src/yuqing/eval/experiment_budget_store.py", "src/yuqing/eval/experiment_budget.py",
        "src/yuqing/eval/fact_v2_experiment.py", "src/yuqing/harness/quote_verifier.py",
        "src/yuqing/harness/verifier.py", "src/yuqing/harness/initial_evidence.py",
        "src/yuqing/harness/feedback.py", "src/yuqing/schemas.py")
    fingerprints = tuple((p, sha256((root / p).read_bytes()).hexdigest()) for p in tracked)
    return ScopeInputs(root, packets, adapter.spec.system_prompt, reviews,
                       fingerprints + (("gateway", gateway_fingerprint()),))


def gateway_fingerprint() -> str:
    """只捕获模型参数，绝不记录密钥或认证头。"""
    settings = get_settings()
    runtime = [platform.python_version(), {name: version(name) for name in
        ("openai", "httpx", "pydantic", "pydantic-settings", "langchain-core", "langgraph")}]
    return sha256(json.dumps([settings.deepseek_model, settings.deepseek_base_url,
        settings.llm_thinking, settings.llm_timeout_s, {"sdk_max_retries": 0}, runtime],
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def assert_unchanged(inputs: ScopeInputs) -> None:
    """每次发送前核冻结字节和环境参数漂移，不自动刷新指纹。"""
    for path, expected in inputs.fingerprints:
        actual = gateway_fingerprint() if path == "gateway" else sha256((inputs.root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"scope_pair_conditions_drift:{path}")


def initial_messages(inputs: ScopeInputs, packet: ScopePacket) -> list[dict[str, object]]:
    """共享完整材料包，仅交付E编号与正文，不传内部来源或Gold。"""
    manifest = _manifest(packet.sample_id, packet.claim, packet.entries)
    block = delivery_block(manifest, [f"E{i}" for i in range(1, len(packet.entries) + 1)])
    return [{"role": "system", "content": inputs.system}, {"role": "user",
             "content": f"<claim>{escape(packet.claim, quote=False)}</claim>\n{block}"}]


def review_messages(inputs: ScopeInputs, initial: list[dict[str, object]], arm: str,
                    assistant: dict[str, object]) -> list[dict[str, object]]:
    """S保留工具对话，M新建上下文；初稿从不变成证据。"""
    if arm == "S":
        call_id = assistant["tool_calls"][0]["id"]
        return [*initial, assistant, {"role": "tool", "tool_call_id": call_id,
            "content": '{"accepted":true}'}, {"role": "user", "content": inputs.reviews[0]}]
    data = escape(json.dumps({"assistant_message": assistant,
        "program_submission_receipt": {"accepted": True}}, ensure_ascii=False), quote=False)
    return [*initial, {"role": "user", "content": inputs.reviews[1] +
        f"\n<previous_submission>{data}</previous_submission>"}]
