"""Explicit sample-only activation binding; construction never invokes a model."""
from hashlib import sha256
from pathlib import Path
from src.yuqing.agents.propagation import run_propagation_agent
from src.yuqing.schemas import AgentSpec, TaskInput
from src.yuqing.settings import PolicyConfig, get_propagation_harness_config
from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask, PublicSignalResult
from src.yuqing.serve.public_propagation import PUBLIC_SLOTS

APPROVED_PROMPT_SHA = "679ad30f976f4e5f6d880b5ddb3a62c5c4a609a20d6190bb5d95b4b0a7499ec7"
APPROVED_URLS = frozenset((
    "https://www.news.cn/politics/20240913/fb547fc824734641a53b3ee5d73a123b/c.html",
    "https://www.dw.com/zh/6355-58%E4%B8%AD%E5%9B%BD%E5%BB%B6%E8%BF%9F%E9%80%80%E4%BC%91%E6%94%BF%E7%AD%96%E8%90%BD%E5%9C%B0/a-70211466"))


def validate_activation(task: PublicPropagationTask, prompt: Path, composition: object, root: Path) -> str:
    """Reject unapproved bytes, sources, identity or wallet before allocating any request."""
    text = prompt.read_bytes()
    if sha256(text).hexdigest() != APPROVED_PROMPT_SHA:
        raise ValueError("unapproved_public_prompt")
    if task.sample_id != composition.inputs.sample.sample_id or task.claim != composition.inputs.sample.claim:
        raise ValueError("public_activation_identity_mismatch")
    if task.batch.provenance != "recorded_response" or not task.batch.observations:
        raise ValueError("public_activation_requires_recorded_material")
    if any(str(row.url) not in APPROVED_URLS or row.source_kind != "news_report" for row in task.batch.observations):
        raise ValueError("unapproved_public_source")
    wallet = composition.inputs.wallet
    if wallet.storage_path.resolve() != (root / "continuation/wallet.sqlite").resolve():
        raise ValueError("public_wallet_binding_mismatch")
    from src.yuqing.serve.research_recovery import predecessor_hold
    import json
    from decimal import Decimal
    plan = json.loads((root / "continuation-plan.json").read_text(encoding="utf-8"))
    if plan["total_limit_cny"] != "5" or plan["predecessor"] != predecessor_hold(root):
        raise ValueError("public_continuation_mismatch")
    if wallet.snapshot().max_cost_cny != Decimal(plan["remaining_limit_cny"]) or wallet.snapshot().halted:
        raise ValueError("public_wallet_unavailable")
    return text.decode("utf-8")


def run_sample_increment(task: PublicPropagationTask, *, prompt: Path, composition: object,
                         root: Path, external_calls_authorized: bool = False) -> PublicSignalResult | None:
    """Run only the sample Agent using the composition's existing wallet and program identity."""
    if not external_calls_authorized:
        raise ValueError("public_external_calls_not_authorized")
    text = validate_activation(task, prompt, composition, root)
    call = composition.propagation_call({})
    names = [*PUBLIC_SLOTS, "submit_propagation_report"]
    policy = PolicyConfig(allow={"propagation_analyst": names}, risk_tier={name: "R0" for name in names}, r2_mode="deny")
    return run_propagation_agent(AgentSpec(agent="propagation_analyst", system_prompt=text),
        TaskInput(sample_id=task.sample_id, claim=task.claim), call.context,
        public_task=task, node_config=get_propagation_harness_config(),
        worker_scope=composition.inputs.propagation_worker_scope, policy_config=policy,
        coordinator_factory=composition.coordinator, experiment_identity=composition.identity(call.context))
