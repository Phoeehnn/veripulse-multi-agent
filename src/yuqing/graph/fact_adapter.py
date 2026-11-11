"""将冻结的事实 V2 循环适配到案件图，并在启动时核验冻结合同。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping

from pydantic import TypeAdapter

from src.yuqing.harness.fact_v2_contract import (
    FactRunContextV2,
    build_fact_run_context_v2,
)
from src.yuqing.harness.loop import run_agent
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.collab.fact_material import FactArchiveReaderV2
from src.yuqing.collab.arbiter_material_v2 import FactArchiveInputV2
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.fact_v2_experiment import build_fact_v2_node_config
from src.yuqing.schemas import (
    AgentResultV2,
    AgentSpec,
    FactScopeV2,
    FollowupScopeV2,
    InheritedEvidencePlanV2,
    ProgramActionClaimV2,
    TaskInput,
    ValidatedFactSnapshotV2,
)
from src.yuqing.settings import (
    PROJECT_ROOT,
    get_graph_config,
    get_harness_config,
    get_fact_v2_evaluation_config,
)
from src.yuqing.harness.trace import RunContext
from src.yuqing.harness.trace import TraceEvent


_FROZEN_FACT_FILES = {
    "src/yuqing/harness/fact_v2_contract.py": "caaf679f9c2f73caace95a81a0f61785a86391bcc5927f3f11f30eacbe02e6eb",
    "src/yuqing/harness/feedback.py": "9f48164338ac276f081cdc03543659bd6707386e52df5c7b8c4b5b2d4566799d",
    "src/yuqing/harness/verifier.py": "f16380b927363f44425aa410873b37cd4d2cf28c1ba1e6ceb293c30e0a46ab4d",
    "src/yuqing/harness/quote_verifier.py": "0dc96d46561b6803fd770576cd6826f1c79b67c4a3f0c341af6245c0fc67294b",
    "config/fact_v2_eval.yaml": "df48ea2a71eada336d7b78d0e59e1079384eecbcdf8ff94ae3c7d62264c9fa35",
}
_PROMPT_PATH = "prompts/fact_check/investigator_v2.md"
_MANIFEST_PATH = "prompts/fact_check/v2_manifest.json"


@dataclass(frozen=True)
class FactNodeCall:
    """一轮事实调查所需的显式依赖，不将运行时对象塞进图状态。"""

    task: TaskInput
    case_id: str
    trace: RunContext
    index: BM25Index
    worker_scope: EvidenceWorkerScope
    store_path: Path
    initial_manifest_path: Path
    coordinator_factory: Callable | None
    experiment_identity: Mapping[str, str] | None


@dataclass(frozen=True)
class FactNodeFollowup:
    """已通过仲裁门卫的补查范围及其程序证明。"""

    call: FactNodeCall
    scope: FollowupScopeV2
    inherited_plan: InheritedEvidencePlanV2
    action_claim: ProgramActionClaimV2


@dataclass(frozen=True)
class FactRoundResult:
    """图节点可安全保存的事实结果摘要及追踪引用。"""

    report: dict[str, object] | None
    status: str
    stop_reason: str
    ledger_refs: tuple[str, ...]
    delivery_refs: tuple[str, ...]
    archive_path: str | None
    archive_sha256: str | None
    usage: dict[str, object]


@dataclass(frozen=True)
class FactRuntimeArtifact:
    """Non-State verified archive objects retained by the per-invocation context."""

    summary: FactRoundResult
    archive_input: FactArchiveInputV2
    snapshot: ValidatedFactSnapshotV2


class FactNodeAdapter:
    """只适配冻结 V2；V1.1 入口显式失败，不静默回退。"""

    def __init__(self, version: str, *, root: Path = PROJECT_ROOT) -> None:
        """核验选择版本、冻结提示词、Schema 和事实专用依赖。"""
        if version != "v2":
            raise NotImplementedError("fact_node_version_not_implemented")
        self.root = root.resolve()
        self._verify_freeze()
        self.spec = AgentSpec(agent="fact_investigator",
                              system_prompt=(self.root / _PROMPT_PATH).read_text(encoding="utf-8"))

    def run_round1(self, call: FactNodeCall) -> FactRoundResult:
        """运行冻结 V2 首轮并转成图节点的可序列化摘要。"""
        artifact = self.run_round1_runtime(call)
        return artifact.summary if artifact else FactRoundResult(
            None, "failed", "no_valid_result", (), (), None, None, {})

    def run_round1_runtime(self, call: FactNodeCall, *, wallet: object | None = None
                           ) -> FactRuntimeArtifact | None:
        """Run and independently archive-check the first-round result for this invoke."""
        scope = _initial_scope(call)
        contract = build_fact_run_context_v2("initial", initial_manifest_path=call.initial_manifest_path,
                                             scope=scope)
        return self._run_runtime(call, contract, scope, wallet)

    def run_followup(self, context: FactNodeFollowup) -> FactRoundResult:
        """只运行与已校验动作、证据继承计划绑定的 V2 补查。"""
        call = context.call
        scope = context.scope
        if (scope.case_id != call.case_id or scope.sample_id != call.task.sample_id
                or scope.run_id != call.trace.run_id or scope.agent != call.trace.agent):
            raise ValueError("fact_followup_identity_mismatch")
        contract = build_fact_run_context_v2(
            "followup", assigned_issue_ids=frozenset(scope.issue_ids), scope=scope,
            inherited_plan=context.inherited_plan, action_claim=context.action_claim,
        )
        return self._run(call, contract)

    def run_followup_runtime(self, context: FactNodeFollowup, *, wallet: object | None = None
                             ) -> FactRuntimeArtifact | None:
        """Run a follow-up and retain only archive-verified objects outside graph State."""
        call, scope = context.call, context.scope
        contract = build_fact_run_context_v2("followup", assigned_issue_ids=frozenset(scope.issue_ids),
            scope=scope, inherited_plan=context.inherited_plan, action_claim=context.action_claim)
        return self._run_runtime(call, contract, scope, wallet)

    def _run(self, call: FactNodeCall, contract: FactRunContextV2) -> FactRoundResult:
        """调用既有有界循环，并保留已校验报告、证据编号和使用量。"""
        result = self._execute(call, contract)
        return _result_summary(result, contract, call.trace)

    def _execute(self, call: FactNodeCall, contract: FactRunContextV2) -> AgentResultV2 | None:
        """Call the existing bounded fact loop with its frozen contract and dependencies."""
        result = run_agent(self.spec, call.task, call.trace, index=call.index,
            worker_scope=call.worker_scope, store_path=call.store_path,
            coordinator_factory=call.coordinator_factory,
            experiment_identity=call.experiment_identity,
            node_config=build_fact_v2_node_config(get_harness_config(),
                get_fact_v2_evaluation_config()), fact_contract=contract)
        if result is not None and not isinstance(result, AgentResultV2):
            raise ValueError("fact_adapter_received_non_v2_result")
        return result

    def _run_runtime(self, call: FactNodeCall, contract: FactRunContextV2,
                     scope: FactScopeV2 | FollowupScopeV2, wallet: ExperimentSession | None
                     ) -> FactRuntimeArtifact | None:
        """Keep rich run objects in the invocation sidecar, never in serializable State."""
        result = self._execute(call, contract)
        summary = _result_summary(result, contract, call.trace)
        if result is None:
            return None
        reader = FactArchiveReaderV2(call.trace.trace_root, call.initial_manifest_path, wallet)
        snapshot = reader.read(result, call.task, contract, scope)
        archive_input = FactArchiveInputV2(reader=reader, result=result, task=call.task,
            contract=contract, scope=scope)
        return FactRuntimeArtifact(summary, archive_input, snapshot)

    def _verify_freeze(self) -> None:
        """拒绝许可、提示词、Schema 或事实专用代码指纹漂移。"""
        manifest = TypeAdapter(dict[str, object]).validate_json(
            (self.root / _MANIFEST_PATH).read_bytes()
        )
        prompts = manifest.get("prompts")
        if not isinstance(prompts, list):
            raise ValueError("fact_v2_manifest_prompts_invalid")
        record = next((item for item in prompts if isinstance(item, dict)
                       and item.get("path") == _PROMPT_PATH), None)
        prompt_path = self.root / _PROMPT_PATH
        digest = sha256(prompt_path.read_bytes()).hexdigest()
        if (not isinstance(record, dict) or record.get("approved_for_dev") is not True
                or record.get("prompt_sha256") != digest):
            raise ValueError("fact_v2_prompt_not_frozen_or_licensed")
        initial = build_fact_run_context_v2(
            "initial", initial_manifest_path=Path("frozen-contract-placeholder.json")
        )
        followup = build_fact_run_context_v2("followup", assigned_issue_ids=frozenset({"I1"}))
        if (manifest.get("initial_schema_sha256") != initial.schema_sha256
                or manifest.get("followup_schema_sha256") != followup.schema_sha256):
            raise ValueError("fact_v2_schema_fingerprint_mismatch")
        for relative, expected in _FROZEN_FACT_FILES.items():
            if not _matches_frozen_text((self.root / relative).read_bytes(), expected):
                raise ValueError(f"fact_v2_dependency_fingerprint_mismatch:{relative}")
        if get_graph_config().fact_node_version != "v2":
            raise ValueError("fact_node_version_not_implemented")


def _matches_frozen_text(raw: bytes, expected: str) -> bool:
    """只兼容文本LF/CRLF检出差异，保留历史指纹并拒绝其他字节变化。"""
    lf = raw.replace(b"\r\n", b"\n")
    return expected in {sha256(value).hexdigest() for value in
                        (raw, lf, lf.replace(b"\n", b"\r\n"))}


def _result_summary(result: AgentResultV2 | None, contract: FactRunContextV2,
                    context: RunContext) -> FactRoundResult:
    """转换受校验结果；失败时不制造结论或引用。"""
    if result is None:
        return FactRoundResult(None, "failed", "no_valid_result", (), (), None, None, {})
    if not isinstance(result, AgentResultV2):
        raise ValueError("fact_adapter_received_non_v2_result")
    parsed = contract.schema.model_validate_json(result.report_json, strict=True)
    report = parsed.model_dump(mode="json")
    refs = tuple(item.evidence_id for item in result.evidence)
    archive_path, archive_sha = _submission_archive(context)
    if archive_path is None or archive_sha is None:
        raise ValueError("fact_submission_archive_missing")
    return FactRoundResult(report, "completed", result.stop_reason, refs, refs,
        archive_path, archive_sha, {"steps": result.steps,
            "input_output_tokens": result.input_output_tokens, "cost_cny": result.cost_cny})


def _initial_scope(call: FactNodeCall) -> FactScopeV2:
    """Bind the first round to one case, sample, trace run, and exact claim text."""
    return FactScopeV2(case_id=call.case_id, sample_id=call.task.sample_id,
        run_id=call.trace.run_id, agent=call.trace.agent, fact_round_no=1,
        claim_sha256=sha256(call.task.claim.encode("utf-8")).hexdigest())


def _submission_archive(context: RunContext) -> tuple[str | None, str | None]:
    """从程序Trace定位并校验通过的提交归档，不读取模型原始文本。"""
    if not context.trace_path.is_file():
        return None, None
    for line in context.trace_path.read_text(encoding="utf-8").splitlines():
        event = TraceEvent.model_validate_json(line, strict=True)
        archive = event.data.get("archive")
        if event.type != "tool_call" or not isinstance(archive, dict):
            continue
        if event.data.get("tool") != "submit_verdict" or event.data.get("decision") != "allow":
            continue
        relative, expected = archive.get("path"), archive.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("fact_submission_archive_identity_missing")
        root = context.trace_root.resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("fact_submission_archive_path_invalid")
        actual = sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("fact_submission_archive_hash_mismatch")
        return str(target), actual
    return None, None
