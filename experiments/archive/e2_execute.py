"""真实两组执行适配，复用已有预算路径；不提供付费许可或正式CLI。"""

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from src.yuqing import llm
from src.yuqing.eval.dataset import RetrievalQuestion
from experiments.archive.e2_artifacts import E2CheckpointRow, build_checkpoint_row
from experiments.archive.e2_checkpoint import CheckpointRun, PlannedRun
from experiments.archive.e2_dispatch import CoordinatorFactory, SlotExecutor
from experiments.archive.e2_single import run_single_with_terminal
from experiments.archive.e2_trace import read_and_audit_trace
from experiments.archive.e2_workspace import E2Workspace, _validate_custom_plan
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.harness.loop import run_agent
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.harness.trace import RunContext, TraceEvent, get_run_usage, use_run_context
from src.yuqing.retrieval.bm25 import BM25Index
from src.yuqing.retrieval.corpus import validate_eval_corpus
from src.yuqing.schemas import AgentSpec, TaskInput, FactVerdict, CitationRepairV2, AgentResult
from src.yuqing.settings import PROJECT_ROOT, get_settings


def _prompts(workspace: E2Workspace) -> dict[str, str]:
    """从已固定角色路径读原字节，逐项核对文件及实际系统消息指纹。"""
    conditions = workspace.conditions
    roles = json.loads(conditions["prompt_roles"])
    if not isinstance(roles, dict) or set(roles) != {"single", "loop"}:
        raise ValueError("invalid_executor_prompt_roles")
    texts: dict[str, str] = {}
    for role, relative in roles.items():
        path = (PROJECT_ROOT / relative).resolve()
        if not path.is_relative_to((PROJECT_ROOT / "prompts").resolve()):
            raise ValueError("executor_prompt_path_escape")
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != conditions["file:" + path.relative_to(PROJECT_ROOT.resolve()).as_posix()]:
            raise ValueError("executor_prompt_changed")
        texts[role] = raw.decode("utf-8")
    systems = {"single_system": llm.build_structured_system(texts["single"], FactVerdict),
        "repair_system": llm.build_structured_system(texts["single"], CitationRepairV2), "loop_system": texts["loop"]}
    if any(sha256(text.encode("utf-8")).hexdigest() != conditions[name + "_sha256"] for name, text in systems.items()):
        raise ValueError("executor_system_changed")
    return texts


def _pack(workspace: E2Workspace, session: ExperimentSession, slot: PlannedRun,
           question: RetrievalQuestion, ctx: RunContext, result: object, elapsed_ms: int) -> E2CheckpointRow:
    """以实际终局和钱包请求封装答卷，未知费用保留小计，不编造结束。"""
    if result is not None and not isinstance(result, (tuple, AgentResult)):
        raise ValueError("executor_invalid_result")
    if isinstance(result, AgentResult) and (result.run_id != ctx.run_id or result.trace_path != str(ctx.trace_path)):
        raise ValueError("executor_result_identity_mismatch")
    history = session.load_history()
    state = replay_history(history, workspace.manifest)
    ids = tuple(identifier for identifier, request in state.requests.items() if
        all(request.identity.get(key) == getattr(slot, key) for key in ("sample_id", "group_id", "run_id")))
    run = CheckpointRun(**slot.model_dump(), execution_status="finished", request_ids=ids)
    raw = ctx.trace_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("executor_trace_incomplete")
    events = [TraceEvent.model_validate_json(line) for line in raw.split(b"\n")[:-1]]
    terminal = [event for event in events if event.type == "run_outcome"]
    if len(terminal) != 1:
        raise ValueError("executor_terminal_missing_or_duplicate")
    success, reason = result is not None, terminal[0].data.get("reason")
    outcome = "submitted_judgment" if success else "no_valid_result"
    digest = sha256(raw).hexdigest()
    audit = read_and_audit_trace(ctx.trace_path, expected_sha256=digest, run=run, expected_agent=ctx.agent,
        expected_outcome=outcome, expected_reason=reason, single_shot=slot.group_id == "single_shot")
    if not audit.consistent:
        raise ValueError("executor_trace_not_verified")
    usage = get_run_usage(ctx.run_id)
    if isinstance(result, tuple):
        verdict, opinion, evidence = result
        saved = dict(verdict=verdict.model_dump(mode="json"), opinion=opinion.model_dump(mode="json"),
            evidence=[item.model_dump(mode="json") for item in evidence])
    elif isinstance(result, AgentResult):
        verdict, saved = result.verdict, result.model_dump(mode="json")
    else:
        verdict, saved = None, None
    gold = {"真": "支持", "假": "反驳", "无法判定": "信息不足"}[question.gold_veracity]
    payload = dict(success=success, result=saved, gold_label=gold,
        predicted_label=verdict.label if verdict is not None else "", correct=verdict is not None and verdict.label == gold,
        **usage.model_dump(), latency_ms=elapsed_ms, stop_reason=reason,
        trace=dict(path=ctx.trace_path.relative_to(PROJECT_ROOT).as_posix(), sha256=digest, outcome_class=outcome, reason=reason))
    return build_checkpoint_row(run, payload, experiment_id=workspace.manifest["experiment_id"],
        condition_sha256=workspace.manifest["conditions_fingerprint"])


def validate_executor_questions(questions: Sequence[RetrievalQuestion], conditions: Mapping[str, str],
                                plan: Sequence[PlannedRun] | None = None) -> None:
    """固定完整题目条件，仅对已许可的精确收尾计划允许覆盖题目子集。"""
    ids = [question.sample_id for question in questions]
    if not ids or len(set(ids)) != len(ids) or (
            [q.model_dump() for q in questions] != json.loads(conditions["questions"])):
        raise ValueError("executor_questions_mismatch")
    if plan is None:
        return
    planned_ids = list(dict.fromkeys(slot.sample_id for slot in plan))
    if conditions.get("recovery:final_segment") == "true":
        _validate_custom_plan(conditions, planned_ids, plan)
        if planned_ids != [sample for sample in ids if sample in set(planned_ids)]:
            raise ValueError("executor_questions_mismatch")
    elif ids != planned_ids:
        raise ValueError("executor_questions_mismatch")


def build_e2_executor(workspace: E2Workspace, session: ExperimentSession, questions: Sequence[RetrievalQuestion],
                       index: BM25Index, store_path: Path, worker_scope: EvidenceWorkerScope) -> SlotExecutor:
    """绑定固定题目和语料，返回调度器调用的同步真实执行函数。"""
    workspace.require_open()
    if dict(session.load_history().manifest) != workspace.manifest:
        raise ValueError("executor_session_mismatch")
    copied = [RetrievalQuestion.model_validate(q.model_dump()) for q in questions]
    validate_executor_questions(copied, workspace.conditions, workspace.plan)
    if store_path.resolve() != get_settings().dev_store_path.resolve() or workspace.conditions.get("split") != "dev":
        raise ValueError("executor_corpus_mismatch")
    validate_eval_corpus(index.chunks, store_path, corpus_name="chef_dev_full")
    _prompts(workspace)
    lookup = {q.sample_id: q for q in copied}
    def execute_slot(slot: PlannedRun, coordinator_factory: CoordinatorFactory) -> E2CheckpointRow:
        """使用原槽位身份调用单次或循环，已有Trace拒绝，不另领钱包或重试。"""
        workspace.require_open()
        if slot not in workspace.plan or not callable(coordinator_factory):
            raise ValueError("executor_invalid_slot_or_factory")
        texts, question = _prompts(workspace), lookup[slot.sample_id]
        ctx = RunContext(run_id=slot.run_id, sample_id=slot.sample_id,
            agent="fact" if slot.group_id == "single_shot" else "fact_investigator", trace_root=workspace.folder)
        if ctx.trace_path.exists():
            raise ValueError("executor_trace_already_exists")
        started = perf_counter()
        with use_run_context(ctx):
            if slot.group_id == "single_shot":
                result = run_single_with_terminal(question.claim, index, texts["single"], ctx,
                    store_path=store_path, coordinator_factory=coordinator_factory, experiment_identity=slot.model_dump())
            else:
                result = run_agent(AgentSpec(agent="fact_investigator", system_prompt=texts["loop"]),
                    TaskInput(sample_id=slot.sample_id, claim=question.claim), ctx, index=index,
                    worker_scope=worker_scope, store_path=store_path, coordinator_factory=coordinator_factory,
                    experiment_identity=slot.model_dump())
            return _pack(workspace, session, slot, question, ctx, result, int((perf_counter() - started) * 1000))
    return execute_slot
