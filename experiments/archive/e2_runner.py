"""有界E2入口组装；复用原调度、钱包和执行器，不实现新的Agent循环。"""
import argparse
from contextlib import ExitStack
from datetime import date
from decimal import Decimal
from functools import partial
import json
import logging
from pathlib import Path
import subprocess
from typing import Callable
from uuid import uuid4

from src.yuqing.eval.dataset import align_question_gold, load_questions
from experiments.archive.e2_conditions import (collect_e2_conditions, assert_e2_conditions_unchanged,
    load_pilot_manifest, write_pilot_manifest)
from experiments.archive.e2_dispatch import dispatch_next, DispatchResult
from experiments.archive.e2_continuation import inspect_continuation_parent
from experiments.archive.e2_continuation_run import (continuation_ceiling, claim_continuation,
    check_continuation_claim, publish_continuation_summary)
from experiments.archive.e2_recovery import (FinalRecovery, inspect_final_recovery, final_recovery_conditions,
    recovery_plan_sha256, claim_final_segment, assert_final_segment_claim_available,
    check_final_segment_claim, publish_final_recovery_summary, zero_start_freeze)
from experiments.archive.e2_execute import build_e2_executor, validate_executor_questions
from experiments.archive.e2_journal import AnswerJournal
from experiments.archive.e2_workspace import open_e2_workspace, E2Workspace
from src.yuqing.eval.experiment_session import ExperimentSession, build_experiment_manifest
from src.yuqing.eval.experiment_budget import ExperimentBudget
from src.yuqing.eval.preflight import validate_fact_check_startup, validate_loop_startup
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.llm import check_provider_connection
from src.yuqing.retrieval.bm25 import build_bm25_index, BM25Index
from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus
from src.yuqing.retrieval.search import search
from src.yuqing.settings import (PROJECT_ROOT, configure_model_cache, get_settings,
    get_evaluation_config, get_cost_config, get_harness_config)

logger = logging.getLogger(__name__)


def _experiment_budget_hit(workspace: E2Workspace, journal: AnswerJournal, session: ExperimentSession) -> bool:
    """余额用尽或最新Trace记录整场预占被拒时停止，不继续制造零调用失败题。"""
    wallet, rows = session.snapshot(), journal.read_answers()
    if wallet.committed_cost_cny >= wallet.max_cost_cny:
        return True
    if not rows:
        return False
    trace = Path(rows[-1].payload["trace"]["path"])
    raw = (workspace.folder / trace.parent.name / trace.name).read_bytes()
    events = [TraceEvent.model_validate_json(line) for line in raw.split(b"\n") if line]
    return any(event.type in {"stop", "budget"} and event.data.get("reason") == "experiment_cost_limit" for event in events)


def code_revision() -> str:
    """正式E2只能在已验收标签运行；拒绝源码、配置和提示词未提交变化。"""
    def git(*args: str) -> str:
        """只读取本项目Git信息，不从参数拼接shell命令。"""
        return subprocess.run(["git", *args],
            cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    revision = git("rev-parse", "HEAD")
    tags = git("tag", "--points-at", revision).splitlines()
    if not any(tag == "e2-ready" or tag.startswith("e2-ready.") for tag in tags):
        raise ValueError("e2_ready_tag_required")
    if git("status", "--porcelain", "--", "src", "scripts", "config", "prompts", "requirements.txt", "pyproject.toml"):
        raise ValueError("e2_execution_files_dirty")
    return revision


def drive_e2(workspace: E2Workspace, journal: AnswerJournal, session: ExperimentSession,
             execute_slot: Callable, check_conditions: Callable[[], None], scope: EvidenceWorkerScope) -> DispatchResult:
    """逐份驱动旧调度器；线程未退先有界等待，故障不自动重跑。"""
    while True:
        workspace.heartbeat()
        try:
            check_conditions()
        except Exception as exc:
            logger.error("条件检查失败 type=%s", type(exc).__name__)
            session.halt("experiment_conditions_changed")
            return DispatchResult(status="failed", reason="experiment_conditions_changed")
        before = journal.audit_saved_answers(session.load_history())
        if not before.consistent:
            return DispatchResult(status="blocked", reason="resume_audit_blocked", audit=before)
        if before.next_run_id is None:
            return DispatchResult(status="finished", reason="plan_finished", audit=before)
        if _experiment_budget_hit(workspace, journal, session):
            session.halt("experiment_cost_limit")
            return DispatchResult(status="blocked", reason="experiment_cost_limit", audit=before)
        if any(workspace.folder.glob(f"*/{before.next_run_id}.jsonl")):
            session.halt("orphan_trace_requires_review")
            return DispatchResult(status="blocked", reason="orphan_trace_requires_review", audit=before)
        if not scope.wait_until_idle(get_evaluation_config().worker_wait_timeout_s):
            session.halt("background_worker_wait_timeout")
            return DispatchResult(status="blocked", reason="background_worker_wait_timeout", audit=before)
        result = dispatch_next(workspace, journal, session, check_conditions=check_conditions, execute_slot=execute_slot)
        logger.info("E2 %d/%d run=%s status=%s reason=%s known_cost=%s",
            len(journal.read_answers()), len(workspace.plan), result.run_id, result.status, result.reason,
            session.snapshot().known_cost_subtotal_cny)
        if result.status != "saved":
            return result


def prepare_e2(args: argparse.Namespace) -> tuple[list, BM25Index, dict[str, str], dict]:
    """先查许可，再固定题目与完整语料；预热不进入任何题的运行或收费。"""
    settings = get_settings()
    single = validate_fact_check_startup(PROJECT_ROOT / "prompts/fact_check/v3.md")
    requested_loop_path = getattr(args, "loop_prompt", PROJECT_ROOT / "prompts/fact_check/investigator_v1.md")
    loop_path = requested_loop_path if requested_loop_path.is_absolute() else PROJECT_ROOT / requested_loop_path
    loop_path = loop_path.resolve()
    loop = validate_loop_startup(loop_path)
    revision = code_revision()
    paths = [PROJECT_ROOT / "eval/questions_30.jsonl", PROJECT_ROOT / "eval/dev_extra_60.jsonl"]
    fixed = [q for path in paths for q in load_questions(path)]
    final_recovery = inspect_final_recovery() if getattr(args, "finish_final_segment", False) else None
    selected = load_pilot_manifest(args.manifest, fixed, paths) if args.mode == "pilot" else fixed
    parent_path = getattr(args, "continue_from", None)
    if final_recovery and parent_path:
        raise ValueError("final_recovery_cannot_use_continue_from")
    if parent_path and args.mode != "full":
        raise ValueError("continuation_requires_full_mode")
    if parent_path:
        selected = fixed[inspect_continuation_parent(parent_path, [q.sample_id for q in fixed]).finished_pairs:]
    questions = align_question_gold(settings.chef_dev_path, selected, get_evaluation_config().gold_review_path)
    configure_model_cache()
    chunks = load_eval_corpus(settings.chef_dev_path, questions, corpus_name="chef_dev_full")
    validate_eval_corpus(chunks, settings.dev_store_path, corpus_name="chef_dev_full")
    index = build_bm25_index(chunks)
    search(questions[0].claim, index, path=settings.dev_store_path)
    options = dict(question_paths=paths, single_prompt=single.path, loop_prompt=loop_path,
        store_path=settings.dev_store_path, pilot_manifest=args.manifest if args.mode == "pilot" else None)
    if parent_path:
        options["continuation_parent"] = parent_path
    conditions = collect_e2_conditions(questions, **options)
    conditions.update(code_revision=revision, loop_contract_version=loop.contract_version)
    if final_recovery:
        conditions.update(final_recovery_conditions(final_recovery, conditions))
        options.update(planned_runs=final_recovery.remaining, final_recovery=final_recovery)
        if getattr(args, "replace_zero_start", None) is not None:
            conditions["recovery:zero_start"] = zero_start_freeze(args.replace_zero_start)["folder"]
    return questions, index, conditions, options


def require_e2_provider_connection() -> None:
    """正式E2启动前验证HTTPS可达，失败时不进入实验写入流程。"""
    settings = get_settings()
    reachable, stage, status = check_provider_connection(settings.deepseek_base_url, settings.llm_timeout_s)
    if not reachable:
        logger.error("E2网络启动检查失败 stage=%s；未创建实验目录、续评认领或钱包", stage)
        raise ValueError(f"provider_preflight_{stage}")
    logger.info("E2网络启动检查通过 status=%s；仅证明HTTPS端点可达，不验证密钥或模型", status)


def _check_e2_runtime(workspace: E2Workspace, conditions: dict, questions: list, options: dict,
                      final_recovery: FinalRecovery | None = None) -> None:
    """每份前后核对实际条件、冻结提交和子场认领，不以空回调绕过。"""
    expected = {k: v for k, v in conditions.items()
        if k not in {"code_revision", "loop_contract_version"} and not k.startswith("recovery:")}
    assert_e2_conditions_unchanged(expected, questions, **options)
    if code_revision() != conditions["code_revision"]:
        raise ValueError("e2_code_revision_changed")
    validate_loop_startup(options["loop_prompt"])
    if final_recovery:
        current = inspect_final_recovery()
        if current.source_sha256 != final_recovery.source_sha256 or current.remaining != final_recovery.remaining:
            raise ValueError("recovery_source_plan_changed")
        check_final_segment_claim(final_recovery, workspace.folder, workspace.manifest["experiment_id"],
            workspace.manifest["conditions_fingerprint"], Decimal(workspace.manifest["max_cost_cny"]),
            PROJECT_ROOT / conditions["recovery:zero_start"] if "recovery:zero_start" in conditions else None)
    else:
        check_continuation_claim(workspace)


def _execute_e2_workspace(args: argparse.Namespace, questions: list, index: BM25Index,
                          conditions: dict[str, str], options: dict, manifest: dict[str, object],
                          folder: Path, confirmation: str | None, planned_runs: tuple | None,
                          final_recovery: FinalRecovery | None, prices: object, config: object,
                          max_run_cost_cny: Decimal | None) -> int:
    """接上固定工作区、钱包、答卷、调度与分段发布。"""
    with ExitStack() as stack:
        workspace = stack.enter_context(open_e2_workspace(folder, mode="resume" if args.resume else "new",
            manifest=manifest, conditions=conditions,
            sample_ids=list(dict.fromkeys(slot.sample_id for slot in planned_runs)) if planned_runs else
                [q.sample_id for q in questions], confirmed_owner_sha256=confirmation, planned_runs=planned_runs))
        if final_recovery:
            claim_final_segment(final_recovery, folder, manifest["experiment_id"],
                manifest["conditions_fingerprint"], Decimal(manifest["max_cost_cny"]), resume=bool(args.resume),
                replacement=getattr(args, "replace_zero_start", None))
        claim_continuation(workspace, resume=bool(args.resume))
        factory = ExperimentSession.resume if args.resume else ExperimentSession.create
        session = factory(folder / "wallet.sqlite", manifest["experiment_id"], conditions, prices, config,
            mode=args.mode, max_run_cost_cny=max_run_cost_cny)
        stack.callback(session.close)
        scope = EvidenceWorkerScope()
        stack.callback(scope.close)
        journal = AnswerJournal.open(workspace, mode="resume" if args.resume else "new")
        execute = build_e2_executor(workspace, session, questions, index, get_settings().dev_store_path, scope)
        check = partial(_check_e2_runtime, workspace, conditions, questions, options, final_recovery)
        result = drive_e2(workspace, journal, session, execute, check, scope)
        try:
            check()
        except Exception:
            session.halt("experiment_conditions_changed")
            result = DispatchResult(status="failed", reason="experiment_conditions_changed")
        from experiments.archive.e2_report import publish_e2_results
        files = publish_e2_results(workspace, journal, session, result)
        if final_recovery and {"single_shot", "iterative"} <= files.keys():
            files["three_segment_summary"] = publish_final_recovery_summary(
                final_recovery, workspace, journal, session, files, getattr(args, "replace_zero_start", None))
        else:
            publish_continuation_summary(workspace, journal, session, files)
        logger.info("E2停止=%s 结果=%s", result.reason, files)
        return 0 if result.status == "finished" else 1


def run_e2_experiment(args: argparse.Namespace) -> int:
    """新建/恢复同一场钱包，末尾只发布已审计答案，不开正式下一场。"""
    confirmation = getattr(args, "confirm_stale_owner_sha256", None)
    if confirmation is not None:
        from pydantic import TypeAdapter
        from experiments.archive.e2_artifacts import Fingerprint
        TypeAdapter(Fingerprint).validate_python(confirmation, strict=True)
        if not args.resume:
            raise ValueError("owner_confirmation_requires_resume")
    config, prices = get_evaluation_config(), get_cost_config()
    override = getattr(args, "max_run_cost_cny", None)
    finish_final = getattr(args, "finish_final_segment", False)
    if getattr(args, "replace_zero_start", None) is not None and not finish_final:
        raise ValueError("zero_start_requires_final_segment")
    if finish_final and (args.mode != "full" or not (args.new or args.resume) or getattr(args, "continue_from", None)):
        raise ValueError("final_recovery_requires_full_new_or_resume")
    if finish_final and override is None:
        raise ValueError("final_recovery_explicit_budget_required")
    if finish_final and override > Decimal("9.589526"):
        raise ValueError("final_recovery_budget_exceeded")
    ceiling = ExperimentBudget("startup", config, mode=args.mode, max_run_cost_cny=override).snapshot().max_cost_cny
    require_e2_provider_connection()
    questions, index, conditions, options = prepare_e2(args)
    planned_runs = options.pop("planned_runs", None)
    final_recovery = options.pop("final_recovery", None)
    validate_executor_questions(questions, conditions, planned_runs)
    if "continuation:parent" in conditions:
        ceiling = override = continuation_ceiling(conditions, ceiling)
    if final_recovery:
        ceiling = override
    plan_size = len(planned_runs) if planned_runs is not None else len(questions) * 2
    logger.info("题数=%s 组数=2 计划槽位=%s 单份上限=%s 节点上限累加（最坏理论费用）=%s 整场共享上限=%s",
        len(questions), plan_size, get_harness_config().max_cost_cny,
        plan_size * get_harness_config().max_cost_cny, ceiling)
    folder = args.resume.resolve() if args.resume else PROJECT_ROOT / "traces/e2" / uuid4().hex
    if args.resume:
        saved = json.loads((folder / "state.json").read_bytes())
        experiment_id = saved["manifest"]["experiment_id"]
    else:
        experiment_id = uuid4().hex
    manifest = build_experiment_manifest(experiment_id, conditions, prices, config, mode=args.mode, max_run_cost_cny=override)
    if final_recovery:
        assert_final_segment_claim_available(final_recovery, folder, experiment_id,
            manifest["conditions_fingerprint"], ceiling, resume=bool(args.resume),
            replacement=getattr(args, "replace_zero_start", None))
    return _execute_e2_workspace(args, questions, index, conditions, options, manifest, folder,
        confirmation, planned_runs, final_recovery, prices, config, override)


def create_pilot_manifest() -> Path:
    """只生成待作者核对的dev清单，不加载模型或给出付费许可。"""
    paths = [PROJECT_ROOT / "eval/questions_30.jsonl", PROJECT_ROOT / "eval/dev_extra_60.jsonl"]
    fixed = [q for path in paths for q in load_questions(path)]
    path = PROJECT_ROOT / "results" / f"{date.today()}_manifest_{uuid4().hex}.jsonl"
    write_pilot_manifest(path, fixed, paths)
    from src.yuqing.eval.run_eval import register_result
    register_result(path, "E2-pilot待审选题清单，非模型成绩")
    return path
