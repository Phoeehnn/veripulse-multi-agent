"""在同一批题目和语料上比较三路检索及重排。"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import json
import logging
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal
from uuid import uuid4

from src.yuqing.eval.dataset import (
    RetrievalQuestion, align_question_gold, load_question_corpus as load_legacy_corpus, load_questions,
    select_acceptance_questions,
)
from src.yuqing.eval.metrics import fact_check_summary, hit_at_k, recall_at_k, reciprocal_rank_at_k
from src.yuqing.eval.preflight import validate_fact_check_startup
from src.yuqing.eval.preflight import validate_propagation_offline_prompt
from src.yuqing.eval import preflight
from src.yuqing.eval import propagation_eval
from src.yuqing.eval.propagation_eval import (manifest_rows, preflight_propagation_batch,
    run_propagation_batch, save_manifest, select_propagation_samples, evaluate_propagation)
from src.yuqing.eval import propagation_ced10
from src.yuqing import settings
from src.yuqing.retrieval.bm25 import BM25Index, build_bm25_index
from src.yuqing.retrieval import search as search_module
from src.yuqing.retrieval.corpus import load_eval_corpus as load_question_corpus, validate_eval_corpus as validate_vector_corpus
from src.yuqing.settings import CostConfig, PROJECT_ROOT, get_agent_config, get_evaluation_config, get_fact_risk_config, get_retrieval_config, get_settings
from src.yuqing.schemas import EvidenceChunk
from src.yuqing.schemas import Sample
from src.yuqing.ingest.chef import load_chef
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing import llm
from src.yuqing.harness.trace import RunContext, RunUsage, get_run_usage, record_event, use_run_context
from src.yuqing.agents.fact_check import verify
from src.yuqing.agents.base import citation_issues


logger = logging.getLogger(__name__)


def register_result(path: Path, purpose: str) -> None:
    """项目正式结果自动追加索引，已有记录不得悄悄改变指纹。"""

    if path.parent.resolve() != (PROJECT_ROOT / "results").resolve():
        return
    index = PROJECT_ROOT / "docs/results-index.md"
    digest = sha256(path.read_bytes()).hexdigest()
    previous = [line for line in index.read_text(encoding="utf-8").splitlines() if f"`{path.name}`" in line]
    if previous:
        if len(previous) != 1 or f"`{digest}`" not in previous[0]:
            raise ValueError("历史结果指纹变化，拒绝修改已有索引")
        return
    with index.open("a", encoding="utf-8", newline="") as stream:
        stream.write(f"| `{path.name}` | {purpose} | `{digest}` | 待复核 |\n")


def fact_check_metadata(system_prompt: str) -> dict[str, str]:
    """校验直接评测入口并记录原文件和两种实际系统消息的独立指纹。"""

    prepared = validate_fact_check_startup(system_prompt=system_prompt)
    return {"model": get_settings().deepseek_model, "system_prompt": prepared.text,
            "prompt_path": str(prepared.path), "prompt_sha256": prepared.prompt_sha256,
            "full_system_sha256": prepared.full_system_sha256, "repair_system_sha256": prepared.repair_system_sha256,
            "retrieval_config": json.dumps(get_retrieval_config().model_dump(), ensure_ascii=False),
            "agent_config": json.dumps(get_agent_config().model_dump(), ensure_ascii=False),
            "fact_risk_config": json.dumps({label: rule.model_dump() for label, rule in
                                            get_fact_risk_config().items()}, ensure_ascii=False),
            "cost_config": prepared.prices.model_dump_json()}


def rank_retrieval_methods(
    query: str, index: BM25Index, path: Path, k: int | None = None
) -> dict[str, list[EvidenceChunk]]:
    """在统一配置下得到四路排名，保持评测与正式检索的选择规则一致。"""

    limit = get_retrieval_config().final_top_k if k is None else k
    if limit <= 0:
        raise ValueError("检索评测 k 必须大于零")
    candidates = search_module.retrieve_candidates(query, index, path)
    return {"vector": candidates["vector"][:limit], "bm25": candidates["bm25"][:limit],
            "hybrid": search_module.rank_candidates(query, candidates["hybrid"], limit, False),
            "rerank": search_module.rank_candidates(query, candidates["hybrid"], limit, True)}


def _score_retrieval_ranking(
    found: list[EvidenceChunk], gold_ids: list[str], hit_k: int, ranking_k: int
) -> dict[str, float | str]:
    """缺少可定位参考文档时全部留空，避免把资料缺口算成检索失败。"""

    return {"hit_at_k": hit_at_k(found, gold_ids, hit_k) if gold_ids else "",
            "recall_at_k": recall_at_k(found, gold_ids, ranking_k) if gold_ids else "",
            "reciprocal_rank_at_k": reciprocal_rank_at_k(found, gold_ids, ranking_k) if gold_ids else ""}


def evaluate_retrieval(
    questions: list[RetrievalQuestion], index: BM25Index, path: Path,
    on_row: Callable[[dict[str, str | float | int]], None] | None = None,
) -> list[dict[str, str | float | int]]:
    """保留四路排名并记录命中、已定位文档召回及截断倒数排名。"""

    config = get_retrieval_config()
    if any(question.gold_alignment == "unverified" for question in questions):
        raise ValueError("评测前必须先核验 gold 原文与候选文档的对应关系")
    limit = config.final_top_k
    ranking_k = get_evaluation_config().retrieval_metrics_k
    rows: list[dict[str, str | float | int]] = []
    for question in questions:
        rankings = rank_retrieval_methods(question.claim, index, path, max(limit, ranking_k))
        row: dict[str, str | float | int] = {
            "sample_id": question.sample_id,
            "gold_veracity": question.gold_veracity,
            "gold_evidence_ids": json.dumps(question.gold_evidence_ids, ensure_ascii=False),
            "gold_alignment": question.gold_alignment,
            "gold_total": question.gold_total,
            "gold_matched": question.gold_matched,
            "k": limit,
            "ranking_k": ranking_k,
            "metric_scope": "located_gold_documents_in_returned_chunks",
            "retrieval_config": json.dumps(config.model_dump(), ensure_ascii=False),
        }
        for name, found in rankings.items():
            row.update({f"{name}_{metric}": value for metric, value in
                        _score_retrieval_ranking(found, question.gold_evidence_ids, limit, ranking_k).items()})
            row[f"{name}_chunk_ids"] = json.dumps(
                [item.chunk_id for item in found], ensure_ascii=False
            )
        rows.append(row)
        if on_row is not None:
            on_row(row)
        logger.info("完成检索评测：%s", question.sample_id)
    return rows


def save_retrieval_results(
    rows: list[dict[str, str | float | int]], results_dir: Path | None = None,
    kind: Literal["retrieval", "factcheck", "manifest", "audit"] = "retrieval",
) -> Path:
    """将逐题结果写入新的日期 CSV，绝不覆盖已有评测。"""

    if not rows:
        raise ValueError("没有可保存的检索评测行")
    directory = results_dir or get_settings().results_dir
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{datetime.now().date().isoformat()}_{kind}_{uuid4().hex}.csv"
    target = directory / filename
    with target.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    register_result(target, {"retrieval": "检索评测输出", "factcheck": "事实核查输出",
                             "manifest": "选题清单", "audit": "重计分/审计输出"}[kind])
    return target


def run_retrieval_eval(
    chef_path: Path, questions_path: Path, store_path: Path,
    on_row: Callable[[dict[str, str | float | int]], None] | None = None,
) -> Path:
    """核对两路语料完全相同后运行评测并保存逐题 CSV。"""

    questions = load_questions(questions_path)
    corpus = load_question_corpus(chef_path, questions)
    validate_vector_corpus(corpus, store_path)
    questions = align_question_gold(chef_path, questions, get_evaluation_config().gold_review_path)
    index = build_bm25_index(corpus)
    rows = evaluate_retrieval(questions, index, store_path, on_row=on_row)
    return save_retrieval_results(rows)


def rescore_retrieval_results(
    csv_path: Path, chef_path: Path, questions_path: Path, review_path: Path | None = None
) -> Path:
    """保留历史排序，用核验后的 gold 重新计分并另存新 CSV。"""

    questions = align_question_gold(chef_path, load_questions(questions_path), review_path)
    lookup = {question.sample_id: question for question in questions}
    corpus = {piece.chunk_id: piece for piece in load_legacy_corpus(chef_path, questions)}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(questions) or {row["sample_id"] for row in rows} != set(lookup):
        raise ValueError("历史 CSV 与冻结题目的样本集合不一致")
    corrected: list[dict[str, str | float | int]] = []
    for previous in rows:
        question = lookup[previous["sample_id"]]
        row: dict[str, str | float | int] = dict(previous)
        row.update(
            original_gold_evidence_ids=previous["gold_evidence_ids"],
            gold_evidence_ids=json.dumps(question.gold_evidence_ids, ensure_ascii=False),
            gold_alignment=question.gold_alignment,
            gold_total=question.gold_total, gold_matched=question.gold_matched,
            ranking_source=csv_path.name,
        )
        for method in ("vector", "bm25", "hybrid", "rerank"):
            found = [corpus[key] for key in json.loads(previous[f"{method}_chunk_ids"])]
            row[f"{method}_hit_at_k"] = (
                hit_at_k(found, question.gold_evidence_ids, int(previous["k"]))
                if question.gold_evidence_ids else ""
            )
        corrected.append(row)
    return save_retrieval_results(corrected, kind="audit")


def _verify_in_context(question: RetrievalQuestion, index: BM25Index, path: Path,
                       system_prompt: str, context: RunContext) -> tuple:
    """把判断与结算绑定到当前题，异常也记录出口并恢复上下文。"""
    with use_run_context(context):
        try:
            result = verify(question.claim, index, system_prompt, store_path=path)
        except Exception:
            record_event("stop", {"reason": "exception"})
            raise
        record_event("stop", {"reason": "completed" if result is not None else "error"})
        return result, get_run_usage(context.run_id)


def _usage_columns(usage: RunUsage) -> dict[str, str | float | int]:
    """导出完整用量及可信小计，未知总量不抹去已确认消费。"""
    return {"successful_llm_responses": usage.successful_responses,
            "input_tokens": usage.input_tokens if usage.usage_known else "",
            "output_tokens": usage.output_tokens if usage.usage_known else "",
            "cost_cny": usage.cost_cny if usage.cost_cny is not None else "",
            "known_cost_subtotal_cny": usage.known_cost_subtotal_cny,
            "known_input_tokens_subtotal": usage.input_tokens,
            "known_output_tokens_subtotal": usage.output_tokens,
            "usage_issues": json.dumps(usage.issues, ensure_ascii=False),
            "usage_known": int(usage.usage_known), "llm_attempts": usage.calls}


def evaluate_fact_check(
    questions: list[RetrievalQuestion], index: BM25Index, path: Path, system_prompt: str,
    on_row: Callable[[dict[str, str | float | int]], None] | None = None,
) -> list[dict[str, str | float | int]]:
    """真实运行固定验收题并记录标签、证据、用量及完整提示词。"""

    labels = {"真": "支持", "假": "反驳", "无法判定": "信息不足"}
    metadata = fact_check_metadata(system_prompt)
    rows: list[dict[str, str | float | int]] = []
    for question in questions:
        context = RunContext(run_id=uuid4().hex, sample_id=question.sample_id, agent="fact")
        started = perf_counter()
        result, usage = _verify_in_context(question, index, path, system_prompt, context)
        output, opinion, evidence = result if result is not None else (None, None, [])
        issues = citation_issues(output, len(evidence)) if output is not None else ["Agent 返回 None"]
        row: dict[str, str | float | int] = {
            "sample_id": question.sample_id, "claim": question.claim,
            "gold_label": labels[question.gold_veracity],
            "predicted_label": output.label if output is not None else "",
            "correct": int(output is not None and output.label == labels[question.gold_veracity]),
            "success": int(output is not None),
            "citation_format_valid": int(output is not None and not issues),
            "citation_issues": json.dumps(issues, ensure_ascii=False),
            "score": opinion.risk if opinion is not None else "",
            "fact_risk": opinion.risk if opinion is not None else "",
            "citation_ok": int(opinion is not None and opinion.citation_ok),
            "flags": json.dumps(opinion.flags if opinion is not None else [], ensure_ascii=False),
            "opinion_json": opinion.model_dump_json() if opinion is not None else "",
            "second_call": int(usage.calls > 1),
            "confidence": output.confidence if output is not None else "",
            "output_json": output.model_dump_json() if output is not None else "",
            "evidence_json": json.dumps([item.model_dump() for item in evidence], ensure_ascii=False),
            "evidence_id_map": json.dumps({f"E{n}": item.chunk_id for n, item in enumerate(evidence, start=1)}),
            "gold_evidence_ids": json.dumps(question.gold_evidence_ids, ensure_ascii=False),
            "gold_alignment": question.gold_alignment,
            "latency_ms": round((perf_counter() - started) * 1000),
            **_usage_columns(usage),
            "run_id": context.run_id, "trace_path": str(context.trace_path.relative_to(PROJECT_ROOT)),
            "stop_reason": "completed" if result is not None else "error",
            **metadata,
        }
        rows.append(row)
        if on_row is not None:
            on_row(row)
        logger.info("事实核查 %s：%s -> %s", question.sample_id, row["gold_label"], row["predicted_label"])
    return rows


def run_fact_check_acceptance(chef_path: Path, questions_path: Path, store_path: Path) -> Path:
    """先保存固定选题清单，再运行真实端到端事实核查并另存结果。"""

    config = get_evaluation_config()
    prepared = validate_fact_check_startup(config.fact_check_prompt_path)
    questions = align_question_gold(chef_path, load_questions(questions_path), config.gold_review_path)
    selected = select_acceptance_questions(questions)
    corpus = load_question_corpus(chef_path, questions)
    validate_vector_corpus(corpus, store_path)
    prompt = prepared.text
    manifest = save_retrieval_results([
        {"sample_id": question.sample_id, "gold_veracity": question.gold_veracity,
         "phase": "selected_before_inference", "prompt_sha256": sha256(prompt.encode()).hexdigest(),
         "gold_review_path": str(config.gold_review_path)} for question in selected
    ], kind="manifest")
    logger.info("固定验收清单：%s", manifest)
    rows = evaluate_fact_check(selected, build_bm25_index(corpus), store_path, prompt)
    for row in rows:
        row["selection_manifest"] = manifest.name
        row["gold_review_path"] = str(config.gold_review_path)
    return save_retrieval_results(rows, kind="factcheck")


def run_fact_check_full_eval(
    chef_path: Path, questions_path: Path, store_path: Path, prompt_path: Path | None = None,
    *, limit: int | None = None,
    on_row: Callable[[dict[str, str | float | int]], None] | None = None,
) -> Path:
    """固定全套题目后逐行落盘，运行现有检索与事实核查入口。"""

    config = get_evaluation_config()
    source = prompt_path or config.fact_check_prompt_path
    prepared = validate_fact_check_startup(source)
    questions = align_question_gold(chef_path, load_questions(questions_path), config.gold_review_path)
    selected = _select_question_prefix(questions, limit)
    corpus = load_question_corpus(chef_path, questions)
    validate_vector_corpus(corpus, store_path)
    prompt = prepared.text
    manifest = save_retrieval_results([
        {"sample_id": q.sample_id, "gold_veracity": q.gold_veracity,
         "gold_alignment": q.gold_alignment, "phase": "full_frozen_questions_selected_before_inference" if limit is None else "prefix_frozen_questions",
         "prompt_sha256": sha256(prompt.encode()).hexdigest(),
         "gold_review_path": str(config.gold_review_path)} for q in selected
    ], kind="manifest")
    logger.info("固定完整评测清单：%s", manifest)
    target = get_settings().results_dir / f"{datetime.now().date().isoformat()}_factcheck_{uuid4().hex}.csv"
    index = build_bm25_index(corpus)
    with target.open("x", newline="", encoding="utf-8-sig") as stream:
        writer: csv.DictWriter | None = None

        def persist(row: dict[str, str | float | int]) -> None:
            """把一题结果立即追加并刷新到磁盘。"""

            nonlocal writer
            row.update(selection_manifest=manifest.name, gold_review_path=str(config.gold_review_path),
                       prompt_path=str(source), evaluation_phase="full_frozen_questions" if limit is None else "prefix_frozen_questions")
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            if on_row is not None:
                on_row(row)

        try:
            rows = evaluate_fact_check(selected, index, store_path, prompt, on_row=persist)
        finally:
            stream.flush()
            register_result(target, "完整事实核查输出（可能中断，按行数复核）")
    if len(rows) != len(selected):
        raise RuntimeError("完整评测保存的行数与冻结题目不一致")
    return target


def _select_question_prefix(
    questions: list[RetrievalQuestion], limit: int | None
) -> list[RetrievalQuestion]:
    """只选冻结前缀，不重排；非法数量在模型调用及落盘前拒绝。"""

    if limit is not None and (limit <= 0 or limit > len(questions)):
        raise ValueError(f"limit 必须在 1 到 {len(questions)} 之间")
    return questions if limit is None else questions[:limit]


def _progress_callback() -> Callable[[dict[str, str | float | int]], None]:
    """为本次 CLI 创建独立计数器，每完成一题输出一行。"""

    completed = 0

    def report(row: dict[str, str | float | int]) -> None:
        """显示题号与当前结果，不展示模型全文。"""

        nonlocal completed
        completed += 1
        logging.getLogger(__name__ + ".cli").info("[%s] %s %s", completed, row["sample_id"],
            row.get("predicted_label", "检索完成") or "失败")

    return report


def _report_result(path: Path, suite: str) -> None:
    """只读已落盘结果，打印事实核查或检索的对应指标。"""

    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    console = logging.getLogger(__name__ + ".cli")
    if suite == "factcheck":
        summary = fact_check_summary(rows)
        console.info("正确数：%s/%s；Macro-F1：%.4f", summary["correct"], summary["total"], summary["macro_f1"])
        if summary["unknown_cost_rows"]:
            console.info("已知费用小计：¥%.6f；%s题费用未知，总费用未知",
                         summary["known_cost_subtotal_cny"], summary["unknown_cost_rows"])
        else:
            console.info("总估计费用：¥%.6f", summary["cost_cny"])
    else:
        console.info("正确数 / Macro-F1：不适用；模型 API 费用：¥0（本地检索不调用 LLM）")
        for method in ("vector", "bm25", "hybrid", "rerank"):
            scored = [float(row[f"{method}_hit_at_k"]) for row in rows if row[f"{method}_hit_at_k"] != ""]
            console.info("%s hit@k：%s；可计分：%s/%s", method,
                         sum(scored) / len(scored) if scored else "不适用", len(scored), len(rows))
            for metric, title in (("recall_at_k", "recall"), ("reciprocal_rank_at_k", "MRR")):
                values = [float(row[f"{method}_{metric}"]) for row in rows
                          if row.get(f"{method}_{metric}", "") != ""]
                console.info("%s %s@%s：%s；可计分：%s/%s", method, title, rows[0].get("ranking_k", "未登记"),
                             sum(values) / len(values) if values else "不适用", len(values), len(rows))
        console.info("指标基于已定位gold；部分对应题：%s；未对应题：%s",
                     sum(row.get("gold_alignment") == "partial" for row in rows),
                     sum(row.get("gold_alignment") == "unmatched" for row in rows))
    console.info("结果文件：%s", path)


def main(argv: list[str] | None = None) -> int:
    """解析入口参数，转交对应评测流程并统一报告可恢复错误。"""

    parser = _build_eval_parser()
    args = parser.parse_args(argv)
    console = logging.getLogger(__name__ + ".cli")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    console.addHandler(handler)
    console.setLevel(logging.INFO)
    console.propagate = False
    try:
        return _dispatch_eval(args, console)
    except (ValueError, OSError, RuntimeError) as exc:
        console.error("评测未完成：%s", exc)
        return 1
    finally:
        console.removeHandler(handler)
        handler.close()


def _build_eval_parser() -> argparse.ArgumentParser:
    """定义传统评测和传播两种入口，不为旧命令改参数语义。"""

    parser = argparse.ArgumentParser(description="项目 A 的 dev 评测（不运行 test）")
    commands = parser.add_subparsers(dest="suite", required=True)
    factcheck = commands.add_parser("factcheck")
    factcheck.add_argument("--prompt", type=Path)
    factcheck.add_argument("--limit", type=int)
    commands.add_parser("retrieval")
    propagation = commands.add_parser("propagation")
    propagation.add_argument("--source", type=Path, help="可选JSONL覆盖；默认读取settings配置的CHEF dev与CED")
    propagation.add_argument("--prompt", type=Path, default=Path("prompts/propagation/v1.md"))
    propagation.add_argument("--manifest", type=Path, required=True)
    propagation.add_argument("--max-cost-cny", type=float, required=True)
    propagation.add_argument("--limit", type=int, default=10)
    propagation.add_argument("--execute", action="store_true", help="显式进入真实传播调用路径")
    propagation.add_argument("--i-authorize-real-calls", action="store_true")
    propagation.add_argument("--ledger", type=Path)
    propagation.add_argument("--experiment-id")
    ced10 = commands.add_parser("propagation-ced10")
    ced10.add_argument("--prompt", type=Path, default=Path("prompts/propagation/v1.md"))
    ced10.add_argument("--manifest", type=Path, required=True)
    ced10.add_argument("--max-cost-cny", type=float, required=True)
    ced10.add_argument("--execute", action="store_true")
    ced10.add_argument("--i-authorize-real-calls", action="store_true")
    ced10.add_argument("--ledger", type=Path)
    ced10.add_argument("--experiment-id")
    case = commands.add_parser("case")
    case.add_argument("--sample-id", required=True)
    case.add_argument("--eval-mode", action="store_true")
    case.add_argument("--dry-run", action="store_true")
    review = case.add_mutually_exclusive_group()
    review.add_argument("--approve", action="store_true")
    review.add_argument("--reject", action="store_true")
    review.add_argument("--modify", metavar="建议动作")
    case.add_argument("--reason")
    case.add_argument("--reviewer")
    return parser


def _dispatch_eval(args: argparse.Namespace, console: logging.Logger) -> int:
    """分派子命令；传播评测的所有启动检查仍在其专用入口执行。"""

    if args.suite == "case":
        _run_case_command(args, console)
        return 0
    if args.suite == "propagation":
        _run_propagation_manifest(args, console) if args.execute else _prepare_propagation_manifest(args, console)
        return 0
    if args.suite == "propagation-ced10":
        _run_ced10_manifest(args, console) if args.execute else _prepare_ced10_manifest(args, console)
        return 0
    settings_value = get_settings()
    progress = _progress_callback()
    if args.suite == "factcheck":
        result = run_fact_check_full_eval(settings_value.chef_dev_path, settings_value.dev_questions_path,
            settings_value.dev_store_path, args.prompt, limit=args.limit, on_row=progress)
    else:
        result = run_retrieval_eval(settings_value.chef_dev_path, settings_value.dev_questions_path,
                                    settings_value.dev_store_path, on_row=progress)
    _report_result(result, args.suite)
    return 0


def _run_case_command(args: argparse.Namespace, console: logging.Logger) -> None:
    """Run a model-free case and optionally resume its in-process approval pause."""
    if not args.sample_id.strip():
        raise ValueError("case_sample_id_required")
    if not args.dry_run:
        raise RuntimeError("case_real_run_requires_authorized_runtime_and_confirmed_budget")
    if args.eval_mode and (args.approve or args.reject or args.modify):
        raise ValueError("eval_mode_does_not_accept_human_decision")
    from src.yuqing.graph.runtime import build_case_dry_run_assembly
    from src.yuqing.graph.build import resume_case_graph

    assembly = build_case_dry_run_assembly(args.sample_id, eval_mode=args.eval_mode)
    from src.yuqing.eval.fact_v2_experiment import collect_fact_v2_conditions
    from src.yuqing.settings import get_fact_v2_evaluation_config

    conditions = collect_fact_v2_conditions(dataset="dev90")
    pairs = zip(conditions.questions, conditions.manifests, strict=True)
    selected = next(((question, manifest) for question, manifest in pairs
                     if question.sample_id == args.sample_id), None)
    if selected is None:
        raise ValueError("case_dry_run_sample_not_in_frozen_dev90")
    question, manifest = selected
    request = assembly.request_factory.fact_initial(args.sample_id, question.claim,
        tuple(manifest.entries), conditions.prompt_texts["single"],
        conditions.prompt_texts["loop"], get_fact_v2_evaluation_config())
    _complete_case_dry_run(args, console, assembly, request)


def _complete_case_dry_run(args: argparse.Namespace, console: logging.Logger,
                            assembly: Any, request: Any) -> None:
    """Run, pause/resume if requested, and print one fully assembled Mock case."""
    from src.yuqing.graph.build import resume_case_graph

    graph, context, result = assembly.graph, assembly.context, assembly.invoke()
    config = {"configurable": {"thread_id": context.thread_id}}
    snapshot = graph.get_state(config)
    interrupts = [item for task in snapshot.tasks for item in task.interrupts]
    if interrupts:
        console.info("approval_pending thread_id=%s case_id=%s request=%s", context.thread_id,
                     result.get("case_id"), interrupts[0].value)
        resume = _case_resume_payload(args, interrupts[0].value, context.thread_id, console)
        result = resume_case_graph(graph, context.thread_id, resume)
    diagram = graph.get_graph().draw_mermaid()
    console.info("dry-run: 完整运行Mock图；没有调用模型、钱包或外部工具")
    console.info("sample_id=%s thread_id=%s", args.sample_id, context.thread_id)
    console.info("request_inputs single=%s loop=%s (same production builder; estimates only)",
        request.single_input_estimate, request.loop_input_estimate)
    console.info("nodes=%s", ",".join(graph.get_graph().nodes))
    console.info("ledger_before=%s", {"issues": [], "actions": []})
    console.info("ledger_after=%s", {"issues": result.get("issues", []),
        "actions": result.get("actions", [])})
    console.info("mock_summary: fact=%s propagation=%s issues=%s decision=%s risk=%s",
        result.get("fact_rounds"), result.get("prop_rounds"), result.get("issues"),
        result.get("decisions"), result.get("risk"))
    console.info("actions=%s validation=%s usage=%s trace=none mock_cost_cny=0",
        result.get("actions", []), result.get("action_validation"), result.get("usage", []))
    console.info("approval=%s alert=%s", result.get("approval"), result.get("alert"))
    console.info("brief:\n%s", result.get("brief"))
    console.info("graph:\n%s", diagram)


def _case_resume_payload(args: argparse.Namespace, interrupt_data: dict[str, object],
                         thread_id: str, console: logging.Logger) -> dict[str, object]:
    """Collect one review choice after the graph has visibly paused."""
    if args.approve:
        decision, disposition = "approve", None
    elif args.reject:
        decision, disposition = "reject", None
    elif args.modify is not None:
        decision, disposition = "modify", args.modify
    else:
        decision = input("复核决定 approve / reject / modify：").strip()
        disposition = input("修改后的建议动作：").strip() if decision == "modify" else None
    reason = args.reason if args.reason is not None else input("处理理由：").strip()
    reviewer = args.reviewer if args.reviewer is not None else input("复核人标识：").strip()
    console.info("resume thread_id=%s decision=%s", thread_id, decision)
    return {"case_id": interrupt_data.get("case_id"),
        "decision_id": interrupt_data.get("decision_id"), "decision": decision,
        "reason": reason, "reviewer": reviewer, "final_disposition": disposition}


def _prepare_propagation_manifest(args: argparse.Namespace, console: logging.Logger) -> None:
    """核验真实来源与冻结排除名单后，只写离线清单，不执行模型。"""

    if args.limit != 10 or args.max_cost_cny <= 0:
        raise ValueError("propagation_limit_must_be_10_and_cost_cap_positive")
    _, prompt_sha = validate_propagation_offline_prompt(args.prompt)
    candidates = _load_propagation_candidates(args.source)
    if any(item.source not in {"chef", "ced"} for item in candidates):
        raise ValueError("propagation_manifest_wrong_source")
    pool_path = PROJECT_ROOT / "eval/label_pool_100.jsonl"
    test_path = PROJECT_ROOT / "eval/test_60.jsonl"
    pool_ids = {json.loads(line)["sample_id"] for line in pool_path.read_text(encoding="utf-8").splitlines() if line}
    test_ids = {json.loads(line)["sample_id"] for line in test_path.read_text(encoding="utf-8").splitlines() if line}
    selected = select_propagation_samples(
        [item for item in candidates if item.source == "chef"],
        [item for item in candidates if item.source == "ced"],
        pool_ids=pool_ids, chef_test_ids=test_ids,
    )
    fingerprints: dict[str, str] | None = None
    registry_path = PROJECT_ROOT / "prompts/propagation/contracts.json"
    contracts = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else []
    registered = any(item.get("path") == args.prompt.resolve().relative_to(
        PROJECT_ROOT.resolve()).as_posix() and item.get("approved_for_dev") is True
        for item in contracts if isinstance(item, dict))
    if registered:
        prepared = preflight.validate_propagation_startup(args.prompt, offline=True)
        fingerprints = {name: getattr(prepared, name) for name in (
            "tools_sha256", "schema_sha256", "config_sha256", "code_sha256")}
    else:
        console.info("提示词未获开发许可，清单仅作离线草稿，不可执行。")
    written = save_manifest(manifest_rows(selected, prompt_sha, args.max_cost_cny, fingerprints), args.manifest)
    console.info("离线传播manifest已固定；此命令没有执行模型调用：%s", written)


def _run_propagation_manifest(args: argparse.Namespace, console: logging.Logger) -> None:
    """只在命令行双重明确许可并通过冻结清单后启动一次真实试跑。"""

    if not args.i_authorize_real_calls or args.ledger is None or not args.experiment_id:
        raise ValueError("真实传播调用需要--i-authorize-real-calls、--ledger和--experiment-id")
    if args.limit != 10 or args.max_cost_cny <= 0:
        raise ValueError("真实传播调用必须是正费用上限的5+5完整10条")
    from src.yuqing.eval.experiment_session import ExperimentSession

    prompt = preflight.validate_propagation_startup(args.prompt)
    candidates = _load_propagation_candidates(args.source)
    selected = _select_propagation_sources(candidates)
    config = settings.get_propagation_harness_config()
    evaluation = settings.get_evaluation_config()
    preflight_propagation_batch(selected, prompt=prompt, node_config=config,
        manifest_path=args.manifest, max_cost_cny=args.max_cost_cny)
    conditions = {"prompt_sha256": prompt.prompt_sha256, "tools_sha256": prompt.tools_sha256,
        "schema_sha256": prompt.schema_sha256, "config_sha256": prompt.config_sha256,
        "code_sha256": prompt.code_sha256, "manifest": str(args.manifest.resolve()),
        "sample_ids": ",".join(item.sample_id for item in selected)}
    session = ExperimentSession.create(args.ledger, args.experiment_id, conditions,
        settings.get_cost_config(), evaluation, mode="pilot",
        max_run_cost_cny=Decimal(str(args.max_cost_cny)))
    try:
        result = run_propagation_batch(selected, prompt=prompt, node_config=config, session=session,
            results_dir=PROJECT_ROOT / "results", max_cost_cny=args.max_cost_cny,
            manifest_path=args.manifest)
        console.info("传播试跑完成或安全停账；结果：%s", result)
    finally:
        session.close()


def _prepare_ced10_manifest(args: argparse.Namespace, console: logging.Logger) -> None:
    """离线固定CED-10运行条件，不创建实验、不接触模型。"""

    _validate_ced10_output_path(args.manifest, "manifest")
    config = settings.get_propagation_harness_config()
    _validate_ced10_cost_cap(args.max_cost_cny, config)
    selection, _ = propagation_ced10.load_ced10_selection(PROJECT_ROOT)
    _, prompt_sha = validate_propagation_offline_prompt(args.prompt)
    fingerprints = preflight.propagation_contract_fingerprints()
    fingerprints["ced10_code_sha256"] = propagation_ced10.ced10_code_fingerprint(PROJECT_ROOT)
    rows = propagation_ced10.ced10_run_rows(selection, prompt_sha, args.max_cost_cny,
        fingerprints, sha256((PROJECT_ROOT / propagation_ced10.SELECTION_PATH).read_bytes()).hexdigest())
    written = propagation_ced10.save_run_manifest(rows, args.manifest)
    console.info("CED-10运行清单已离线固定；没有创建钱包或调用模型：%s", written)


def _run_ced10_manifest(args: argparse.Namespace, console: logging.Logger) -> None:
    """通过全部启动检查后才创建钱包并执行固定CED-10清单。"""

    if not args.i_authorize_real_calls or args.ledger is None or not args.experiment_id:
        raise ValueError("CED-10真实调用需要--i-authorize-real-calls、--ledger和--experiment-id")
    _validate_ced10_output_path(args.manifest, "manifest")
    _validate_ced10_output_path(args.ledger, "ledger")
    if args.manifest.resolve() == args.ledger.resolve():
        raise ValueError("ced10_manifest_ledger_path_collision")
    config = settings.get_propagation_harness_config()
    _validate_ced10_cost_cap(args.max_cost_cny, config)
    prompt = preflight.validate_propagation_startup(args.prompt)
    selection, samples = propagation_ced10.load_ced10_selection(PROJECT_ROOT)
    fingerprints = {name: getattr(prompt, name) for name in (
        "tools_sha256", "schema_sha256", "config_sha256", "code_sha256")}
    fingerprints["ced10_code_sha256"] = propagation_ced10.ced10_code_fingerprint(PROJECT_ROOT)
    selection_sha = sha256((PROJECT_ROOT / propagation_ced10.SELECTION_PATH).read_bytes()).hexdigest()
    expected = propagation_ced10.ced10_run_rows(selection, prompt.prompt_sha256,
        args.max_cost_cny, fingerprints, selection_sha)
    propagation_ced10.validate_run_rows(propagation_ced10.read_run_manifest(args.manifest), expected)
    propagation_eval._validate_batch_capacity(samples, prompt, config)
    _execute_ced10_samples(args, console, samples, selection, prompt, config, fingerprints, selection_sha)


def _validate_ced10_cost_cap(max_cost_cny: float, config: settings.PropagationHarnessConfig) -> None:
    """限CED-10整场费用不超过十个现行传播节点额度。"""

    amount = Decimal(str(max_cost_cny))
    maximum = Decimal("10") * Decimal(str(config.max_cost_cny))
    if not amount.is_finite() or amount <= 0 or amount > maximum:
        raise ValueError("ced10_cost_cap_invalid_or_exceeds_ten_node_limits")


def _validate_ced10_output_path(path: Path | None, name: str) -> None:
    """仅允许将CED-10运行清单和钱包放在D盘项目traces目录。"""

    if path is None or not path.resolve().is_relative_to((PROJECT_ROOT / "traces").resolve()):
        raise ValueError(f"ced10_{name}_must_be_inside_project_traces")


def _execute_ced10_samples(args: argparse.Namespace, console: logging.Logger, samples: list[Sample],
        selection: propagation_ced10.CED10Selection, prompt: preflight.PreparedPropagationPrompt,
        config: settings.PropagationHarnessConfig, fingerprints: dict[str, str], selection_sha: str) -> None:
    """启动共享钱包并复用传播执行器保存CED-10全部结果行。"""

    from src.yuqing.eval.experiment_session import ExperimentSession

    evaluation = settings.get_evaluation_config()
    grouped = {item.sample_id: item.group for item in selection.selected}
    conditions = {**fingerprints, "prompt_sha256": prompt.prompt_sha256,
        "selection_sha256": selection_sha, "manifest": str(args.manifest.resolve()),
        "sample_ids": ",".join(item.sample_id for item in samples),
        "regression_ids": ",".join(selection.regression_sample_ids),
        "new_validation_ids": ",".join(selection.new_validation_sample_ids)}
    session = ExperimentSession.create(args.ledger, args.experiment_id, conditions,
        settings.get_cost_config(), evaluation, mode="pilot", max_run_cost_cny=Decimal(str(args.max_cost_cny)))
    try:
        result = evaluate_propagation(samples,
            lambda sample, wallet: propagation_eval._run_propagation_sample(
                sample, wallet, prompt=prompt, node_config=config),
            wallet=session, prompt_sha256=prompt.prompt_sha256, results_dir=PROJECT_ROOT / "results",
            max_cost_cny=args.max_cost_cny, fingerprints=fingerprints, sample_groups=grouped)
        console.info("CED-10运行完成或安全停账；结果：%s", result)
    finally:
        session.close()


def _select_propagation_sources(candidates: list[Sample]) -> list[Sample]:
    """从显式候选文件选择5+5并排除冻结标注池与CHEF测试题。"""

    if any(item.source not in {"chef", "ced"} for item in candidates):
        raise ValueError("propagation_manifest_wrong_source")
    pool_path, test_path = PROJECT_ROOT / "eval/label_pool_100.jsonl", PROJECT_ROOT / "eval/test_60.jsonl"
    pool_ids = {json.loads(line)["sample_id"] for line in pool_path.read_text(encoding="utf-8").splitlines() if line}
    test_ids = {json.loads(line)["sample_id"] for line in test_path.read_text(encoding="utf-8").splitlines() if line}
    return select_propagation_samples([item for item in candidates if item.source == "chef"],
        [item for item in candidates if item.source == "ced"], pool_ids=pool_ids, chef_test_ids=test_ids)


def _load_propagation_candidates(source: Path | None) -> list[Sample]:
    """优先读取项目配置CHEF dev/CED原始数据，允许测试注入显式JSONL。"""

    if source is not None:
        return [Sample.model_validate_json(line) for line in
                source.read_text(encoding="utf-8").splitlines() if line]
    paths = settings.get_settings()
    chef = load_chef(paths.chef_dev_path)
    pool_path = PROJECT_ROOT / "eval/label_pool_100.jsonl"
    pool_ids = {json.loads(line)["sample_id"] for line in
                pool_path.read_text(encoding="utf-8").splitlines() if line}
    ced_files = [(f"ced:{category}:{path.stem}", path.name, rumor)
        for category, folder, rumor in (("non_rumor", "non-rumor-repost", False),
                                        ("rumor", "rumor-repost", True))
        for path in (paths.ced_dataset_path / folder).glob("*.json")]
    ced: list[Sample] = []
    for sample_id, filename, rumor in sorted(ced_files):
        if sample_id in pool_ids:
            continue
        ced.append(load_ced_sample(paths.ced_dataset_path, filename, rumor=rumor))
        if len(ced) == 5:
            break
    return chef + ced


if __name__ == "__main__":
    raise SystemExit(main())
