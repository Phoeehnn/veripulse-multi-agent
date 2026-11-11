"""固定CED20对的单进程入口；复用传播节点、共享钱包与审计，不支持猜测续跑。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
from time import perf_counter

from src.yuqing import llm
from src.yuqing.agents.arbiter import score_case
from src.yuqing.agents.compliance import check_compliance
from src.yuqing.eval import propagation_eval as existing
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.preflight import PreparedPropagationPrompt, validate_propagation_startup
from src.yuqing.eval.propagation_v11 import (PropagationSelection, SELECTION_SHA256, build_pair_plan,
    check_source_snapshot, score_fingerprints, validate_selection, validate_sample)
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.schemas import ComplianceResult, Sample
from src.yuqing.settings import get_cost_config, get_evaluation_config, get_propagation_harness_config, get_settings

EXPERIMENT_ID = "a-v1.1-propagation-ced20-v1"
PLAN_PATH = "docs/cards/A-v1.1-1_pair-plan-v2.json"
logger = logging.getLogger(__name__)


@dataclass
class _Prepared:
    """仅装配已校验的固定输入与本次身份，不持有新钱包。"""

    root: Path
    plan: dict[str, object]
    samples: list[Sample]
    prompt: PreparedPropagationPrompt
    selection_path: Path
    plan_path: Path
    experiment_dir: Path
    selection: PropagationSelection
    plan_sha256: str


def _save(path: Path, value: object) -> None:
    """先完整刷盘临时文件再独占发布，目标存在时拒绝覆盖。"""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2).encode("utf8")
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "nt":
        partial.rename(path)
    else:
        os.link(partial, path)
        partial.unlink()


def prepare(root: Path, plan_path: Path, *, offline: bool = False) -> _Prepared:
    """先核对精确计划和来源，拒绝阶段没有目录、认领、钱包或模型调用。"""
    root = root.resolve()
    if plan_path.resolve() != root / PLAN_PATH:
        raise ValueError("propagation_pair_unregistered_plan")
    plan = json.loads(plan_path.read_bytes())
    selection_path = root / "docs/cards/A-v1.1-1_selection.json"
    selection, samples = validate_selection(root, selection_path)
    expected = build_pair_plan(root, selection_path)
    if plan != expected or len(plan["pairs"]) != 20:
        raise ValueError("propagation_pair_plan_drift")
    if Decimal(plan["slot_node_budget_cny"]) != Decimal("0.08") or Decimal(plan["shared_budget_cap_cny"]) != Decimal("3.50"):
        raise ValueError("propagation_pair_budget_drift")
    if check_compliance("").claim_status != "rules_empty":
        raise ValueError("propagation_pair_compliance_scope_changed")
    prompt = validate_propagation_startup(root / plan["prompt_path"], offline=offline)
    folder = root / "traces/A-v1.1-1-paired" / EXPERIMENT_ID
    if not folder.resolve().is_relative_to(root / "traces") or folder.exists():
        raise ValueError("propagation_pair_already_claimed_or_unsafe")
    return _Prepared(root, plan, samples, prompt, selection_path, plan_path, folder, selection,
                     sha256(plan_path.read_bytes()).hexdigest())


def _check_drift(prepared: _Prepared, sample: Sample) -> Sample:
    """每对/每槽后复核冻结条件和当前实际输入，自身产出不回流选样。"""
    current = json.loads(prepared.plan_path.read_bytes())
    if current != prepared.plan or sha256(prepared.selection_path.read_bytes()).hexdigest() != SELECTION_SHA256:
        raise ValueError("propagation_pair_conditions_drift")
    prompt = validate_propagation_startup(prepared.prompt.path, offline=True)
    conditions = {name: getattr(prompt, name) for name in ("tools_sha256", "schema_sha256", "config_sha256", "code_sha256")}
    conditions.update(score_fingerprints(prepared.root))
    if conditions != prepared.plan["condition_fingerprints"] or prompt.prompt_sha256 != prepared.plan["prompt_sha256"]:
        raise ValueError("propagation_pair_conditions_drift")
    check_source_snapshot(prepared.root, prepared.selection)
    item = next(item for item in prepared.selection.selected if item.sample_id == sample.sample_id)
    return validate_sample(prepared.root, item)


def _pair_budget_reason(wallet: ExperimentSession, reserve: Decimal) -> str | None:
    """只检查可信余额，不重复扣减下一对的预检费用。"""
    state = wallet.snapshot()
    if state.halted or state.unknown_requests or state.pending_requests:
        return state.halt_reason or "wallet_unsettled"
    if state.max_cost_cny - state.committed_cost_cny < reserve:
        return "pair_budget_insufficient"
    return None


def _trace_details(outcome: existing.PropagationEvalOutcome, trace_root: Path, sample_id: str) -> dict[str, object]:
    """读取本槽可信Trace和回执，保留原始原因与格式位置，不读取外部指令。"""
    path = Path(outcome.trace_path).resolve()
    if not path.is_relative_to(trace_root.resolve()):
        raise ValueError("propagation_pair_trace_escape")
    events = [TraceEvent.model_validate_json(line) for line in path.read_text(encoding="utf8").splitlines()]
    if any(event.run_id != outcome.run_id or event.sample_id != sample_id for event in events):
        raise ValueError("propagation_pair_trace_identity")
    reasons = []
    for event in events:
        if event.type == "verifier":
            reasons.extend("schema:" + ".".join(map(str, error.get("loc", ()))) + ":" + str(error.get("type"))
                           for error in event.data.get("errors", []))
        archive = event.data.get("archive")
        if event.type == "tool_call" and archive:
            target = (trace_root / archive["path"]).resolve()
            if not target.is_relative_to(trace_root.resolve()) or sha256(target.read_bytes()).hexdigest() != archive["sha256"]:
                raise ValueError("propagation_pair_archive_drift")
            body = json.loads(target.read_bytes())
            if body.get("run_id") != outcome.run_id or body.get("sample_id") != sample_id:
                raise ValueError("propagation_pair_archive_identity")
            reasons.extend(body["payload"].get("program_metadata", {}).get("detailed_feedback_issues") or [])
    final = [event.data for event in events if event.type == "run_outcome"]
    if len(final) != 1:
        raise ValueError("propagation_pair_terminal_count")
    request_ids = [event.data["request_id"] for event in events if event.type == "llm_request"]
    response_ids = [event.data["request_id"] for event in events if event.type == "llm_call"]
    if len(set(request_ids)) != len(request_ids) or sorted(request_ids) != sorted(response_ids):
        raise ValueError("propagation_pair_receipt_mismatch")
    return {"request_count": sum(event.type == "llm_call" for event in events),
        "steps": final[0]["steps"], "raw_reason": final[0].get("raw_reason"),
        "rejection_codes": reasons, "trace_sha256": sha256(path.read_bytes()).hexdigest(),
        "success_after_feedback": outcome.result is not None and bool(reasons), "request_ids": request_ids}


def _risk_level(outcome: existing.PropagationEvalOutcome) -> int | None:
    """只计算传播维度可用等级，事实与合规缺席，不调用仲裁模型。"""
    report = outcome.result.report if outcome.result else None
    result = score_case(fact_status=None, fact_risk=None, fact_coverage="none", fact_failed=False,
        fact_citation_ok=False, emotion_level=report.emotion_level if report else None,
        spread_level=report.spread_level if report else None, scenario="general",
        compliance=ComplianceResult(claim_status="rules_empty", comment_status="rules_empty"))
    return result.level


def _slot(prepared: _Prepared, sample: Sample, variant: str, wallet: ExperimentSession,
          pair_index: int) -> dict[str, object]:
    """调用已有真实传播节点，结果与版本、用量和Trace指纹绑定。"""
    started = perf_counter()
    outcome = existing._run_propagation_sample(sample, wallet, prompt=prepared.prompt,
        node_config=get_propagation_harness_config(), trace_root=prepared.experiment_dir / "traces",
        contract_version=variant)
    outcome = existing._validate_outcome(sample, outcome)
    _save(prepared.experiment_dir / f"outcome-{pair_index:02d}-{variant}.json", outcome.model_dump(mode="json"))
    if outcome.halt_remaining:
        outcome = outcome.model_copy(update={"result": None, "status": "failed"})
    row = existing._row(sample, outcome, prepared.prompt.prompt_sha256,
                        prepared.plan["condition_fingerprints"])
    row.update({"variant": variant, "pair_index": pair_index, "elapsed_s": perf_counter() - started,
        "experiment_id": EXPERIMENT_ID, "input_sha256": existing._input_fingerprint(sample),
        "plan_sha256": prepared.plan_sha256,
        "qualified": outcome.result is not None and outcome.cost_cny is not None,
        "risk_level": _risk_level(outcome), "halt_remaining": outcome.halt_remaining,
        "outcome": outcome.model_dump(mode="json")})
    row.update(_trace_details(outcome, prepared.experiment_dir / "traces", sample.sample_id))
    return row


def _not_run(sample_id: str, variant: str, pair: int, reason: str) -> dict[str, object]:
    """未执行不伪装为零费用，仍保留40槽分母。"""
    return {"sample_id": sample_id, "variant": variant, "pair_index": pair, "status": "not_run",
        "qualified": False, "stop_reason": reason, "cost_cny": None, "request_count": None,
        "steps": None, "elapsed_s": None, "risk_level": None, "rejection_codes": []}


def _execute(prepared: _Prepared, wallet: ExperimentSession) -> tuple[list[dict[str, object]], str]:
    """逐槽发布；遇到未知费用、审计或条件异常即停，绝不自动恢复或补跑。"""
    rows, stop = [], ""
    for pair, sample in zip(prepared.plan["pairs"], prepared.samples, strict=True):
        if not stop:
            try:
                _check_drift(prepared, sample)
                stop = _pair_budget_reason(wallet, Decimal(prepared.plan["pair_reserve_cny"])) or ""
            except (ValueError, OSError) as exc:
                stop = type(exc).__name__ + ":" + str(exc)
        for variant in pair["order"]:
            row = _not_run(sample.sample_id, variant, pair["pair_index"], stop)
            started = False
            if not stop:
                try:
                    sample = _check_drift(prepared, sample)
                    started = True
                    row = _slot(prepared, sample, variant, wallet, pair["pair_index"])
                    _save(prepared.experiment_dir / f"attempt-{len(rows) + 1:03d}.json", row)
                    _check_drift(prepared, sample)
                    _audit([*rows, row], wallet)
                    stop = ("usage_unknown" if row["cost_cny"] in (None, "") else
                            row["stop_reason"] if row["halt_remaining"] else "")
                    stop = stop or _pair_budget_reason(wallet, Decimal(0)) or ""
                except (Exception, KeyboardInterrupt) as exc:
                    code = str(exc) if isinstance(exc, ValueError) and str(exc).replace("_", "").isalpha() else type(exc).__name__
                    stop = "interrupted" if isinstance(exc, KeyboardInterrupt) else "slot_audit_or_execution_failed:" + code
                    try:
                        wallet.halt(stop)
                    except Exception:
                        wallet.close()
                    row.update({"qualified": False, "status": "failed" if started else "not_run",
                                "risk_level": None, "stop_reason": stop})
            rows.append(row)
            path = prepared.experiment_dir / f"slot-{len(rows):03d}.json"
            if not path.exists():
                _save(path, row)
            logger.info("slot=%s/40 sample=%s variant=%s status=%s cost=%s", len(rows),
                        row["sample_id"], variant, row["status"], row.get("cost_cny"))
    return rows, stop or "completed"


def _reason_category(code: str) -> str:
    """保留原始原因，另按预登记统一类别归并。"""
    if "spread_basis" in code or "spread_must" in code:
        return "统计依据资格"
    if "emotion" in code:
        return "情绪引文"
    if "narrative" in code or "lead" in code:
        return "其他来源资格"
    if "schema" in code or "format" in code:
        return "结构/协议"
    if "limit" in code or "budget" in code:
        return "预算/步数"
    return "工具/归档/运行故障"


def _mean(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    """只平均确实有记录的已执行用量，另外给出缺失数与分母。"""
    known = [row[field] for row in rows if row.get(field) not in (None, "")]
    return {"value": sum(float(value) for value in known) / len(known) if known else None,
            "denominator": len(known), "unknown": len(rows) - len(known)}


def _group_summary(rows: list[dict[str, object]], variant: str) -> dict[str, object]:
    """平均值注明已执行分母，未知费用单列，不把未执行算免费。"""
    chosen = [row for row in rows if row["variant"] == variant]
    executed = [row for row in chosen if row["status"] != "not_run"]
    known = [row for row in executed if row.get("cost_cny") not in (None, "")]
    codes = Counter(code for row in executed for code in row.get("rejection_codes", []))
    final_codes = Counter(row["stop_reason"] for row in executed if not row["qualified"])
    return {"denominator": 20, "executed": len(executed), "qualified": sum(row["qualified"] for row in chosen),
        "failed": sum(not row["qualified"] for row in executed), "not_run": len(chosen) - len(executed),
        "risk_available": sum(row.get("risk_level") is not None for row in chosen),
        "first_pass": sum(row["qualified"] and not row.get("success_after_feedback") for row in chosen),
        "after_feedback": sum(row["qualified"] and bool(row.get("success_after_feedback")) for row in chosen),
        "means": {name: _mean(executed, name) for name in ("request_count", "steps", "elapsed_s", "cost_cny")},
        "known_cost_denominator": len(known), "unknown_cost_cases": len(executed) - len(known),
        "qualified_emotion_unknown": sum(row["qualified"] and row.get("emotion_level") == "未知" for row in chosen),
        "qualified_spread_unknown": sum(row["qualified"] and row.get("spread_level") == "未知" for row in chosen),
        "original_rejection_codes": dict(codes), "final_failure_codes": dict(final_codes),
        "rejection_categories": dict(Counter({category: sum(count for code, count in codes.items()
            if _reason_category(code) == category) for category in {_reason_category(code) for code in codes}}))}


def summarize(rows: list[dict[str, object]], stop: str) -> dict[str, object]:
    """完整20对才判断工程门槛；配对变化只描述，不宣称显著。"""
    grouped = {variant: _group_summary(rows, variant) for variant in ("v1.0", "v1.1")}
    by_id = {(row["sample_id"], row["variant"]): row for row in rows}
    improved, worsened = [], []
    for sid in dict.fromkeys(row["sample_id"] for row in rows):
        old, new = by_id[(sid, "v1.0")], by_id[(sid, "v1.1")]
        if old["status"] == "not_run" or new["status"] == "not_run":
            continue
        if not old["qualified"] and new["qualified"]:
            improved.append(sid)
        if old["qualified"] and not new["qualified"]:
            worsened.append(sid)
    complete = len(rows) == 40 and all(row["status"] != "not_run" for row in rows) and stop == "completed"
    return {"groups": grouped, "improved_sample_ids": improved, "worsened_sample_ids": worsened,
        "b": len(improved), "c": len(worsened), "stop_reason": stop, "complete": complete,
        "new_rejection_categories": sorted(set(grouped["v1.1"]["rejection_categories"]) -
                                           set(grouped["v1.0"]["rejection_categories"])),
        "engineering_numeric_gate": complete and grouped["v1.1"]["qualified"] >= 17 and
        len(improved) - len(worsened) >= 3, "engineering_gate": None if complete else False,
        "content_review_required": True,
        "scope": "传播交卷开发验证；风险仅统计等级可用性。工程数字门槛不替代逐条内容/安全复核。"}


def _audit(rows: list[dict[str, object]], wallet: ExperimentSession) -> None:
    """整场交叉核对请求集合、身份、已知费用及未变的Trace字节。"""
    history = wallet.load_history()
    reservations = {event["request_id"]: event for event in history.events if event["event_type"] == "reserved"}
    ids = [rid for row in rows for rid in row.get("request_ids", [])]
    missing = set(reservations) - set(ids)
    cancelled = {event["request_id"] for event in history.events if event["event_type"] == "cancelled"}
    dispatched = {event["request_id"] for event in history.events if event["event_type"] == "dispatched"}
    if len(ids) != len(set(ids)) or set(ids) - set(reservations) or not missing <= cancelled - dispatched:
        raise ValueError("propagation_pair_wallet_request_mismatch")
    for row in rows:
        if row.get("trace_sha256") and sha256(Path(row["trace_path"]).read_bytes()).hexdigest() != row["trace_sha256"]:
            raise ValueError("propagation_pair_trace_changed")
        for request_id in row.get("request_ids", []):
            event = reservations[request_id]
            if (event["run_id"] != row["run_id"] or event["sample_id"] != row["sample_id"] or event["group_id"] != EXPERIMENT_ID):
                raise ValueError("propagation_pair_wallet_identity_mismatch")
    subtotal = sum((Decimal(str(row.get("known_cost_subtotal_cny", 0))) for row in rows), Decimal(0))
    if subtotal != wallet.snapshot().known_cost_subtotal_cny:
        raise ValueError("propagation_pair_wallet_subtotal_mismatch")


def _complete_report(rows: list[dict[str, object]], stop: str, wallet: ExperimentSession) -> dict[str, object]:
    """全场审计失败时只保留临时计数，不能发布成可信成绩。"""
    report = summarize(rows, stop)
    try:
        report["wallet"] = wallet.snapshot().model_dump(mode="json")
    except Exception as exc:
        report["wallet"] = None
        report["wallet_error"] = type(exc).__name__
    try:
        _audit(rows, wallet)
        report["audit"] = "wallet_and_slot_checks_completed"
    except Exception as exc:
        report["provisional_groups"] = report.pop("groups")
        report.update({"groups": None, "audit": "failed:" + type(exc).__name__, "complete": False,
            "stop_reason": "final_audit_failed", "engineering_numeric_gate": False, "engineering_gate": False})
    return report


def run_pair_evaluation(root: Path, plan_path: Path, *, paid_authorized: bool = False) -> Path:
    """显式许可后，在同进程网络预检通过才认领唯一场并创建持久钱包。"""
    if paid_authorized is not True:
        raise ValueError("propagation_pair_paid_authorization_required")
    prepared = prepare(root, plan_path)
    settings = get_settings()
    reachable, reason, _ = llm.check_provider_connection(settings.deepseek_base_url, settings.llm_timeout_s)
    if not reachable:
        raise ValueError("propagation_pair_network_preflight_failed:" + reason)
    if json.loads(plan_path.read_bytes()) != prepared.plan:
        raise ValueError("propagation_pair_plan_changed_before_claim")
    validate_selection(prepared.root, prepared.selection_path)
    prepared.experiment_dir.mkdir(parents=True, exist_ok=False)
    conditions = {"plan_sha256": sha256(plan_path.read_bytes()).hexdigest(),
        **prepared.plan["condition_fingerprints"]}
    _save(prepared.experiment_dir / "manifest.json", prepared.plan)
    wallet = ExperimentSession.create(prepared.experiment_dir / "wallet.sqlite", EXPERIMENT_ID,
        conditions, get_cost_config(), get_evaluation_config(), mode="full", max_run_cost_cny=Decimal("3.50"))
    try:
        rows, stop = _execute(prepared, wallet)
        report = _complete_report(rows, stop, wallet)
        target = prepared.experiment_dir / "report.json"
        _save(target, report)
        if not str(report["audit"]).startswith("failed"):
            columns = list(dict.fromkeys(key for row in rows for key in row if key != "outcome"))
            prepared_rows = [{key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), (list, dict))
                             else row.get(key) for key in columns} for row in rows]
            csv = existing._write_rows(prepared_rows, root / "results")
            _save(prepared.experiment_dir / "result.json", {"path": str(csv), "sha256": sha256(csv.read_bytes()).hexdigest()})
        return target
    finally:
        wallet.close()


def main() -> None:
    """仅支持显式新建及付费许可，不提供恢复、隐含额度或自动重跑参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", action="store_true", required=True)
    parser.add_argument("--authorize-paid", action="store_true", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("40槽节点费用上界¥3.20；两版共享硬上限¥3.50；每对启动检查¥0.16。")
    root = get_settings().project_root
    print(run_pair_evaluation(root, root / PLAN_PATH, paid_authorized=args.authorize_paid))


if __name__ == "__main__":
    main()
