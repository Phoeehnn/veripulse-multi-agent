"""E2描述性统计与只追加发布；失败、未执行和未知费用分开。"""
from collections import Counter
from datetime import date
from hashlib import sha256
import csv
import io
import json
import math
import os
from pathlib import Path
from statistics import median
from uuid import uuid4

from src.yuqing.agents.base import citation_issues
from experiments.archive.e2_artifacts import E2CheckpointRow
from experiments.archive.e2_dispatch import DispatchResult
from experiments.archive.e2_journal import AnswerJournal
from experiments.archive.e2_workspace import E2Workspace, _publish
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.metrics import fact_check_summary
from src.yuqing.eval.run_eval import register_result
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.schemas import FactVerdict
from src.yuqing.settings import PROJECT_ROOT, get_evaluation_config


def _json(data: object) -> str:
    """明确JSON编码，不把未知费用变成字符串零。"""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _percentile(values: list[float], q: float) -> float | None:
    """P50/P95使用最近秩定义，单条记录也可复核。"""
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def _read_events(item: E2CheckpointRow, workspace: E2Workspace) -> list[TraceEvent]:
    """逐个核对Trace及其请求、响应、阅读和提交归档，不只检查日志文件。"""
    ref = item.payload["trace"]
    raw = (PROJECT_ROOT / ref["path"]).read_bytes()
    if sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("report_trace_changed")
    events = [TraceEvent.model_validate_json(line) for line in raw.split(b"\n") if line]
    pending = [event.data for event in events]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path = (workspace.folder / value["path"]).resolve()
                if Path(value["path"]).is_absolute() or not path.is_relative_to(workspace.folder.resolve()):
                    raise ValueError("report_archive_path_escape")
                archive = path.read_bytes()
                if sha256(archive).hexdigest() != value["sha256"]:
                    raise ValueError("report_archive_changed")
                saved = json.loads(archive)
                if (saved["run_id"], saved["sample_id"]) != (item.run.run_id, item.run.sample_id):
                    raise ValueError("report_archive_identity_mismatch")
            else:
                pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return events


def _loop_step_summary(events: list[TraceEvent], result: dict | None) -> dict[str, object]:
    """优先读取终局预算计数；历史失败缺少可信计数时留空，不猜测零步。"""
    terminal = [event.data for event in events if event.type == "run_outcome"]
    if len(terminal) != 1:
        raise ValueError("report_terminal_count")
    data = terminal[0]
    if "step_count_version" in data:
        ordinary, final, steps = data.get("ordinary_steps"), data.get("final_used"), data.get("steps")
        if (data["step_count_version"] != "budget-reservations-v1" or type(ordinary) is not int or
                ordinary < 0 or type(final) is not bool or type(steps) is not int or
                steps != ordinary + int(final)):
            raise ValueError("report_step_count_invalid")
        if result is not None and result["steps"] != steps:
            raise ValueError("report_step_count_mismatch")
        return dict(steps=steps, steps_known=True, steps_source="terminal")
    if result is not None:
        steps = result["steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("report_step_count_invalid")
        return dict(steps=steps, steps_known=True, steps_source="result")
    return dict(steps=None, steps_known=False, steps_source="unavailable")


def _detail(item: E2CheckpointRow, workspace: E2Workspace, requests: dict) -> dict[str, object]:
    """从核验过的原始结果、Trace和钱包提取指标，二次读取仍校验字节指纹。"""
    data, ref = item.payload, item.payload["trace"]
    events = _read_events(item, workspace)
    result, group = data["result"], item.run.group_id
    verdict = FactVerdict.model_validate(result["verdict"]) if result is not None else None
    if data["predicted_label"] != (verdict.label if verdict else ""):
        raise ValueError("report_prediction_mismatch")
    estimates = [e.data["input_token_estimate"] for e in events if e.type == "input_estimate"]
    ratios = [e.data["actual_to_estimate_ratio"] for e in events
        if e.type == "llm_call" and e.data.get("actual_to_estimate_ratio") is not None]
    conditions, role = workspace.conditions, "single" if group == "single_shot" else "loop"
    path = json.loads(conditions["prompt_roles"])[role]
    retained = sum(float(requests[identifier].reserved) for identifier in item.run.request_ids
        if requests[identifier].state in {"reserved", "dispatched", "unknown"})
    valid = bool(verdict) and (group == "iterative" or not citation_issues(verdict, len(result["evidence"])))
    step_summary = _loop_step_summary(events, result) if group == "iterative" else dict(
        steps=0, steps_known=True, steps_source="not_applicable")
    return {**step_summary, "sample_id": item.run.sample_id, "run_id": item.run.run_id, "variant": group, "split": "dev",
        "execution_status": "finished", "gold_label": data["gold_label"], "predicted_label": data["predicted_label"],
        "correct": bool(verdict and verdict.label == data["gold_label"]), "stop_reason": data["stop_reason"],
        "outcome_class": ref["outcome_class"],
        "calls": data["calls"], "tool_calls": sum(e.type == "tool_call" for e in events),
        "input_tokens": data["input_tokens"], "output_tokens": data["output_tokens"], "cost_cny": data["cost_cny"],
        "known_cost_subtotal_cny": data["known_cost_subtotal_cny"], "usage_known": data["usage_known"],
        "usage_issues": data["issues"], "retained_cny": retained, "latency_ms": data["latency_ms"],
        "input_estimate_max": max(estimates, default=None), "actual_to_estimate_max": max(ratios, default=None),
        "citation_format_valid": valid, "citation_ok": bool(verdict) and group == "iterative",
        "submission_check": "passed" if verdict and group == "iterative" else "not_applicable" if verdict else "failed",
        "trace_path": ref["path"], "trace_sha256": ref["sha256"], "prompt_path": path,
        "prompt_sha256": conditions["file:" + path],
        "full_system_sha256": conditions[role + "_system_sha256"],
        "contract_version": conditions.get("loop_contract_version", "") if role == "loop" else "fact-verdict-v3",
        "condition_sha256": item.condition_sha256, "experiment_id": item.experiment_id, "result": result}


def _group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """失败题保留在端到端分母，引用比例与合格判断比例单列。"""
    executed = [row for row in rows if row["execution_status"] == "finished"]
    stats = fact_check_summary(executed) if executed else {"total": 0, "correct": 0, "macro_f1": None,
        "cost_cny": None, "known_cost_subtotal_cny": 0, "unknown_cost_rows": 0}
    costs = [row["cost_cny"] for row in executed if row["cost_cny"] is not None]
    latencies = [row["latency_ms"] for row in executed]
    ratios = [row["actual_to_estimate_max"] for row in executed if row["actual_to_estimate_max"] is not None]
    submitted = [row for row in executed if row["outcome_class"] == "submitted_judgment"]
    failed = [row for row in executed if row["outcome_class"] == "no_valid_result"]
    return stats | {"planned": len(rows), "not_executed": len(rows) - len(executed),
        "submitted": len(submitted), "normal_submitted": sum(row["stop_reason"] != "forced_final" for row in submitted),
        "forced_final": sum(row["stop_reason"] == "forced_final" for row in submitted), "no_valid_result": len(failed),
        "valid_judgment_ratio": len(submitted) / len(executed) if executed else None,
        "citation_format_valid_ratio": sum(row["citation_format_valid"] for row in executed) / len(executed) if executed else None,
        "citation_ok_count": sum(row["citation_ok"] for row in executed),
        "submission_checks": dict(Counter(row["submission_check"] for row in executed)),
        "failure_reasons": dict(Counter(row["stop_reason"] for row in failed)),
        "cost_median_known": median(costs) if costs else None, "cost_max_known": max(costs, default=None),
        "latency_p50_ms": _percentile(latencies, 0.5), "latency_p95_ms": _percentile(latencies, 0.95),
        "actual_to_estimate_max": max(ratios, default=None)}


def summarize_e2(rows: list[dict[str, object]]) -> dict[str, object]:
    """只对完整执行配对比较正确性，半对和未执行单列。"""
    groups = {name: _group_summary([row for row in rows if row["variant"] == name])
        for name in ("single_shot", "iterative")}
    cases: dict[str, dict[str, dict]] = {}
    for row in rows:
        if row["execution_status"] == "finished":
            cases.setdefault(row["sample_id"], {})[row["variant"]] = row
    pairs = {key: value for key, value in cases.items() if len(value) == 2}
    table = Counter()
    corrected, regressed = [], []
    for sample, pair in pairs.items():
        single, loop = pair["single_shot"], pair["iterative"]
        key = ("both_correct" if loop["correct"] else "single_only") if single["correct"] else (
            "iterative_only" if loop["correct"] else "both_wrong")
        table[key] += 1
        if key == "iterative_only" and single["predicted_label"] == "信息不足":
            corrected.append(sample)
        if key == "single_only":
            regressed.append(sample)
    b, c = table["single_only"], table["iterative_only"]
    p = min(1.0, 2 * sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)) if pairs else None
    return {"groups": groups, "completed_pairs": len(pairs), "half_pairs": [key for key, value in cases.items() if len(value) == 1],
        "paired_table": {key: table[key] for key in ("both_correct", "single_only", "iterative_only", "both_wrong")},
        "mcnemar_method": "two-sided-exact; executed-failures=wrong; half-pairs-excluded",
        "mcnemar_p": p, "insufficient_corrected_ids": corrected, "regressed_ids": regressed,
        "interpretation": "dev试跑为描述性观察；比较两条完整路径，不单独归因为多轮。费用按回执token和确认单价估算，非服务商账单。"}


def _report_rows(workspace: E2Workspace, journal: AnswerJournal, session: ExperimentSession) -> tuple[object, list[dict]]:
    """审计受阻只容许停账/未知费用的部分报告，身份或Trace不符不出成绩。"""
    items, history = journal.read_answers(), session.load_history()
    audit = journal.audit_saved_answers(history)
    operational = {"ledger_halted", "unresolved_request"}
    if any(issue.code not in operational for issue in audit.issues):
        return audit, []
    suffix = workspace.plan[len(items):]
    if any(any(workspace.folder.glob(f"*/{slot.run_id}.jsonl")) for slot in suffix):
        return audit, []
    state = replay_history(history, workspace.manifest)
    rows = [_detail(item, workspace, state.requests) for item in items]
    golds = {q["sample_id"]: {"真": "支持", "假": "反驳", "无法判定": "信息不足"}[q["gold_veracity"]]
        for q in json.loads(workspace.conditions["questions"])}
    if any(row["gold_label"] != golds[row["sample_id"]] for row in rows):
        raise ValueError("report_gold_mismatch")
    for slot in suffix:
        rows.append(dict(sample_id=slot.sample_id, run_id=slot.run_id, variant=slot.group_id, split="dev",
            execution_status="not_executed", outcome_class="not_executed", stop_reason="not_executed",
            gold_label=golds[slot.sample_id], predicted_label="", correct=None, cost_cny=None,
            known_cost_subtotal_cny=None, trace_path="", experiment_id=workspace.manifest["experiment_id"]))
    return audit, rows


def _write_result(workspace: E2Workspace, kind: str, suffix: str, raw: bytes, purpose: str) -> Path:
    """先在工作区完整写入并同步，再独占发布结果，最后登记索引。"""
    name = f"{date.today()}_{kind}_{uuid4().hex}.{suffix}"
    temporary = workspace.folder / (name + ".partial")
    target = PROJECT_ROOT / "results" / name
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _publish(temporary, target)
    register_result(target, purpose)
    return target


def _csv_bytes(rows: list[dict]) -> bytes:
    """同一组保留完整结果JSON；未知值写为空而不是零。"""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def _pilot_gate(summary: dict, audit: object, rows: list[dict], session: ExperimentSession) -> dict:
    """完整且费用可信才能评价试跑门槛；通过也不授予full付费许可。"""
    config, groups = get_evaluation_config(), summary["groups"]
    over = sum("reservation_exceeded" in row.get("usage_issues", []) for row in rows)
    complete = audit.consistent and all(group["planned"] == sum(config.fact_check_counts.values()) and
        group["not_executed"] == 0 and group["unknown_cost_rows"] == 0 for group in groups.values())
    loop_cost = groups["iterative"]["cost_median_known"]
    passed = complete and loop_cost is not None and loop_cost <= config.pilot_max_median_cost_cny and (
        all(group["no_valid_result"] <= config.pilot_max_no_valid_results for group in groups.values())) and (
        over <= config.pilot_max_reservation_exceeded) and session.snapshot().halt_reason != "reservation_exceeded"
    return dict(passed=passed, complete=complete, reservation_exceeded=over,
        max_median_cost_cny=config.pilot_max_median_cost_cny, max_no_valid_results=config.pilot_max_no_valid_results,
        full_authorized=False)


def _example_steps(workspace: E2Workspace, journal: AnswerJournal) -> list[dict]:
    """选第一份已执行循环展示完整调用，不为凑改对案例挑题。"""
    item = next((item for item in journal.read_answers() if item.run.group_id == "iterative"), None)
    steps = []
    for event in _read_events(item, workspace) if item else []:
        if event.type in {"llm_call", "tool_call", "run_outcome"}:
            entry = dict(step=event.step, event=event.type, tool=event.data.get("tool"),
                status=event.data.get("status", event.data.get("outcome_class")), reason=event.data.get("reason"),
                cost_cny=event.data.get("cost_cny"), input_tokens=event.data.get("input_tokens"),
                output_tokens=event.data.get("output_tokens"))
            if event.type == "tool_call" and event.data.get("archive"):
                raw = (workspace.folder / event.data["archive"]["path"]).read_bytes()
                if sha256(raw).hexdigest() != event.data["archive"]["sha256"]:
                    raise ValueError("report_archive_changed")
                saved = json.loads(raw)
                entry["request"] = saved["payload"].get("request", {})
            steps.append(entry)
    return steps


def _handoff(report: dict, audit_path: str) -> Path:
    """一页可读交接，完整数字和完整步骤始终保留在审计JSON。"""
    lines = [f"# E2 {report['mode']} 运行交接", "", f"实验：{report['experiment_id']}；停止：{report['stop_reason']}",
        "费用按回执token与确认单价估算；dev描述性观察，不是test成绩。", f"完整报告：{audit_path}", ""]
    summary = report["summary"]
    if summary:
        for name, group in summary["groups"].items():
            lines.append(f"- {name}：执行{group['total']}/{group['planned']}；正确{group['correct']}；Macro-F1={group['macro_f1']}；已知小计={group['known_cost_subtotal_cny']}；未知费用{group['unknown_cost_rows']}份。")
    lines += ["", "| 步骤 | 动作与实际记录 |", "|---|---|"]
    lines += [f"| {step['step']} | {_json(step).replace('|', ' / ')} |" for step in report["example_steps"]]
    if len(lines) > 30:
        lines = lines[:9] + ["完整步骤超过一页，请打开上述审计报告的example_steps；此处不截取冒充完整。"]
    folder = PROJECT_ROOT / "docs/handoff"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"E2-{report['mode']}-{uuid4().hex}.md"
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return path


def publish_e2_results(workspace: E2Workspace, journal: AnswerJournal, session: ExperimentSession,
                       result: DispatchResult) -> dict[str, str]:
    """只追加两组答卷与审计报告，写失败不重跑模型、不覆盖历史。"""
    audit, rows = _report_rows(workspace, journal, session)
    if result.reason in {"experiment_conditions_changed", "orphan_trace_requires_review"}:
        rows = []
    files = {}
    summary = summarize_e2(rows) if rows else None
    if rows:
        for group in ("single_shot", "iterative"):
            subset = [row for row in rows if row["variant"] == group]
            files[group] = str(_write_result(workspace, "factcheck", "csv", _csv_bytes(subset),
                f"E2 {workspace.manifest['mode']} dev {group}；执行失败保留、未执行单列"))
    report = dict(kind="audit", mode=workspace.manifest["mode"], experiment_id=workspace.manifest["experiment_id"],
        stop_status=result.status, stop_reason=result.reason, audit=audit.model_dump(mode="json"),
        wallet=session.snapshot().model_dump(mode="json"), summary=summary,
        files={key: {"path": value, "sha256": sha256(Path(value).read_bytes()).hexdigest()} for key, value in files.items()})
    report["example_steps"] = _example_steps(workspace, journal) if rows else []
    if summary and workspace.manifest["mode"] == "pilot":
        report["pilot_gate"] = _pilot_gate(summary, audit, rows, session)
    files["audit"] = str(_write_result(workspace, "audit", "jsonl", (_json(report) + "\n").encode(),
        "E2运行审计及描述性报告；dev非test，Mock不作为真实成绩"))
    files["handoff"] = str(_handoff(report, files["audit"]))
    return files
