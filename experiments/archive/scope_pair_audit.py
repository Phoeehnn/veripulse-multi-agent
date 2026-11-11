"""两阶段先导的完整槽位、持久费用及Trace终态审计。"""

from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path

from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.harness.fact_v2_contract import build_fact_run_context_v2, DeliveredEvidenceV2, FactRunContextV2, validate_fact_v2_core
from src.yuqing.harness.initial_evidence import parse_delivery_message


def audit_slots(rows: list[dict[str, object]], expected: tuple[tuple[str, str], ...],
                session: ExperimentSession, root: Path, caps: tuple[Decimal, Decimal, Decimal]) -> list[str]:
    """独立重放钱包并逐项对齐全部槽位，失败槽不能被静默删除。"""
    issues, seen = [], set()
    context = build_fact_run_context_v2("initial", initial_manifest_path=Path("audit-packet.json"))
    if [(r["sample_id"], r["arm"]) for r in rows] != list(expected):
        issues.append("slot_roster_or_order_mismatch")
    history = session.load_history()
    recovered = replay_history(history, history.manifest)
    totals = {"S": Decimal(0), "M": Decimal(0)}
    for row in rows:
        key = (row["sample_id"], row["arm"])
        if key in seen:
            issues.append("duplicate_slot")
        seen.add(key)
        stages = row["stages"]
        if len(stages) > 2 or [s["stage"] for s in stages] != list(range(1, len(stages) + 1)):
            issues.append("stage_order_or_count")
        if row["status"] not in {"completed", "fallback", "failed", "stopped"}:
            issues.append("slot_not_terminal")
        reports = [s["report"] for s in stages if s["accepted"]]
        if any(s["accepted"] != (s["report"] is not None) for s in stages):
            issues.append("accepted_report_mismatch")
        if row["report"] != (reports[-1] if reports else None):
            issues.append("final_selection_mismatch")
        if row["status"] == "completed" and (len(stages) != 2 or not all(s["accepted"] for s in stages)):
            issues.append("invalid_completed_slot")
        if ((row["status"] == "fallback" and (not reports or (len(stages) == 2 and stages[-1]["accepted"])))
                or (row["status"] == "failed" and (reports or not stages))
                or (row["status"] == "stopped" and stages)):
            issues.append("invalid_terminal_state")
        cost = sum((Decimal(s["cost_cny"] or "0") for s in stages), Decimal(0))
        totals[row["arm"]] += cost
        if cost > caps[1]:
            issues.append("case_cap_exceeded")
        issues.extend(_audit_stages(row, recovered.requests, root, caps[0], context))
    recorded = [s["request_id"] for r in rows for s in r["stages"] if s["request_id"] in recovered.requests]
    if len(recorded) != len(set(recorded)) or set(recorded) != set(recovered.requests):
        issues.append("request_slot_bijection_failed")
    if sum(totals.values()) != recovered.snapshot().known_cost_subtotal_cny:
        issues.append("wallet_cost_mismatch")
    if any(value > caps[1] * 12 for value in totals.values()) or sum(totals.values()) > caps[2]:
        issues.append("group_or_total_cap_exceeded")
    if recovered.snapshot().pending_requests or recovered.snapshot().unknown_requests:
        issues.append("unsettled_usage")
    return sorted(set(issues))


def _audit_stages(row: dict[str, object], requests: dict, root: Path, phase_cap: Decimal,
                  context: FactRunContextV2) -> list[str]:
    """核阶段费用、请求身份及所有归档SHA，不把未知费用当零结算。"""
    issues = []
    for stage in row["stages"]:
        request = requests.get(stage["request_id"])
        if request is None:
            if stage["accepted"] or stage["report"] is not None:
                issues.append("unsent_stage_has_report")
            if stage["status"] not in {"budget_denied", "startup_error"}:
                issues.append("missing_wallet_request")
        else:
            expected = {"sample_id": row["sample_id"], "group_id": f'{row["arm"]}/{row["sample_id"]}',
                "run_id": stage["run_id"], "variant": f'stage{stage["stage"]}'}
            if dict(request.identity) != expected:
                issues.append("request_identity_mismatch")
            actual = None if stage["cost_cny"] is None else Decimal(stage["cost_cny"])
            if request.actual != actual or request.reserved > phase_cap:
                issues.append("stage_cost_or_reservation_mismatch")
        if stage["cost_cny"] is not None and Decimal(stage["cost_cny"]) > phase_cap:
            issues.append("phase_cap_exceeded")
        trace = Path(stage["trace_path"]).resolve()
        if not trace.is_relative_to(root.resolve()) or not trace.is_file():
            issues.append("trace_missing_or_outside_run")
            continue
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        issues.extend(_bind_submission(stage, request, events, root, context))
        for event in events:
            if event["run_id"] != stage["run_id"] or event["sample_id"] != row["sample_id"]:
                issues.append("trace_identity_mismatch")
            archive = event["data"].get("archive")
            if archive:
                target = (root / archive["path"]).resolve()
                if (not target.is_relative_to(root.resolve()) or not target.is_file()
                        or sha256(target.read_bytes()).hexdigest() != archive["sha256"]):
                    issues.append("archive_hash_mismatch")
    return issues


def _payload(event: dict, root: Path) -> dict:
    """只读取根内且哈希匹配的归档，不以路径存在代替完整性。"""
    archive = event["data"]["archive"]
    path = (root / archive["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path.read_bytes()).hexdigest() != archive["sha256"]:
        raise ValueError("archive_hash_mismatch")
    return json.loads(path.read_bytes())["payload"]


def _bind_submission(stage: dict, request: object, events: list[dict], root: Path,
                     context: FactRunContextV2) -> list[str]:
    """将已发送请求、真实响应及重新校验的提交核心绑定到汇总。"""
    if request is None:
        return []
    try:
        dispatch = [e for e in events if e["type"] == "llm_request"]
        if len(dispatch) != 1 or dispatch[0]["data"]["request_id"] != stage["request_id"]:
            return ["request_trace_binding_failed"]
        sent = _payload(dispatch[0], root)
        responses = [e for e in events if e["type"] == "model_response_archive"]
        submissions = [e for e in events if e["type"] == "scope_submission"]
        if request.state == "settled" and len(responses) != 1:
            return ["response_trace_binding_failed"]
        if stage["status"] != "success":
            return ["unexpected_submission"] if submissions or stage["accepted"] else []
        if len(submissions) != 1 or len(responses) != 1:
            return ["submission_trace_binding_failed"]
        response = _payload(responses[0], root)
        submission = _payload(submissions[0], root)
        if response["request_id"] != stage["request_id"] or sent["request_id"] != stage["request_id"]:
            return ["response_request_mismatch"]
        raw = response["body"]["choices"][0]["message"]["tool_calls"][0]["function"]
        args = context.response_schema.model_validate_json(raw["arguments"], strict=True).model_dump(mode="json", exclude_unset=True)
        entries = parse_delivery_message(sent["body"]["messages"][1]["content"])
        checked = validate_fact_v2_core(args, delivered={e.evidence_id: DeliveredEvidenceV2(text=e.text,
            read_end=len(e.text)) for e in entries}, limits=context.limits,
            plan_max_text_chars=context.plan_max_text_chars, mode="initial")
        report = checked.report.model_dump(mode="json") if checked.validation.accepted else None
        if (raw["name"] != "submit_verdict" or submission["arguments"] != args
                or submission["validation"] != checked.validation.model_dump(mode="json")
                or stage["accepted"] != checked.validation.accepted or stage["report"] != report):
            return ["submission_report_binding_failed"]
    except ValueError as exc:
        return ["archive_hash_mismatch"] if str(exc) == "archive_hash_mismatch" else ["archive_binding_invalid"]
    except (OSError, KeyError, TypeError, IndexError):
        return ["archive_hash_mismatch_or_binding_invalid"]
    return []
