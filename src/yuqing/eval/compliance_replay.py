"""Read archived cases and compare deterministic compliance scoring offline."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.yuqing.agents.arbiter import score_case
from src.yuqing.agents.compliance import check_compliance
from src.yuqing.agents.fact_check import compute_fact_risk
from experiments.archive.m8_cache_validation import _decode_propagation_payload, _read_tool_archive
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.schemas import ComplianceResult, FactVerdictV2Envelope


class InputAudit:
    """Remember source bytes and reject files that change while replay runs."""

    def __init__(self, root: Path) -> None:
        """Restrict source reads to the project and track exact hashes."""
        self.root = root.resolve(strict=True)
        self.hashes: dict[str, str] = {}

    def read(self, path: Path) -> bytes:
        """Read a regular in-project file and retain its first observed fingerprint."""
        path = path.resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("source_outside_project")
        raw = path.read_bytes()
        key = path.relative_to(self.root).as_posix()
        digest = sha256(raw).hexdigest()
        if key in self.hashes and self.hashes[key] != digest:
            raise ValueError("source_changed_during_read")
        self.hashes[key] = digest
        return raw

    def verify(self) -> None:
        """Recheck all input bytes before publishing any output."""
        for key, expected in self.hashes.items():
            path = (self.root / key).resolve(strict=True)
            if not path.is_relative_to(self.root) or sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("source_changed_before_publish")


def propagation_comments(state: dict[str, Any], sample_id: str, run_root: Path,
                         audit: InputAudit) -> tuple[tuple[str, str], ...]:
    """Reconstruct only successful, archived propagation deliveries from this run."""
    original = state["compliance"]
    if original["comment_status"] == "no_comments":
        return ()
    report = state.get("propagation")
    rounds = state.get("prop_rounds") or []
    if not isinstance(report, dict) or len(rounds) != 1 or not rounds[0].get("trace_path"):
        raise ValueError("original_comment_input_unavailable")
    path = Path(rounds[0]["trace_path"]).resolve(strict=True)
    if not path.is_relative_to(run_root.resolve(strict=True)):
        raise ValueError("propagation_trace_wrong_run")
    events = [TraceEvent.model_validate_json(line, strict=True)
              for line in audit.read(path).splitlines() if line]
    run_id = rounds[0]["run_id"]
    if any((event.run_id, event.sample_id, event.agent) !=
           (run_id, sample_id, "propagation_analyst") for event in events):
        raise ValueError("propagation_trace_identity_mismatch")
    return _read_comments(events, path.parents[1], report, audit)


def _read_comments(events: list[TraceEvent], root: Path, report: dict[str, Any],
                   audit: InputAudit) -> tuple[tuple[str, str], ...]:
    """Verify accepted report and collect only successful comment tool payloads."""
    comments: dict[str, str] = {}
    accepted = 0
    for event in events:
        if event.type != "tool_call" or event.data.get("status") != "success":
            continue
        archive = event.data.get("archive")
        if not isinstance(archive, dict) or not isinstance(archive.get("path"), str):
            raise ValueError("propagation_tool_archive_missing")
        audit.read(root / archive["path"])
        item = _read_tool_archive(event, root)
        payload = item["payload"]
        if event.data.get("tool") == "submit_propagation_report":
            if (payload.get("feedback", {}).get("accepted") is True
                    and item.get("request", {}).get("arguments") == report):
                accepted += 1
        if event.data.get("tool") in {"sample_comments", "search_comments"}:
            if payload.get("status") == "success":
                for comment in _decode_propagation_payload(payload).comments:
                    comments[comment.comment_id] = comment.text
    if accepted != 1 or not comments:
        raise ValueError("original_comment_delivery_unverified")
    return tuple(comments.items())


def score_inputs(state: dict[str, Any], compliance: ComplianceResult) -> dict[str, Any]:
    """Reconstruct the M5 score arguments from a complete archived state."""
    for key in ("coverage", "fact", "propagation", "decisions", "fact_selection", "risk"):
        if key not in state:
            raise ValueError("original_scoring_input_missing")
    verdict = FactVerdictV2Envelope.model_validate(state["fact"], strict=True) if state["fact"] is not None else None
    if state["propagation"] is not None and (not isinstance(state["propagation"], dict)
            or not {"emotion_level", "spread_level"}.issubset(state["propagation"])):
        raise ValueError("original_propagation_input_missing")
    if not isinstance(state["decisions"], list):
        raise ValueError("original_decisions_invalid")
    prop = state["propagation"] or {}
    decision = (state["decisions"] or [{}])[-1]
    if state["decisions"] and (not isinstance(decision, dict)
            or not {"needs_human_review", "adjustment", "adjustment_reason"}.issubset(decision)):
        raise ValueError("original_decision_input_missing")
    choice = state["fact_selection"] or {}
    if "force_review" not in choice or "scenario" not in state["risk"]:
        raise ValueError("original_selection_or_scenario_missing")
    force = choice["force_review"]
    if type(force) is not bool:
        raise ValueError("original_force_review_invalid")
    return {"fact_status": "信息不足" if force else verdict.label if verdict else None,
        "fact_risk": compute_fact_risk(verdict) if verdict else None,
        "fact_coverage": state["coverage"], "fact_failed": verdict is None and state["coverage"] != "none",
        "fact_citation_ok": verdict is not None, "emotion_level": prop.get("emotion_level"),
        "spread_level": prop.get("spread_level"), "compliance": compliance,
        "arbiter_requests_review": True if force else decision.get("needs_human_review", False),
        "adjustment": decision.get("adjustment", 0), "adjustment_reason": decision.get("adjustment_reason"),
        "unresolved_key_conflict": force, "scenario": state["risk"]["scenario"]}


def replay_case(state: dict[str, Any], sample_id: str, run_root: Path,
                rules_path: Path, audit: InputAudit) -> dict[str, Any]:
    """Require original rule/score reproduction before computing a new result."""
    sample = state.get("sample") or {}
    if sample.get("sample_id") != sample_id or not isinstance(sample.get("claim"), str):
        raise ValueError("original_sample_identity_mismatch")
    if (state.get("ticket") or {}).get("claim") != sample["claim"]:
        raise ValueError("original_claim_identity_mismatch")
    comments = propagation_comments(state, sample_id, run_root, audit)
    before = ComplianceResult.model_validate(state["compliance"], strict=True)
    recreated = check_compliance(sample["claim"], comments, rules_path=audit.root / "config/rules_compliance.yaml")
    if recreated != before:
        raise ValueError("original_compliance_does_not_reproduce")
    args = score_inputs(state, before)
    old = score_case(**args).model_dump(mode="json")
    if old != state["risk"]:
        raise ValueError("original_risk_does_not_reproduce")
    after = check_compliance(sample["claim"], comments, rules_path=rules_path)
    if after.claim_status == "check_failed" or after.comment_status == "check_failed":
        raise ValueError("candidate_rule_check_failed")
    args["compliance"] = after
    new = score_case(**args).model_dump(mode="json")
    return {"old_compliance": before.model_dump(mode="json"),
        "new_compliance": after.model_dump(mode="json"), "old_risk": old, "new_risk": new,
        "input_comment_count": len(comments)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count completed, skipped and changes without merging distinct result groups."""
    statuses = Counter(str(row["status"]) for row in rows)
    directions: Counter[str] = Counter()
    for row in rows:
        if row["status"] != "recomputed":
            continue
        old, new = row["old_risk"]["level"], row["new_risk"]["level"]
        if type(old) is not int or type(new) is not int:
            directions["no_comparable_level"] += 1
        else:
            directions["higher" if new > old else "lower" if new < old else "unchanged"] += 1
    return {"planned": len(rows), "status_counts": dict(statuses), "level_changes": dict(directions)}
