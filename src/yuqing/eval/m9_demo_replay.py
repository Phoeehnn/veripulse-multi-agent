"""Replay a saved M9 case result and deterministically re-score its action profile."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.yuqing.agents.arbiter import score_case
from src.yuqing.agents.fact_check import compute_fact_risk
from src.yuqing.eval.m9_demo import DEMO_CASES
from src.yuqing.settings import PROJECT_ROOT
from src.yuqing.schemas import ComplianceResult, FactVerdict


def replay_m9_demo(run_dir: Path) -> list[dict[str, Any]]:
    """Read archived answers only; no model, network, wallet mutation, or graph invocation."""
    target, traces = run_dir.resolve(), (PROJECT_ROOT / "traces").resolve()
    answers, summary_path = target / "answers.jsonl", target / "summary.json"
    if (not target.is_relative_to(traces) or not answers.is_file()
            or not summary_path.is_file()):
        raise ValueError("m9_demo_archive_incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("audit") != "verified" or summary.get("samples") != list(DEMO_CASES)
            or summary.get("answers_sha256") != sha256(answers.read_bytes()).hexdigest()):
        raise ValueError("m9_demo_archive_integrity_invalid")
    rows = [json.loads(line) for line in answers.read_text(encoding="utf-8").splitlines()
        if line.strip()]
    if len(rows) != len(DEMO_CASES) or tuple(row.get("sample_id") for row in rows) != DEMO_CASES:
        raise ValueError("m9_demo_archive_identity_invalid")
    return [_replay_row(row) for row in rows]


def _replay_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute the content-platform action from a saved, already validated case state."""
    state = row.get("state")
    if not isinstance(state, dict) or not isinstance(state.get("risk"), dict):
        raise ValueError("m9_demo_case_state_missing")
    fact = state.get("fact")
    verdict = FactVerdict.model_validate(fact, strict=True) if isinstance(fact, dict) else None
    rounds = state.get("fact_rounds", [])
    last_round = rounds[-1] if rounds and isinstance(rounds[-1], dict) else {}
    propagation = state.get("propagation") or {}
    decisions = state.get("decisions", [])
    decision = decisions[-1] if decisions and isinstance(decisions[-1], dict) else {}
    compliance = ComplianceResult.model_validate(state["compliance"], strict=True)
    coverage = str(state.get("coverage", "none"))
    inputs = {"fact_status": verdict.label if verdict else None,
        "fact_risk": compute_fact_risk(verdict) if verdict else None,
        "fact_coverage": coverage, "fact_failed": verdict is None and coverage != "none",
        "fact_citation_ok": bool(verdict and last_round.get("status") == "completed"),
        "emotion_level": propagation.get("emotion_level"),
        "spread_level": propagation.get("spread_level"), "compliance": compliance,
        "arbiter_requests_review": bool(decision.get("needs_human_review", False)),
        "adjustment": decision.get("adjustment", 0),
        "adjustment_reason": decision.get("adjustment_reason"),
        "unresolved_key_conflict": bool(state.get("fact_selection", {}).get(
            "force_review", False))}
    revised = score_case(**inputs, scenario="content_platform")
    previous = state["risk"]
    return {"sample_id": row["sample_id"], "from_scenario": previous.get("scenario"),
        "from_action": previous.get("disposition"), "to_scenario": revised.scenario,
        "to_action": revised.disposition, "risk_level_unchanged": revised.level == previous.get("level"),
        "model_calls": 0}
