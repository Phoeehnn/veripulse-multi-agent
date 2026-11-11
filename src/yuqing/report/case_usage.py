"""Aggregate per-case model use from verified run traces and the case wallet."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.harness.trace import RunContext, get_run_usage, use_run_context


def summarize_case_usage(contexts: tuple[RunContext, ...], wallet: object | None) -> dict[str, Any]:
    """Combine request/token totals from Trace with settled cost from its wallet ledger."""
    runs = {item.run_id: item for item in contexts}
    if len(runs) != len(contexts):
        raise ValueError("duplicate_case_usage_run")
    calls, input_tokens, output_tokens = 0, 0, 0
    known_cost, issues = Decimal(0), []
    trace_complete = True
    for run in runs.values():
        try:
            with use_run_context(run):
                usage = get_run_usage(run.run_id)
        except (OSError, ValueError) as exc:
            trace_complete = False
            issues.append(type(exc).__name__)
            continue
        calls += usage.calls
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        known_cost += Decimal(str(usage.known_cost_subtotal_cny))
        if not usage.usage_known:
            trace_complete = False
            issues.extend(usage.issues)

    ledger = _wallet_totals(wallet, frozenset(runs))
    ledger_calls = ledger["request_count"]
    reconciled = ledger["available"] and ledger_calls == calls
    if ledger["available"] and not reconciled:
        issues.append("trace_wallet_request_count_mismatch")
    cost_mismatch = ledger["available"] and abs(ledger["settled_cost"] - known_cost) > Decimal("0.000001")
    if cost_mismatch:
        issues.append("trace_wallet_cost_mismatch")
    if ledger["unknown_or_pending"]:
        issues.append("wallet_unknown_or_pending")
    if not ledger["available"]:
        issues.append("wallet_unavailable")
    cost_known = bool(ledger["available"] and reconciled and not cost_mismatch and not ledger["unknown_or_pending"]
        and trace_complete and not ledger["identity_incomplete"])
    return {"request_count": calls, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "estimated_cost_cny": _money(ledger["settled_cost"])
            if cost_known else None,
        "known_cost_subtotal_cny": _money(ledger["settled_cost"] if ledger["available"]
            else known_cost), "unknown_or_pending_requests": ledger["unknown_or_pending"],
        "usage_complete": bool(trace_complete and reconciled and not ledger["identity_incomplete"]),
        "issues": sorted(set(issues))}


def _wallet_totals(wallet: object | None, run_ids: frozenset[str]) -> dict[str, Any]:
    """Replay the case wallet and count only requests bound to this invocation's runs."""
    if wallet is None:
        return {"available": False, "request_count": 0, "settled_cost": Decimal(0),
            "unknown_or_pending": 0, "identity_incomplete": True}
    try:
        manifest = wallet.manifest_snapshot()
        state = replay_history(wallet.load_history(), manifest)
    except (AttributeError, OSError, TypeError, ValueError):
        return {"available": False, "request_count": 0, "settled_cost": Decimal(0),
            "unknown_or_pending": 0, "identity_incomplete": True}
    requests = [item for item in state.requests.values()
        if item.identity.get("run_id") in run_ids]
    incomplete = any(item.state not in {"settled", "unknown", "cancelled"} for item in requests)
    active = [item for item in requests if item.state != "cancelled"]
    settled = sum((item.actual for item in active if item.state == "settled" and item.actual is not None),
        Decimal(0))
    pending = sum(item.state in {"reserved", "dispatched", "unknown"} for item in active)
    return {"available": True, "request_count": len(active), "settled_cost": settled,
        "unknown_or_pending": pending, "identity_incomplete": incomplete}


def _money(value: Decimal) -> str:
    """Render yuan with six decimal places for stable, readable reports."""
    return f"{value:.6f}"
