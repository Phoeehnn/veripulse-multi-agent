"""Conservative continuation accounting; never clear or alter a halted predecessor wallet."""
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore
from src.yuqing.eval.experiment_budget_recovery import replay_history


def predecessor_hold(root: Path) -> dict[str, str]:
    """Read and verify the original hash chain through a read-only SQLite connection."""
    path = root / "wallet.sqlite"
    store = ExperimentBudgetStore(path, sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True))
    try:
        history = store.load()
        snapshot = replay_history(history, history.manifest).snapshot()
        if not snapshot.halted or not snapshot.unknown_requests or snapshot.pending_requests:
            raise ValueError("continuation_requires_closed_unknown_predecessor")
        return {"sha256": sha256(path.read_bytes()).hexdigest(),
                "committed_cny": str(snapshot.committed_cost_cny),
                "known_cny": str(snapshot.known_cost_subtotal_cny),
                "unknown_reserved_cny": str(snapshot.unknown_reserved_cny),
                "experiment_id": snapshot.experiment_id}
    finally:
        store.close()


def prepare_continuation(root: Path, *, total_limit: Decimal) -> dict[str, object]:
    """Bind one continuation allowance to the unchanged predecessor and original total cap."""
    original = json.loads((root / "session-manifest.json").read_text(encoding="utf-8"))
    if Decimal(original["max_cost_cny"]) != total_limit:
        raise ValueError("continuation_total_limit_mismatch")
    hold = predecessor_hold(root)
    remaining = total_limit - Decimal(hold["committed_cny"])
    if remaining <= 0:
        raise ValueError("research_cumulative_budget_exhausted")
    expected = {"predecessor": hold, "total_limit_cny": str(total_limit),
                "remaining_limit_cny": str(remaining)}
    path = root / "continuation-plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("continuation_predecessor_changed")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2)
    return expected


def continuation_root(root: Path, *, total_limit: Decimal) -> tuple[Path, Decimal]:
    """Always reopen the same derived wallet; repeated restarts never restore its allowance."""
    plan = prepare_continuation(root, total_limit=total_limit)
    target = root / "continuation"
    target.mkdir(exist_ok=True)
    return target, Decimal(plan["remaining_limit_cny"])
