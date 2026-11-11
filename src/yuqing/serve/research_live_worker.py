"""Core-process live sample bridge: read-only preview, explicitly authorized execution only."""
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import sqlite3
import sys
from src.yuqing import settings
from src.yuqing.eval.experiment_budget_store import ExperimentBudgetStore
from src.yuqing.eval.experiment_budget_recovery import replay_history
from src.yuqing.serve.research_recovery import predecessor_hold
from src.yuqing.serve.research_increment_live import LiveSamplePlan, LiveSampleExecutor, validate_live_plan
from src.yuqing.serve.research_live_ui import LiveUIPlan
from src.yuqing.serve.research_public_activation import APPROVED_URLS


def load_live_plan(path: Path) -> LiveSamplePlan:
    """Read only a checklist within the D-drive worktree; it does not grant call permission."""
    path=path.resolve()
    if path.drive.lower()!="d:" or not path.is_relative_to(settings.PROJECT_ROOT.resolve()):
        raise ValueError("live_checklist_outside_worktree")
    data=json.loads(path.read_text(encoding="utf-8"))
    model=data["model_agent_process_observation"]
    return LiveSamplePlan(data["main_report_case"],data["source_case"],data["main_snapshot_sha256"],
        data["material_file_sha256"],data["prompt_sha256"],model["model"],model["base_url"],model["thinking"],data["price_fingerprint"])


def _read_wallet(path: Path) -> tuple[object, dict[str, object]]:
    """Replay an existing SQLite chain through a read-only connection, never resume/create it."""
    store=ExperimentBudgetStore(path,sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True))
    try:
        history=store.load()
        return replay_history(history,history.manifest).snapshot(),dict(history.manifest)
    finally:
        store.close()


def preview_live_plan(root: Path, plan: LiveSamplePlan) -> LiveUIPlan:
    """Derive a fresh UI plan from immutable source checks and current conservative wallet state."""
    root=root.resolve()
    if root.drive.lower()!="d:" or not root.is_relative_to((settings.PROJECT_ROOT/"traces").resolve()):
        raise ValueError("live_preview_root_not_isolated_on_d")
    hold=predecessor_hold(root)
    continuation=json.loads((root/"continuation-plan.json").read_text(encoding="utf-8"))
    total=Decimal(continuation["total_limit_cny"])
    if total!=Decimal("5") or continuation["predecessor"]!=hold:
        raise ValueError("live_preview_predecessor_mismatch")
    cap=total-Decimal(hold["committed_cny"])
    if Decimal(continuation["remaining_limit_cny"])!=cap:
        raise ValueError("live_preview_allowance_mismatch")
    path=root/"continuation/wallet.sqlite"
    active,manifest=_read_wallet(path)
    if active.max_cost_cny!=cap or active.halted or active.pending_requests or active.unknown_requests:
        raise ValueError("live_preview_active_wallet_not_ready")
    remaining=cap-active.committed_cost_cny
    if remaining<=0:
        raise ValueError("live_preview_wallet_exhausted")
    readonly=SimpleNamespace(storage_path=path,snapshot=lambda:active,manifest_snapshot=lambda:manifest)
    validate_live_plan(root,plan,readonly)
    material=root/plan.material_source/"retrieval/evidence/materials.json"
    pages=json.loads(material.read_text(encoding="utf-8"))["pages"]
    urls=tuple(page["url"] for page in pages if page["url"] in APPROVED_URLS)
    if not urls:
        raise ValueError("live_preview_no_approved_source")
    return LiveUIPlan(plan.main_source,plan.main_sha256,urls,plan.prompt_sha256,plan.model,
        plan.price_sha256,str(remaining),str(settings.get_propagation_harness_config().max_cost_cny))


def execute_live_plan(root: Path, plan: LiveSamplePlan, *, authorized: bool = False) -> str:
    """Open the existing active wallet only after explicit authorization and fresh read-only checks."""
    if not authorized:
        raise ValueError("live_worker_execution_not_authorized")
    root=root.resolve()
    if root!=(settings.PROJECT_ROOT/"traces/research-runs").resolve():
        raise ValueError("live_worker_requires_formal_root")
    preview_live_plan(root,plan)
    from src.yuqing.serve.research_runner import open_research_wallet
    rate=Decimal(os.environ["LIVE_RESEARCH_CNY_PER_CREDIT"])
    wallet=open_research_wallet(root,limit_cny=Decimal("5"),credit_cny=rate)
    try:
        executor=LiveSampleExecutor(root,wallet,plan,external_calls_authorized=True)
        return executor(plan.main_source)
    finally:
        wallet.close()


def main() -> None:
    """Emit only safe structured metadata; errors never print credentials or raw process state."""
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--checklist",type=Path,required=True)
    parser.add_argument("--source-id",required=True)
    parser.add_argument("--action",choices=("preview","execute"),required=True)
    parser.add_argument("--allow-live",action="store_true")
    args=parser.parse_args()
    try:
        if args.action=="execute" and not args.allow_live:
            raise ValueError("live_worker_execution_not_authorized")
        plan=load_live_plan(args.checklist)
        if args.source_id!=plan.main_source:
            raise ValueError("live_worker_source_mismatch")
        result=asdict(preview_live_plan(args.root,plan)) if args.action=="preview" else {
            "research_id":execute_live_plan(args.root,plan,authorized=args.allow_live)}
        sys.stdout.write(json.dumps({"ok":True,"result":result},ensure_ascii=False)+"\n")
    except Exception:
        sys.stdout.write(json.dumps({"ok":False,"error":"live_worker_preflight_or_execution_failed"})+"\n")


if __name__=="__main__":
    main()
