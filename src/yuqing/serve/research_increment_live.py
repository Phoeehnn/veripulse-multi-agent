"""Explicit live sample orchestration, with a separate and honestly labelled validation lane."""
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from src.yuqing import settings
from src.yuqing.llm import OfflineCompletionReplay, use_offline_completion_replay
from src.yuqing.eval.dual_budget import price_fingerprint
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget
from src.yuqing.graph.composition import CaseComposition
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.serve.research_increment import source_case, prepare_increment
from src.yuqing.serve.research_public_activation import run_sample_increment, APPROVED_PROMPT_SHA
from src.yuqing.serve.research_public import PublicSampleSession, project_public_sample
from src.yuqing.serve.research_runner import _case_inputs, save_case_cost
from src.yuqing.serve.research_service import ResearchSample, save_research_snapshot
from src.yuqing.serve.research_contracts import ResearchSnapshot


@dataclass(frozen=True)
class LiveSamplePlan:
    """Pin one reuse source and execution configuration; this object grants no network permission."""

    main_source: str
    material_source: str
    main_sha256: str
    material_sha256: str
    prompt_sha256: str
    model: str
    base_url: str
    thinking: bool
    price_sha256: str


def validate_live_plan(root: Path, plan: LiveSamplePlan, wallet: PersistentExperimentBudget) -> None:
    """Check source bytes, model/prices and same wallet before creating a new attempt."""
    main=source_case(root,plan.main_source)
    material=source_case(root,plan.material_source)
    if sha256((main/"snapshot.json").read_bytes()).hexdigest()!=plan.main_sha256:
        raise ValueError("live_main_sha_mismatch")
    if sha256((material/"retrieval/evidence/materials.json").read_bytes()).hexdigest()!=plan.material_sha256:
        raise ValueError("live_material_sha_mismatch")
    reuse=json.loads((main/"reuse.json").read_text(encoding="utf-8")) if (main/"reuse.json").exists() else {}
    if reuse.get("source_case",main.name)!=plan.material_source:
        raise ValueError("live_material_lineage_mismatch")
    prompt=settings.PROJECT_ROOT/"prompts/propagation/public_materials.live-v1.md"
    if plan.prompt_sha256!=APPROVED_PROMPT_SHA or sha256(prompt.read_bytes()).hexdigest()!=plan.prompt_sha256:
        raise ValueError("live_prompt_sha_mismatch")
    config=settings.get_settings()
    if (config.deepseek_model,config.deepseek_base_url,config.llm_thinking)!=(plan.model,plan.base_url,plan.thinking):
        raise ValueError("live_model_mismatch")
    fingerprint=price_fingerprint(settings.get_cost_config())
    if fingerprint!=plan.price_sha256 or wallet.manifest_snapshot()["price_fingerprint"]!=fingerprint:
        raise ValueError("live_prices_mismatch")
    if wallet.storage_path.resolve()!=(root/"continuation/wallet.sqlite").resolve() or wallet.snapshot().halted:
        raise ValueError("live_wallet_mismatch_or_halted")


def persist_live_attempt(case: Path, base: ResearchSnapshot, session: PublicSampleSession,
                         cost: dict[str, object], *, validation: bool) -> None:
    """Persist actual execution mode, optional failure and per-attempt cost without rescoring."""
    mode="mock" if validation else "live"
    base=base.model_copy(update={"research_id":case.name,"execution_mode":mode,"last_seq":0})
    snapshot=project_public_sample(base,session,sample_id=session.task.sample_id)
    prefix="live编排离线验证" if validation else "真实样本增量"
    snapshot=snapshot.model_copy(update={"status":"completed" if session.status=="completed" else "failed",
        "report":prefix+" · 原事实/仲裁报告复用，本次未重跑\n\n"+(snapshot.report or ""),
        "cost_status":cost["cost_status"],"known_cost_cny":cost.get("known_cost_cny")})
    (case/"cost.json").write_text(json.dumps(cost),encoding="utf-8")
    (case/"sample-result.json").write_text(session.result.model_dump_json() if session.result else "null",encoding="utf-8")
    save_research_snapshot(case,snapshot)


class LiveSampleExecutor:
    """Bind one explicit plan; neither construction nor history reading runs an Agent."""

    def __init__(self, root: Path, wallet: PersistentExperimentBudget, plan: LiveSamplePlan,
                 *, external_calls_authorized: bool = False, validation_replay: OfflineCompletionReplay | None = None) -> None:
        """Reject unapproved real execution or validation against formal archives before any write."""
        self.root=root.resolve()
        if validation_replay is not None:
            allowed=[(settings.PROJECT_ROOT/"traces"/name).resolve() for name in ("tmp","review-packages")]
            if not isinstance(validation_replay,OfflineCompletionReplay) or not any(self.root.is_relative_to(p) for p in allowed):
                raise ValueError("live_validation_requires_isolation_and_replay")
        elif not external_calls_authorized:
            raise ValueError("live_increment_not_authorized")
        elif self.root!=(settings.PROJECT_ROOT/"traces/research-runs").resolve():
            raise ValueError("live_increment_requires_formal_root")
        if self.root.drive.lower()!="d:":
            raise ValueError("live_increment_requires_d_drive")
        self.wallet,self.plan,self.replay=wallet,plan,validation_replay
        self.used=False
        self.program_run_id: str | None = None

    def __call__(self, identity: str) -> str:
        """Consume one attempt only after pinned preflight; archive failures instead of retrying."""
        if self.used or identity!=self.plan.main_source:
            raise ValueError("live_increment_consumed_or_wrong_source")
        validate_live_plan(self.root,self.plan,self.wallet)
        self.used=True
        case,base,task=prepare_increment(self.root,identity)
        before=self.replay.consumed if self.replay else 0
        session=PublicSampleSession(task,lambda:self._sample(task,case))
        stage="live编排离线验证" if self.replay else "真实样本增量（事实/仲裁复用）"
        self._stage(case,stage,"running")
        session.round1({})
        try:
            cost=save_case_cost(case,task.sample_id,self.wallet)
        except Exception:
            cost={"known_cost_cny":None,"cost_status":"unknown","cumulative_known_cny":None,"cumulative_limit_cny":"5"}
        cost["billing_provenance"]="offline_validation_not_bill" if self.replay else "wallet_estimate_not_provider_bill"
        path=case/"sample-increment.json"
        manifest=json.loads(path.read_text(encoding="utf-8"))
        manifest.update(mode="live_sample_validation" if self.replay else "live_sample_increment",
            target_execution_mode="live",offline_validation=self.replay is not None,
            bound_prompt_sha256=self.plan.prompt_sha256,
            execution_prompt_sha256=self.plan.prompt_sha256 if session.result or self.replay and self.replay.consumed>before else None,
            model=self.plan.model,base_url=self.plan.base_url,thinking=self.plan.thinking,
            price_fingerprint=self.plan.price_sha256,
            program_run_id=self.program_run_id,sample_id=task.sample_id,
            replay_responses=self.replay.consumed-before if self.replay else None)
        path.write_text(json.dumps(manifest),encoding="utf-8")
        persist_live_attempt(case,base,session,cost,validation=self.replay is not None)
        self._stage(case,stage,session.status)
        return case.name

    def _sample(self, task: object, case: Path) -> object:
        """Use the original sample loop with the same wallet and fresh program run identity."""
        fact,prop=EvidenceWorkerScope(),EvidenceWorkerScope()
        try:
            inputs=_case_inputs(case,ResearchSample(sample_id=task.sample_id,claim=task.claim),self.wallet,
                                case/"unused-index",fact,prop)
            composition=CaseComposition(replace(inputs,coverage="none",initial_manifest_path=None))
            lane=use_offline_completion_replay(self.replay) if self.replay else nullcontext()
            with lane:
                result=run_sample_increment(task,prompt=settings.PROJECT_ROOT/"prompts/propagation/public_materials.live-v1.md",
                    composition=composition,root=self.root,external_calls_authorized=True)
            self.program_run_id=next(iter(composition.runs),None)
            return result
        finally:
            fact.close()
            prop.close()

    def _stage(self, case: Path, stage: str, status: str) -> None:
        """Record only observed task lifecycle, with live and validation labels distinct."""
        with (case/"stages.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({"time":datetime.now(timezone.utc).isoformat(),"stage":stage,"status":status},ensure_ascii=False)+"\n")
