"""Strictly offline integration of sample activation, wallet receipts and archived increments."""
from collections.abc import Callable
from pathlib import Path
from dataclasses import replace
import json
from src.yuqing.llm import OfflineCompletionReplay, use_offline_completion_replay
from src.yuqing.settings import PROJECT_ROOT
from src.yuqing.serve.research_increment import archive_sample_increment
from src.yuqing.serve.research_public_activation import run_sample_increment, APPROVED_PROMPT_SHA
from src.yuqing.serve.research_runner import _case_inputs, save_case_cost
from src.yuqing.serve.research_service import ResearchSample
from src.yuqing.graph.composition import CaseComposition
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask, PublicSignalResult
from src.yuqing.eval.persistent_experiment_budget import PersistentExperimentBudget


def offline_increment_runner(root: Path, wallet: PersistentExperimentBudget,
                             replay: OfflineCompletionReplay) -> Callable[[str], str]:
    """Bind only an isolated D-drive archive and mandatory no-network completion replay."""
    root=root.resolve()
    allowed=[(PROJECT_ROOT/"traces"/name).resolve() for name in ("tmp","review-packages")]
    if root.drive.lower() != "d:" or not any(root.is_relative_to(parent) for parent in allowed):
        raise ValueError("offline_increment_requires_isolated_d_root")
    if not isinstance(replay,OfflineCompletionReplay):
        raise ValueError("offline_increment_requires_replay")
    if wallet.storage_path.resolve() != (root/"continuation/wallet.sqlite").resolve():
        raise ValueError("offline_increment_wallet_mismatch")
    cases: dict[str, Path]={}

    def job(task: PublicPropagationTask, case: Path) -> PublicSignalResult | None:
        """Execute the existing propagation loop only, with both budgets and forced replay."""
        cases[task.sample_id]=case
        fact,prop=EvidenceWorkerScope(),EvidenceWorkerScope()
        try:
            inputs=_case_inputs(case,ResearchSample(sample_id=task.sample_id,claim=task.claim),wallet,
                                case/"unused-index",fact,prop)
            composition=CaseComposition(replace(inputs,coverage="none",initial_manifest_path=None))
            before=replay.consumed
            with use_offline_completion_replay(replay):
                result=run_sample_increment(task,prompt=PROJECT_ROOT/"prompts/propagation/public_materials.live-v1.md",
                    composition=composition,root=root,external_calls_authorized=True)
            path=case/"sample-increment.json"
            manifest=json.loads(path.read_text(encoding="utf-8"))
            manifest.update(execution_prompt_sha256=APPROVED_PROMPT_SHA if replay.consumed>before else None,
                            model_transport="offline_completion_replay",new_model_responses=replay.consumed-before)
            path.write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
            return result
        finally:
            fact.close()
            prop.close()

    def cost(sample_id: str) -> dict[str, object]:
        """Derive per-attempt costs from the same simulated wallet, preserving unknown holds."""
        data=save_case_cost(cases[sample_id],sample_id,wallet)
        return dict(data,billing_provenance="offline_replay_not_provider_bill")

    def execute(identity: str) -> str:
        """Archive one explicit offline increment; history reads never call this closure."""
        return archive_sample_increment(root,identity,job=job,cost_receipt=cost)
    return execute
