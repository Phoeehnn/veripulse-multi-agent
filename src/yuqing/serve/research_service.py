"""Bind archived webpage evidence to the existing case graph without changing core Agents."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, Callable

from pydantic import TypeAdapter
from src.yuqing.graph.composition import CaseComposition, CaseInputs
from src.yuqing.graph.fact_adapter import FactNodeAdapter
from src.yuqing.graph.runtime import (CaseRuntimeAssembly, CaseServiceBindings,
    assemble_bound_case_runtime, bind_fact_node_handlers, bind_arbiter_v2_handler)
from src.yuqing.harness.initial_evidence import load_initial_evidence_manifest
from src.yuqing.retrieval.bm25 import build_bm25_index
from src.yuqing.schemas import EvidenceChunk, Sample
from src.yuqing.serve.research_contracts import ResearchRequest, ResearchSnapshot, ResearchAgent, ResearchCitation
from src.yuqing.settings import get_graph_config
from src.yuqing.serve.research_public import PublicSampleSession, project_public_sample


class ResearchSample(Sample):
    """Use an explicit live-news origin without relabelling it as a benchmark corpus."""

    source: Literal["live_research"] = "live_research"


def load_research_chunks(directory: Path, sample_id: str, claim: str) -> list[EvidenceChunk]:
    """Verify archived corpus identity and initial deliveries before graph assembly."""
    manifest = load_initial_evidence_manifest(directory / "initial-manifest.json", sample_id, claim)
    chunks = TypeAdapter(list[EvidenceChunk]).validate_json((directory / "chunks.json").read_bytes())
    payload = json.dumps([chunk.model_dump() for chunk in chunks], ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    if sha256(payload.encode("utf-8")).hexdigest() != manifest.corpus_sha256:
        raise ValueError("research_corpus_sha_mismatch")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("research_duplicate_chunk_id")
    for entry in manifest.entries:
        if any(source != by_id.get(source.chunk_id) for source in entry.sources):
            raise ValueError("research_initial_source_mismatch")
    return chunks


class ResearchComposition(CaseComposition):
    """Keep the optional sample side task outside legacy Graph scoring context."""

    def __init__(self, inputs: CaseInputs, public_sample: PublicSampleSession | None) -> None:
        """Bind only a matching sample/claim; a missing side task performs no work."""
        super().__init__(inputs)
        if public_sample and (public_sample.task.sample_id != inputs.sample.sample_id
                              or public_sample.task.claim != inputs.sample.claim):
            raise ValueError("public_service_task_mismatch")
        self.public_sample = public_sample


def assemble_research_case(request: ResearchRequest, inputs: CaseInputs, directory: Path, *,
                            external_calls_authorized: bool = False, public_sample: PublicSampleSession | None = None,
                            sample_bind: Callable | None = None, fact_guard: Callable | None = None,
                            offline_bind: Callable[[CaseServiceBindings, ResearchComposition], CaseServiceBindings] | None = None
                            ) -> tuple[CaseRuntimeAssembly, CaseComposition]:
    """Use real fact/arbiter services and skip propagation when no propagation data exists."""
    if not external_calls_authorized:
        raise ValueError("research_external_calls_not_authorized")
    if (not isinstance(inputs.sample, ResearchSample) or inputs.sample.gold_veracity is not None
            or inputs.sample.gold_risk is not None):
        raise ValueError("research_requires_unlabelled_live_sample")
    if request.input_kind != "text" or request.input_value != inputs.sample.claim:
        raise ValueError("research_claim_mismatch_or_url_not_resolved")
    if inputs.sample.propagation is not None or inputs.m6b or inputs.m6c:
        raise ValueError("research_first_version_requires_no_propagation")
    chunks = load_research_chunks(directory, inputs.sample.sample_id, inputs.sample.claim)
    manifest = load_initial_evidence_manifest(directory / "initial-manifest.json",
                                             inputs.sample.sample_id, inputs.sample.claim)
    bound_inputs = replace(inputs, index=build_bm25_index(chunks),
        initial_manifest_path=directory / "initial-manifest.json",
        coverage="full" if manifest.entries else "none")
    composition = ResearchComposition(bound_inputs, public_sample)
    sample_handler = sample_bind(composition) if sample_bind else None
    adapter = FactNodeAdapter(get_graph_config().fact_node_version)
    fact, followup = bind_fact_node_handlers(adapter, composition.context, composition.fact_call,
        composition.followup_call, lambda: bound_inputs.wallet)
    if fact_guard is not None:
        fact, followup = fact_guard(fact, composition), fact_guard(followup, composition)
    bindings = CaseServiceBindings(claim=bound_inputs.sample.claim, comments=(),
        coverage=bound_inputs.coverage, fact_round1=fact, propagation_round1=sample_handler or (public_sample.round1 if public_sample else _no_propagation),
        arbitrate=bind_arbiter_v2_handler(composition.arbiter_call, composition.context),
        fact_followup=followup, action_snapshot=composition.action_snapshot,
        references=composition.references, score_inputs=composition.score_inputs)
    if offline_bind is not None:
        bindings = offline_bind(bindings, composition)
    state = {"case_id": bound_inputs.case_id, "sample": bound_inputs.sample.model_dump(mode="json"),
             "ticket": {"claim": bound_inputs.sample.claim}}
    assembly = assemble_bound_case_runtime(state, bindings, config=get_graph_config(),
        eval_mode=True, context=composition.context, external_calls_authorized=True)
    return assembly, composition


def _no_propagation(_state: dict[str, object]) -> dict[str, object]:
    """Supply no propagation report, statistics or model call for unavailable data."""
    return {}


def research_snapshot(output: dict[str, object], composition: CaseComposition, *,
                       execution_mode: Literal["mock", "live"]) -> ResearchSnapshot:
    """Project observed graph results; missing propagation remains explicitly unavailable."""
    snapshots = composition.context.fact_snapshots()
    citations = []
    for snapshot in snapshots:
        delivered = {entry.evidence_id: entry for entry in snapshot.delivered}
        for quote in snapshot.validation.quote_records:
            entry = delivered[quote.evidence_id]
            sources = TypeAdapter(list[EvidenceChunk]).validate_json(entry.sources_json)
            for number, source in enumerate(sources):
                citations.append(ResearchCitation(
                    citation_id=f"{snapshot.scope.run_id}:{quote.fact_index}:{quote.quote_index}:{number}",
                    run_id=snapshot.scope.run_id, evidence_id=quote.evidence_id,
                    quote=quote.original_quote, source_url=source.source_doc,
                    source_label=source.source_doc, body_sha256=quote.body_sha256, validation="passed"))
    decisions = output.get("decisions") or []
    failed = bool(output.get("stop_reason"))
    unresolved = tuple(str(row.get("text", row.get("issue_id", "未解决问题")))
        for row in output.get("issues", []) if row.get("epistemic") in {"unknown", "conflicting"})
    snapshot = ResearchSnapshot(research_id=composition.inputs.case_id, execution_mode=execution_mode,
        status="failed" if failed else "completed", report=output.get("brief"),
        citations=tuple(citations), unresolved=unresolved, propagation_status="unavailable",
        agents=(ResearchAgent(role="fact", status="unavailable" if composition.inputs.coverage == "none"
                    else "completed" if snapshots else "failed"),
                ResearchAgent(role="propagation", status="unavailable", reason="no_reliable_propagation_data"),
                ResearchAgent(role="arbiter", status="completed" if decisions else
                    "failed" if any(run.role == "decision" for run in composition.runs.values()) else "not_started")),
        cost_status="unknown")

    return project_public_sample(snapshot, getattr(composition, "public_sample", None),
                                 sample_id=composition.inputs.sample.sample_id)


def save_research_snapshot(case: Path, snapshot: ResearchSnapshot) -> None:
    """Save the public snapshot and snapshot-derived business events without executing a job."""
    from datetime import datetime, timezone
    from src.yuqing.serve.research_events import public_snapshot_events

    events = public_snapshot_events(snapshot, occurred_at=datetime.now(timezone.utc), source_ref="pending")
    saved = snapshot.model_copy(update={"last_seq": events[-1].seq}) if events else snapshot
    payload = saved.model_dump_json(indent=2).encode("utf-8")
    reference = "snapshot-sha256:" + sha256(payload).hexdigest()
    (case / "snapshot.json").write_bytes(payload)
    (case / "report.md").write_text(saved.report or "未产生合格报告", encoding="utf-8")
    (case / "business-events.jsonl").write_text("".join(
        event.model_copy(update={"source_event_ref": reference}).model_dump_json() + "\n" for event in events), encoding="utf-8")
