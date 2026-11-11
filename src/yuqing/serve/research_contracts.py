"""Stable public Live Research contracts, independent of Agent runtime internals."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ResearchModel(BaseModel):
    """Reject unknown public fields rather than accepting internal runtime state."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["live-research.v1"] = "live-research.v1"


class ResearchRequest(ResearchModel):
    """Describe a research input without granting network or spending permission."""

    input_kind: Literal["text", "url"]
    input_value: str = Field(min_length=1, pattern=r"\S")


class ResearchAgent(ResearchModel):
    """Expose observed role status without internal budgets or loop counters."""

    role: Literal["fact", "propagation", "arbiter"]
    status: Literal["not_started", "running", "completed", "failed", "interrupted", "unavailable"]
    reason: str | None = None


class ResearchCitation(ResearchModel):
    """Keep original quotation, provenance and qualification distinct."""

    citation_id: str
    run_id: str
    evidence_id: str
    quote: str
    source_url: str | None = None
    source_label: str
    body_sha256: str | None = None
    validation: Literal["passed", "failed", "unknown"] = "unknown"


class ResearchSnapshot(ResearchModel):
    """Present replay, Mock and live outcomes through one public projection."""

    research_id: str
    execution_mode: Literal["historical_replay", "mock", "live"]
    status: Literal["queued", "running", "awaiting_approval", "completed", "failed", "interrupted"]
    agents: tuple[ResearchAgent, ...] = ()
    report: str | None = None
    citations: tuple[ResearchCitation, ...] = ()
    unresolved: tuple[str, ...] = ()
    propagation_status: Literal["available", "unavailable", "unknown"] = "unknown"
    known_cost_cny: str | None = None
    cost_status: Literal["known", "partial", "unknown"] = "unknown"
    elapsed_ms: int | None = Field(default=None, ge=0)
    last_seq: int = Field(default=0, ge=0)
    trace_ref: str | None = None


class ResearchEvent(ResearchModel):
    """Carry an observed event with explicit archival provenance."""

    research_id: str
    seq: int = Field(ge=1)
    execution_mode: Literal["historical_replay", "mock", "live"]
    occurred_at: datetime
    kind: Literal["node_status", "tool_call", "citation_validation", "report", "outcome"]
    actor: Literal["fact", "propagation", "arbiter", "program"]
    status: str
    summary: str
    run_id: str | None = None
    source_event_ref: str
