"""Sample-only public propagation task, report and qualified result contracts."""
from hashlib import sha256
from typing import Literal
from pydantic import Field, model_validator
from src.yuqing.retrieval.public_provider import ProviderModel, SourceKind, PublicProviderSummary
from src.yuqing.retrieval.public_material_contracts import PublicDiscussionBatch, PublicDeliveryReceipt


class PublicPropagationTask(ProviderModel):
    """Bind explicit live task identity to its independently archived public batch."""

    sample_id: str = Field(min_length=1, pattern=r"\S")
    claim: str = Field(min_length=1, pattern=r"\S")
    batch: PublicDiscussionBatch

    @model_validator(mode="after")
    def check_binding(self) -> "PublicPropagationTask":
        """Reject material from another case or another exact claim."""
        if self.batch.case_id != self.sample_id or self.batch.claim_sha256 != sha256(self.claim.encode()).hexdigest():
            raise ValueError("public_task_identity_mismatch")
        return self


class PublicSignal(ProviderModel):
    """One qualitative observation grounded in one delivered original quotation."""

    material_id: str = Field(pattern=r"^W[1-9][0-9]*$")
    source_kind: SourceKind
    quote: str = Field(min_length=1, pattern=r"\S")
    interpretation: str = Field(min_length=1, pattern=r"\S")
    kind: Literal["media_narrative", "public_expression", "verification_lead"]


class PublicSignalReport(ProviderModel):
    """No population risk labels, counts, scores or representativeness claims."""

    analysis_scope: Literal["sample_only"]
    population_spread_status: Literal["unavailable"]
    population_emotion_status: Literal["unavailable"]
    signals: tuple[PublicSignal, ...]
    unresolved_questions: tuple[str, ...]


class PublicSignalResult(ProviderModel):
    """Independent appendix result; deliberately incompatible with legacy risk reports."""

    sample_id: str
    run_id: str
    report: PublicSignalReport
    coverage: PublicProviderSummary
    receipts: tuple[PublicDeliveryReceipt, ...]
    stop_reason: Literal["submitted", "forced_final"]
    steps: int
    input_output_tokens: int
    cost_cny: float
    trace_path: str
