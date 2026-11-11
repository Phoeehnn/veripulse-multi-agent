"""Case-bound, offline public-material provider; no model or network execution."""
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl
from src.yuqing.harness.ledger import EvidenceLedger
from src.yuqing.retrieval.public_material_contracts import (
    PublicDiscussionBatch, PublicDeliveryReceipt, ReportedPublicMetrics, PublicMaterialGap)
from src.yuqing.retrieval.public_materials import load_public_material, deliver_public_material, qualify_public_quote


SourceKind = Literal["news_report", "public_post", "comment"]


class ProviderModel(BaseModel):
    """Reject unspecified fields in provider interface models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicMaterialCard(ProviderModel):
    """Discovery metadata without body delivery or comment-tree conversion."""

    material_id: str
    source_kind: SourceKind
    source_url: HttpUrl
    source_label: str
    published_at: AwareDatetime | None
    body_sha256: str
    body_chars: int
    reported_metrics: ReportedPublicMetrics | None


class PublicSearchRequest(ProviderModel):
    """Search this archived batch with explicit type and result bound."""

    query: str = Field(min_length=1, pattern=r"\S")
    limit: int = Field(gt=0, strict=True)
    source_kind: SourceKind | None = None


class PublicReadRequest(ProviderModel):
    """Read an original text prefix; no external URL fetching is permitted."""

    material_id: str = Field(pattern=r"^W[1-9][0-9]*$")
    max_chars: int = Field(gt=0, strict=True)


class PublicQuoteRequest(ProviderModel):
    """Validate a quote against previously returned text in this provider instance."""

    material_id: str = Field(pattern=r"^W[1-9][0-9]*$")
    quote: str = Field(min_length=1, pattern=r"\S")


class PublicSearchResult(ProviderModel):
    """Local discovery results grant no reading or citation qualification."""

    search_scope: Literal["archived_batch_only"] = "archived_batch_only"
    match_method: Literal["casefolded_whitespace_terms_all"] = "casefolded_whitespace_terms_all"
    materials: tuple[PublicMaterialCard, ...]


class PublicReadResult(ProviderModel):
    """One actual provider response with its program-owned delivery receipt."""

    status: Literal["success", "not_found"]
    message: str
    receipt: PublicDeliveryReceipt | None = None


class PublicQuoteResult(ProviderModel):
    """Separate missing delivery from failed original-text matching."""

    status: Literal["passed", "not_found", "not_delivered", "offset_unresolved"]
    material_id: str
    source_kind: SourceKind | None = None
    source_url: HttpUrl | None = None
    start: int | None = None
    end: int | None = None


class PublicProviderSummary(ProviderModel):
    """Describe available samples and gaps without estimating population activity."""

    case_id: str
    provenance: Literal["synthetic_fixture", "recorded_response"]
    material_counts: dict[SourceKind, int]
    counts_meaning: Literal["archived_material_records_not_reach"] = "archived_material_records_not_reach"
    media_status: Literal["available", "unavailable"]
    discussion_status: Literal["available", "unavailable"]
    analysis_scope: Literal["sample_only"] = "sample_only"
    population_spread_status: Literal["unavailable"] = "unavailable"
    population_emotion_status: Literal["unavailable"] = "unavailable"
    gaps: tuple[PublicMaterialGap, ...]


class PublicMaterialProvider:
    """Own one case's independent ledger and read receipts, with no writable shared state."""

    def __init__(self, batch: PublicDiscussionBatch) -> None:
        """Accept validated material and start with no delivered bodies."""
        self._batch = batch
        self._ledger = EvidenceLedger()
        self._receipts: dict[str, PublicDeliveryReceipt] = {}
        self._by_id = {item.material_id: item for item in batch.observations}

    @classmethod
    def open(cls, directory: Path, *, case_id: str, claim: str) -> "PublicMaterialProvider":
        """Validate archive SHA and case/claim before opening a fresh provider instance."""
        return cls(load_public_material(directory, case_id=case_id, claim=claim))

    def describe(self) -> PublicProviderSummary:
        """Report per-type material counts, not numbers of comments across the internet."""
        counts = {kind: sum(item.source_kind == kind for item in self._batch.observations)
                  for kind in ("news_report", "public_post", "comment")}
        return PublicProviderSummary(case_id=self._batch.case_id, provenance=self._batch.provenance,
            material_counts=counts, media_status=self._batch.media_status,
            discussion_status=self._batch.discussion_status, gaps=self._batch.gaps)

    def list_materials(self, source_kind: SourceKind | None = None) -> tuple[PublicMaterialCard, ...]:
        """List discovery cards without exposing bodies or marking evidence read."""
        if source_kind not in {None, "news_report", "public_post", "comment"}:
            raise ValueError("invalid_public_source_kind")
        return tuple(PublicMaterialCard(material_id=item.material_id, source_kind=item.source_kind,
            source_url=item.url, source_label=item.source_label, published_at=item.published_at,
            body_sha256=item.body_sha256, body_chars=len(item.text), reported_metrics=item.metrics)
            for item in self._batch.observations if source_kind is None or item.source_kind == source_kind)

    def search(self, request: PublicSearchRequest) -> PublicSearchResult:
        """Use deterministic AND-substring matching with stable archive order, never web search."""
        terms = request.query.casefold().split()
        cards = self.list_materials(request.source_kind)
        matches = tuple(card for card in cards if all(term in
            (self._by_id[card.material_id].source_label + " " + self._by_id[card.material_id].text).casefold()
            for term in terms))
        return PublicSearchResult(materials=matches[:request.limit])

    def read(self, request: PublicReadRequest) -> PublicReadResult:
        """Return a bounded data block and preserve the longest actual delivered prefix."""
        if request.material_id not in self._by_id:
            return PublicReadResult(status="not_found", message="该批材料中不存在此编号。")
        message, receipts = deliver_public_material(self._batch, self._ledger,
            material_ids=(request.material_id,), max_chars=request.max_chars)
        receipt = receipts[0]
        previous = self._receipts.get(request.material_id)
        if previous is None or len(previous.text) < len(receipt.text):
            self._receipts[request.material_id] = receipt
        return PublicReadResult(status="success", message=message, receipt=receipt)

    def delivered_receipts(self) -> tuple[PublicDeliveryReceipt, ...]:
        """Return immutable receipts for this instance only; metadata search grants none."""
        return tuple(self._receipts.values())

    def qualify(self, request: PublicQuoteRequest) -> PublicQuoteResult:
        """Reuse original prefix and case-bound quote checks without allowing self-reported reads."""
        source = self._by_id.get(request.material_id)
        if source is None:
            return PublicQuoteResult(status="not_found", material_id=request.material_id)
        location = qualify_public_quote(self._batch, request.material_id, request.quote, self.delivered_receipts())
        if location is None:
            status = "not_delivered"
        else:
            status = "passed" if location.status == "matched" else location.status
        return PublicQuoteResult(status=status, material_id=request.material_id,
            source_kind=source.source_kind, source_url=source.url,
            start=location.start if location else None, end=location.end if location else None)
