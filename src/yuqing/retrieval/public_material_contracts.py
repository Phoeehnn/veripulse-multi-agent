"""Public media/discussion material contracts; no claims of population coverage."""
from typing import Literal
from hashlib import sha256
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, computed_field, model_validator
from src.yuqing.retrieval.web_contracts import SearchCandidate


class ReportedPublicMetrics(BaseModel):
    """Preserve explicitly supplied counters as unverified records, never inferred zeros."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_url: HttpUrl
    observed_at: AwareDatetime
    basis: str = Field(min_length=1, pattern=r"\S")
    likes: int | None = Field(default=None, ge=0, strict=True)
    comments: int | None = Field(default=None, ge=0, strict=True)
    reposts: int | None = Field(default=None, ge=0, strict=True)
    views: int | None = Field(default=None, ge=0, strict=True)
    qualification: Literal["unverified"] = "unverified"

    @model_validator(mode="after")
    def require_counter(self) -> "ReportedPublicMetrics":
        """An empty metric record must instead be represented as unavailable None."""
        if all(value is None for value in (self.likes, self.comments, self.reposts, self.views)):
            raise ValueError("public_metrics_no_reported_values")
        return self


class PublicSourceDescriptor(BaseModel):
    """Require explicit caller attribution, keeping reports, posts and comments separate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: HttpUrl
    source_kind: Literal["news_report", "public_post", "comment"]
    classification_basis: str = Field(min_length=1, pattern=r"\S")
    source_label: str = Field(min_length=1, pattern=r"\S")
    context_url: HttpUrl | None = None
    platform: str | None = None
    author: str | None = None
    published_at: AwareDatetime | None = None
    metrics: ReportedPublicMetrics | None = None


class PublicObservation(PublicSourceDescriptor):
    """Bind classified extracted text to a source, fingerprint and independent W identifier."""

    material_id: str = Field(pattern=r"^W[1-9][0-9]*$")
    text: str = Field(min_length=1, pattern=r"\S")
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def verify_body(self) -> "PublicObservation":
        """Reject text that no longer matches its archived extracted-body fingerprint."""
        if sha256(self.text.encode("utf-8")).hexdigest() != self.body_sha256:
            raise ValueError("public_body_sha_mismatch")
        return self


class PublicMaterialGap(BaseModel):
    """Retain extraction failures and missing source classification without manufacturing text."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: HttpUrl
    reason: Literal["unclassified_source", "body_not_extracted", "extraction_failed"]
    provider_reason: str | None = None


class PublicDiscussionBatch(BaseModel):
    """A collection of retrieved samples, never an estimate of the public as a whole."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["public-propagation.v1"] = "public-propagation.v1"
    case_id: str = Field(min_length=1, pattern=r"\S")
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Literal["synthetic_fixture", "recorded_response"]
    fetched_at: AwareDatetime
    observations: tuple[PublicObservation, ...]
    candidates: tuple[SearchCandidate, ...]
    gaps: tuple[PublicMaterialGap, ...]
    analysis_scope: Literal["sample_only"] = "sample_only"
    population_spread_status: Literal["unavailable"] = "unavailable"
    population_emotion_status: Literal["unavailable"] = "unavailable"

    @model_validator(mode="after")
    def require_unique_sources(self) -> "PublicDiscussionBatch":
        """Reject ambiguous material IDs or multiple observations of the same URL in one batch."""
        if (len({item.material_id for item in self.observations}) != len(self.observations)
                or len({item.url for item in self.observations}) != len(self.observations)):
            raise ValueError("public_material_identity_ambiguous")
        return self

    @computed_field
    @property
    def discussion_status(self) -> Literal["available", "unavailable"]:
        """Report availability of discussion samples without treating news as comments."""
        return "available" if any(item.source_kind in {"public_post", "comment"} for item in self.observations) else "unavailable"

    @computed_field
    @property
    def media_status(self) -> Literal["available", "unavailable"]:
        """Report availability of media text independently of discussion samples."""
        return "available" if any(item.source_kind == "news_report" for item in self.observations) else "unavailable"


class PublicDeliveryReceipt(BaseModel):
    """Program-owned actual delivery of one text prefix with original source attribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=1, pattern=r"\S")
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    material_id: str = Field(pattern=r"^W[1-9][0-9]*$")
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    source_kind: Literal["news_report", "public_post", "comment"]
    source_url: HttpUrl
    body_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(min_length=1, pattern=r"\S")
