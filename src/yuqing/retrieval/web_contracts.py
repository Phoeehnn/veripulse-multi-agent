"""Offline webpage response contracts; these objects grant no network permission."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl


class SearchCandidate(BaseModel):
    """A search snippet is a lead, never delivered evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: HttpUrl
    title: str
    snippet: str


class ExtractedPage(BaseModel):
    """Preserve provider-extracted text without claiming original HTML fidelity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: HttpUrl
    text: str = Field(min_length=1, pattern=r"\S")


class ExtractionFailure(BaseModel):
    """Retain a failed URL and its provider explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    url: HttpUrl
    reason: str


class WebMaterialBatch(BaseModel):
    """Validated provider material, explicitly labelled by fixture provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    provenance: Literal["synthetic_fixture", "recorded_response"]
    fetched_at: AwareDatetime
    query: str = Field(min_length=1, pattern=r"\S")
    candidates: tuple[SearchCandidate, ...]
    pages: tuple[ExtractedPage, ...]
    failures: tuple[ExtractionFailure, ...]
