"""Convert a caller-selected video's plain-text top-level comments into W samples."""
from datetime import datetime
from hashlib import sha256
from typing import Literal
from urllib.parse import urlencode

from pydantic import AwareDatetime, BaseModel, Field

from src.yuqing.retrieval.public_material_contracts import PublicDiscussionBatch, PublicObservation


class _Snippet(BaseModel):
    """Read only the identity, publication time and displayed text used by the report."""
    videoId: str
    publishedAt: AwareDatetime
    textDisplay: str = Field(min_length=1, pattern=r"\S")


class _Comment(BaseModel):
    """Validate the comment ID and provider snippet together."""
    id: str = Field(min_length=1, pattern=r"\S")
    snippet: _Snippet


def youtube_comment_batch(rows: list[dict], *, video_id: str, start: datetime, end: datetime,
                         fetched_at: datetime, case_id: str, claim: str,
                         provenance: Literal["synthetic_fixture", "recorded_response"]) -> PublicDiscussionBatch:
    """Bind one page's topLevelComment objects to the selected video and half-open time window."""
    if (any(t.utcoffset() is None for t in (start, end, fetched_at)) or start >= end
            or not video_id.strip() or not claim.strip()):
        raise ValueError("invalid_youtube_window_or_identity")
    seen, observations = set(), []
    context = "https://www.youtube.com/watch?" + urlencode({"v": video_id})
    for row in rows:
        comment = _Comment.model_validate(row)
        snippet = comment.snippet
        if snippet.videoId != video_id:
            raise ValueError("youtube_comment_video_mismatch")
        if not start <= snippet.publishedAt < end or comment.id in seen:
            continue
        seen.add(comment.id)
        observations.append(PublicObservation(material_id=f"W{len(observations)+1}", platform="YouTube",
            source_kind="comment", source_label="YouTube 顶层评论", classification_basis="commentThreads.topLevelComment",
            context_url=context, url=context + "&" + urlencode({"lc": comment.id}), published_at=snippet.publishedAt,
            text=snippet.textDisplay, body_sha256=sha256(snippet.textDisplay.encode()).hexdigest()))
    return PublicDiscussionBatch(case_id=case_id, claim_sha256=sha256(claim.encode()).hexdigest(),
        provenance=provenance, fetched_at=fetched_at, observations=tuple(observations), candidates=(), gaps=())
