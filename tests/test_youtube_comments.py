"""Synthetic adaptation of the original YouTube window and identity checks."""
from datetime import datetime, timezone
import pytest
from src.yuqing.retrieval.youtube_comments import youtube_comment_batch


def test_window_identity_deduplication_and_original_link():
    """Keep inclusive start, exclude exclusive end, deduplicate IDs and bind the video."""
    start = datetime(2024, 9, 13, tzinfo=timezone.utc)
    end = datetime(2024, 9, 14, tzinfo=timezone.utc)
    def row(identity, stamp):
        """Build a comment with no real user or public data."""
        return dict(id=identity, snippet=dict(videoId="synthetic-video", publishedAt=stamp,
                                             textDisplay="合成评论：希望开放时间更清晰。"))
    first = row("synthetic-comment", "2024-09-13T00:00:00Z")
    rows = [row("older", "2024-09-12T23:59:59Z"), first, first,
            row("end", "2024-09-14T00:00:00Z")]
    options = dict(video_id="synthetic-video", start=start, end=end, fetched_at=end,
                   case_id="synthetic-case", claim="合成声明", provenance="synthetic_fixture")
    batch = youtube_comment_batch(rows, **options)
    assert len(batch.observations) == 1
    assert batch.observations[0].material_id == "W1"
    assert str(batch.observations[0].url).endswith("lc=synthetic-comment")
    assert batch.observations[0].published_at == start
    with pytest.raises(ValueError, match="video_mismatch"):
        youtube_comment_batch([first], **dict(options, video_id="another-video"))
    with pytest.raises(ValueError, match="invalid_youtube_window"):
        youtube_comment_batch([first], **dict(options, end=start))
