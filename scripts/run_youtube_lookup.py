"""Explicit on-click public YouTube sample lookup in the core environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import date, datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from typing import Callable
from uuid import uuid4

from dotenv import dotenv_values
from pydantic import SecretStr
from src.yuqing.llm import youtube_public_get
from src.yuqing.retrieval.youtube_comments import youtube_comment_batch


def window_dates(start: str, end: str) -> tuple[datetime, datetime]:
    """Use inclusive calendar dates in Beijing, internally represented as a half-open interval."""
    zone = timezone(timedelta(hours=8))
    first = datetime.combine(date.fromisoformat(start), time.min, zone)
    after = datetime.combine(date.fromisoformat(end) + timedelta(days=1), time.min, zone)
    if first >= after:
        raise ValueError("结束日须等于或晚于开始日")
    return first, after


def lookup(query: str, start: str, end: str, get: Callable[[str, dict], dict]) -> dict:
    """List at most one video's first comment page and identify the actual sample denominator."""
    first, after = window_dates(start, end)
    if not query.strip() or len(query) > 300:
        raise ValueError("请输入 1–300 字的新闻主题或关键词")
    found = get("search", dict(part="snippet", type="video", q=query, maxResults="1", relevanceLanguage="zh"))
    result = dict(query=query, start=start, end=end, requests=1, samples=[], comments=[],
                  fetched_at=datetime.now(timezone.utc).isoformat(), video_id="unavailable", outcome="no_video")
    if not found["items"]:
        return dict(result, status="未找到候选视频")
    video = found["items"][0]
    vid = video["id"]["videoId"]
    result.update(video_id=vid, video_title=video["snippet"]["title"], video_url="https://www.youtube.com/watch?v="+vid, requests=2)
    try:
        page = get("commentThreads", dict(part="snippet", videoId=vid, maxResults="20",
                                          order="relevance", textFormat="plainText"))
    except (ValueError, OSError) as exc:
        return dict(result, status="评论页读取失败", outcome="query_failed", detail=type(exc).__name__)
    result["fetched_at"] = datetime.now(timezone.utc).isoformat()
    rows = [item["snippet"]["topLevelComment"] for item in page["items"]]
    batch = youtube_comment_batch(rows, video_id=vid, start=first, end=after, fetched_at=datetime.now(timezone.utc),
        case_id=uuid4().hex, claim=query, provenance="recorded_response")
    result.update(page_count=len(rows), comments=rows, outcome=("available" if batch.observations else
                  "outside_window" if rows else "empty_page"), samples=[dict(id=o.material_id, published_at=o.published_at.isoformat(),
        text=o.text, url=str(o.url)) for o in batch.observations])
    return dict(result, status="候选视频评论样本（主题相关性待核对）" if result["samples"] else "本页没有符合时间窗的评论")


def main() -> None:
    """Read inputs from stdin and load only the required credential inside the worker."""
    try:
        value = os.environ.get("YOUTUBE_API_KEY") or dotenv_values(Path(os.environ.get("PROJECTA_ENV_FILE", str(Path(__file__).resolve().parents[1] / ".env"))), interpolate=False).get("YOUTUBE_API_KEY")
        if not value:
            raise ValueError("缺少 YOUTUBE_API_KEY")
        key = SecretStr(value)
        params = json.load(sys.stdin)
        result = lookup(**params, get=lambda op, args: youtube_public_get(op, args, key))
    except ValueError as exc:
        result = dict(status=str(exc) if type(exc) is ValueError else "响应格式无效")
    except (OSError, KeyError, TypeError, RuntimeError):
        result = dict(status="查询失败")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
