"""Server-owned offline query selection. Browser values are IDs, never material bodies or paths."""
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import re
import subprocess
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4
from typing import Callable


def pinned_youtube_fixture(project: Path) -> dict:
    """Read the reviewed replay bytes with portable checkout line endings."""
    payload = (project / "docs/cards/LR-youtube-comment-replay.json").read_bytes().replace(bytes([13, 10]), bytes([10]))
    if sha256(payload).hexdigest() != "23e976ebf34b33755003731b5e536a1339681b7d792edeab6c75002808a0847c":
        raise ValueError("youtube_fixture_identity_mismatch")
    return json.loads(payload)


def read_query(case: Path, query_id: str, pin: str, claim: str) -> dict:
    """Check archived bytes, original claim and case before returning selected materials."""
    if not re.fullmatch(r"[0-9a-f]{32}", query_id):
        raise ValueError("invalid_query_id")
    payload = (case / "queries" / (query_id + ".json")).read_bytes()
    if sha256(payload).hexdigest() != pin:
        raise ValueError("youtube_query_drift")
    row = json.loads(payload)
    request = json.loads((case / "request.json").read_bytes())
    if (row["case_id"] != case.name or row["query_id"] != query_id or request["input_value"] != claim
            or row["claim_sha256"] != sha256(claim.encode()).hexdigest()):
        raise ValueError("youtube_query_case_mismatch")
    return row


def observation_window(start: str, end: str) -> tuple[datetime, datetime]:
    """Use inclusive Beijing calendar dates and return a half-open interval."""
    zone = timezone(timedelta(hours=8))
    first = datetime.combine(date.fromisoformat(start), time.min, zone)
    after = datetime.combine(date.fromisoformat(end), time.min, zone) + timedelta(days=1)
    if first >= after:
        raise ValueError("invalid_query_window")
    return first, after


@dataclass
class YoutubeCaseReplay:
    """Own fixed-news replay cases and immutable query fingerprints for one UI instance."""
    root: Path
    project: Path
    page_lookup: Callable[[str, str, str], dict] | None = None
    cases: dict = field(default_factory=dict)
    queries: dict = field(default_factory=dict)
    selected: dict = field(default_factory=dict)
    started: set = field(default_factory=set)

    def create(self, claim: str) -> str:
        """Create one isolated case only for the fixed offline retirement story."""
        expected = json.loads((self.project / "docs/cards/LR-news-retirement-fixture.json").read_bytes())["claim"]
        sandbox = (self.project / "traces/tmp").resolve()
        if self.root.resolve().drive.lower() != "d:" or not self.root.resolve().is_relative_to(sandbox):
            raise ValueError("offline_case_requires_sandbox")
        if claim != expected:
            raise ValueError("offline_case_requires_fixed_claim")
        identity = uuid4().hex
        case = self.root / identity / identity
        case.mkdir(parents=True)
        payload = json.dumps(dict(input_kind="text", input_value=claim), ensure_ascii=False).encode()
        (case / "request.json").write_bytes(payload)
        self.cases[identity] = (case, sha256(payload).hexdigest(), claim)
        return identity

    def _case(self, identity: str) -> tuple[Path, str]:
        """Reject a changed request before reading a query or starting a worker."""
        case, digest, claim = self.cases[identity]
        if sha256((case / "request.json").read_bytes()).hexdigest() != digest:
            raise ValueError("youtube_case_request_drift")
        if identity in self.started:
            raise ValueError("offline_case_already_started")
        return case, claim

    def query(self, identity: str, keyword: str, start: str, end: str) -> dict:
        """Archive one pinned page with separate search words and original case identity."""
        self.selected.pop(identity, None)
        case, claim = self._case(identity)
        if not keyword.strip() or len(keyword) > 300:
            raise ValueError("invalid_keyword")
        first, after = observation_window(start, end)
        source = pinned_youtube_fixture(self.project)
        receipt = source["receipt"]
        ids = parse_qs(urlsplit(receipt["source_url"]).query)
        query_id = uuid4().hex
        row = dict(case_id=identity, query_id=query_id, claim_sha256=sha256(claim.encode()).hexdigest(),
            keyword=keyword, start=first.isoformat(), end=after.isoformat(),
            fetched_at=datetime.now(timezone.utc).isoformat(), video_id=ids["v"][0],
            video_title="延迟退休（固定视频评论回放，主题相关性请核对）",
            video_url="https://www.youtube.com/watch?v="+ids["v"][0], returned_count=1,
            comments=[dict(id=ids["lc"][0], snippet=dict(videoId=ids["v"][0],
                publishedAt=source["published_at"], textDisplay=receipt["text"]))])
        row["outcome"] = "available"
        if self.page_lookup is not None:
            try:
                response = self.page_lookup(keyword, start, end)
            except (OSError, ValueError):
                response = dict(outcome="query_failed", comments=[], video_id="unavailable",
                                fetched_at=datetime.now(timezone.utc).isoformat(), requests=None)
            row.update(video_title="", video_url="", comments=[])
            if not {"video_id", "fetched_at", "comments", "outcome"} <= response.keys():
                raise ValueError("incomplete_youtube_receipt")
            for key in ("video_id", "video_title", "video_url", "fetched_at", "comments", "outcome", "requests"):
                if key in response:
                    row[key] = response[key]
            if row["outcome"] in {"query_failed", "no_video"}:
                row.update(comments=[], video_title="未取得候选视频或评论", video_url="")
        return self._archive_query(case, row, start, end)

    def _archive_query(self, case: Path, row: dict, start: str, end: str) -> dict:
        """Validate provider comments once and persist their case-bound immutable receipt."""
        from src.yuqing.retrieval.youtube_comments import youtube_comment_batch
        first, after = observation_window(start, end)
        _, claim = self._case(case.name)
        batch = youtube_comment_batch(row["comments"], video_id=row["video_id"], start=first, end=after,
            fetched_at=datetime.fromisoformat(row["fetched_at"]), case_id=case.name, claim=claim,
            provenance="synthetic_fixture")
        row["returned_count"], row["window_count"] = len(row["comments"]), len(batch.observations)
        if row["outcome"] == "available" and not batch.observations:
            row["outcome"] = "outside_window" if row["comments"] else "empty_page"
        query_id, identity = row["query_id"], case.name
        payload = json.dumps(row, ensure_ascii=False).encode()
        folder = case / "queries"
        folder.mkdir(exist_ok=True)
        (folder / (query_id + ".json")).write_bytes(payload)
        pin = sha256(payload).hexdigest()
        with (case / "query-manifest.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(query_id=query_id, sha256=pin, case_id=identity)) + chr(10))
        self.queries[query_id] = (identity, pin)
        return dict(query_id=query_id, video_title=row["video_title"], video_url=row["video_url"],
                    returned_count=row["returned_count"], window_count=row["window_count"],
                    outcome=row["outcome"], start=start, end=end)

    def select(self, identity: str, query_id: str, use: bool) -> str:
        """Reload the same server-owned page and record explicit selection or skip."""
        self.selected.pop(identity, None)
        case, claim = self._case(identity)
        owner, pin = self.queries[query_id]
        if owner != identity:
            raise ValueError("youtube_query_case_mismatch")
        row = read_query(case, query_id, pin, claim)
        if use and row.get("outcome") in {"query_failed", "no_video"}:
            raise ValueError("youtube_unavailable_requires_explicit_skip")
        self.selected[identity] = (query_id, pin, use)
        return "已选用于本案" if use else "已跳过本案评论样本"

    def generate(self, identity: str) -> dict:
        """Validate selection before starting the real, network-blocked replay worker once."""
        case, claim = self._case(identity)
        query_id, pin, use = self.selected[identity]
        read_query(case, query_id, pin, claim)
        env = dict(os.environ, PYTHONPATH=str(self.project), PYTHONIOENCODING="utf-8",
            PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            LR_SYNTHETIC_ACCEPTANCE=sha256(claim.encode()).hexdigest(),
            LR_SELECTED_YOUTUBE_QUERY=query_id, LR_SELECTED_YOUTUBE_SHA=pin,
            LR_SELECTED_YOUTUBE_USE="1" if use else "0", TEMP=str(case.parent), TMP=str(case.parent))
        env.pop("LR_OFFLINE_YOUTUBE_WINDOW", None)
        self.started.add(identity)
        (case / "selection.json").write_text(json.dumps(dict(query_id=query_id, sha256=pin, use=use)))
        with (case / "worker.log").open("w", encoding="utf8") as log:
            result = subprocess.run([sys.executable, "-B",
                "-m", "scripts.run_research_worker", "--synthetic-acceptance",
                "--case", str(case), "--root", str(case.parent)], cwd=self.project, env=env,
                stdout=log, stderr=log, timeout=120, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise ValueError("offline_selected_worker_failed")
        return json.loads((case / "snapshot.json").read_bytes())
