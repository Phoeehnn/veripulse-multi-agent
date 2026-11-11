"""Bridge offline webpage materials to the existing evidence delivery contract."""
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from src.yuqing.harness.initial_evidence import InitialEvidenceItemV2, InitialEvidenceManifestV2
from src.yuqing.retrieval.web_contracts import WebMaterialBatch
from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_harness_config


def parse_web_responses(search: dict[str, Any], extract: dict[str, Any], *,
                        query: str, fetched_at: str, provenance: str) -> WebMaterialBatch:
    """Validate Tavily-shaped offline responses without an SDK or HTTP client."""
    return WebMaterialBatch.model_validate({
        "provenance": provenance, "fetched_at": fetched_at, "query": query,
        "candidates": [{"url": row["url"], "title": row["title"],
                        "snippet": row["content"]} for row in search["results"]],
        "pages": [{"url": row["url"], "text": row["raw_content"]}
                  for row in extract["results"]],
        "failures": [{"url": row["url"], "reason": row["error"]}
                     for row in extract["failed_results"]],
    })


def _digest(value: object) -> str:
    """Hash canonical JSON for independently reproducible material fingerprints."""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def archive_web_material(batch: WebMaterialBatch, destination: Path, *, sample_id: str,
                         claim: str, chunk_chars: int) -> InitialEvidenceManifestV2:
    """Create a new per-investigation archive; never overwrite existing runs."""
    if chunk_chars <= 0 or not claim.strip() or not sample_id.strip():
        raise ValueError("invalid_web_material_identity_or_chunk_size")
    chunks, sources = _page_chunks(batch, chunk_chars)
    entries = _initial_entries(chunks)
    manifest = InitialEvidenceManifestV2(
        manifest_version="fact-v2-initial-v1", sample_id=sample_id,
        claim_sha256=sha256(claim.encode("utf-8")).hexdigest(),
        retrieval_sha256=_digest(batch.model_dump(mode="json")),
        corpus_sha256=_digest([chunk.model_dump() for chunk in chunks]),
        limits_sha256=_digest({"chunk_chars": chunk_chars,
                              "search_k": get_harness_config().search_k}),
        entries_sha256=_digest([entry.model_dump(mode="json") for entry in entries]),
        retrieval_status="success" if entries else "empty",
        original_candidate_count=len(entries), delivered_count=len(entries), entries=entries)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "materials.json").write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    (destination / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "chunks.json").write_text(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False), encoding="utf-8")
    (destination / "initial-manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _page_chunks(batch: WebMaterialBatch, chunk_chars: int
                 ) -> tuple[list[EvidenceChunk], list[dict[str, object]]]:
    """Slice extracted text exactly, retaining offsets and whole-body SHA provenance."""
    chunks: list[EvidenceChunk] = []
    sources: list[dict[str, object]] = []
    for number, page in enumerate(batch.pages):
        body_sha = sha256(page.text.encode("utf-8")).hexdigest()
        for start in range(0, len(page.text), chunk_chars):
            text = page.text[start:start + chunk_chars]
            if not text.strip():
                continue
            chunk_id = f"web-{number}-{start}"
            chunks.append(EvidenceChunk(chunk_id=chunk_id, text=text,
                                        source_doc=str(page.url), position=start))
            sources.append({"chunk_id": chunk_id, "url": str(page.url),
                            "body_sha256": body_sha, "start": start, "end": start + len(text),
                            "fetched_at": batch.fetched_at.isoformat(), "text_kind": "extracted_text"})
    return chunks, sources


def _initial_entries(chunks: list[EvidenceChunk]) -> list[InitialEvidenceItemV2]:
    """Select bounded initial text candidates and retain all identical-text sources."""
    grouped: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.text, []).append(chunk)
    entries = []
    for rank, (text, sources) in enumerate(grouped.items(), start=1):
        if len(entries) >= get_harness_config().search_k:
            break
        first = sources[0]
        entries.append(InitialEvidenceItemV2(
            chunk_id=first.chunk_id, source_doc=first.source_doc, position=first.position,
            sources=sources, text=text, text_sha256=sha256(text.encode("utf-8")).hexdigest(),
            initial_rank=rank, read_start=0, read_end=len(text)))
    return entries
