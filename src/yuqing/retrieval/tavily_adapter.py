"""Small search/extract orchestration; HTTP remains exclusively in llm.py."""
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ValidationError

from src.yuqing.harness.initial_evidence import InitialEvidenceManifestV2
from src.yuqing.retrieval.web_contracts import SearchCandidate, WebMaterialBatch
from src.yuqing.retrieval.web_evidence import archive_web_material, parse_web_responses
from src.yuqing.retrieval.web_provider_contracts import WebRetrievalRequest, WebRetrievalReceipt


class WebResearchMaterialResult(BaseModel):
    """Separate retrieval failure from valid but empty material."""

    status: Literal["prepared", "empty", "retrieval_failed", "invalid_material"]
    manifest: InitialEvidenceManifestV2 | None = None
    receipts: tuple[WebRetrievalReceipt, ...]


def _invoke_archived(request: WebRetrievalRequest, destination: Path,
                      invoke: Callable[[WebRetrievalRequest], WebRetrievalReceipt]) -> WebRetrievalReceipt:
    """Record sanitized invocation receipts in an already isolated run directory."""
    receipt = invoke(request)
    (destination / f"{request.operation}-receipt.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return receipt


def run_tavily_materials(search_request: WebRetrievalRequest, *, extract_options: dict[str, object],
                         max_extract_urls: int, invoke: Callable[[WebRetrievalRequest], WebRetrievalReceipt],
                         destination: Path, sample_id: str, claim: str, chunk_chars: int,
                         fetched_at: str, provenance: Literal["synthetic_fixture", "recorded_response"]
                         ) -> WebResearchMaterialResult:
    """Run search then bounded extraction and reuse the original evidence bridge."""
    search_request = WebRetrievalRequest.model_validate(search_request.model_dump())
    if (search_request.operation != "search" or type(max_extract_urls) is not int
            or max_extract_urls <= 0):
        raise ValueError("invalid_search_or_extract_limit")
    if set(extract_options) - {"extract_depth", "format", "include_usage"}:
        raise ValueError("invalid_extract_options")
    if (type(chunk_chars) is not int or chunk_chars <= 0 or not isinstance(claim, str)
            or not claim.strip() or not isinstance(sample_id, str) or not sample_id.strip()):
        raise ValueError("invalid_material_identity_or_chunk_size")
    WebMaterialBatch.model_validate({"query": search_request.payload["query"],
        "fetched_at": fetched_at, "provenance": provenance, "candidates": (), "pages": (), "failures": ()})
    # 提取选项在发送搜索前校验，不等待真实候选URL出现。
    WebRetrievalRequest(operation="extract", payload={"urls": ["https://api.tavily.com"], **extract_options})
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "request.json").write_text(search_request.model_dump_json(indent=2), encoding="utf-8")
    search = _invoke_archived(search_request, destination, invoke)
    if search.status != "success":
        return WebResearchMaterialResult(status="retrieval_failed", receipts=(search,))
    try:
        urls = list(dict.fromkeys(str(SearchCandidate(url=row["url"], title=row["title"],
                    snippet=row["content"]).url) for row in search.response["results"]))[:max_extract_urls]
    except (KeyError, TypeError, ValidationError):
        return WebResearchMaterialResult(status="invalid_material", receipts=(search,))
    extract_request = WebRetrievalRequest(operation="extract", payload={"urls": urls, **extract_options}) if urls else None
    if extract_request is not None:
        (destination / "extract-request.json").write_text(extract_request.model_dump_json(indent=2), encoding="utf-8")
    extract = _invoke_archived(extract_request, destination, invoke) if extract_request else None
    if extract is not None and extract.status != "success":
        return WebResearchMaterialResult(status="retrieval_failed", receipts=(search, extract))
    receipts = (search, extract) if extract else (search,)
    try:
        batch = parse_web_responses(search.response, extract.response if extract else {"results": [], "failed_results": []},
            query=search_request.payload["query"], fetched_at=fetched_at, provenance=provenance)
    except (KeyError, TypeError, ValidationError):
        return WebResearchMaterialResult(status="invalid_material", receipts=receipts)
    manifest = archive_web_material(batch, destination / "evidence", sample_id=sample_id,
                                    claim=claim, chunk_chars=chunk_chars)
    return WebResearchMaterialResult(status="prepared" if manifest.entries else "empty", manifest=manifest, receipts=receipts)
