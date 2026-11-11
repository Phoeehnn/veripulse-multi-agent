"""用排序名次融合向量和关键词检索结果。"""

from __future__ import annotations

import re

from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_retrieval_config


def rrf_fuse(
    a: list[EvidenceChunk], b: list[EvidenceChunk], k: int | None = None
) -> list[EvidenceChunk]:
    """按倒数排名分数融合两路结果，并按片段 ID 去重。"""

    damping = get_retrieval_config().rrf_k if k is None else k
    if damping <= 0:
        raise ValueError("RRF 的 k 必须大于零")
    scores: dict[str, float] = {}
    chunks: dict[str, EvidenceChunk] = {}
    for ranking in (a, b):
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            chunks.setdefault(item.chunk_id, item)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1 / (damping + rank)
    return [
        chunks[chunk_id].model_copy(update={"score": score})
        for chunk_id, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ]


def select_diverse_evidence(ranked: list[EvidenceChunk], k: int) -> list[EvidenceChunk]:
    """优先保留不同文档，去除重复文本，再补同文档的不同片段。"""

    if k <= 0:
        raise ValueError("证据条数必须大于零")
    selected: list[EvidenceChunk] = []
    sources: set[str] = set()
    signatures: set[str] = set()
    selected_ids: set[str] = set()
    for distinct_source in (True, False):
        for item in ranked:
            signature = re.sub(r"\s+", "", item.text)
            if item.chunk_id in selected_ids or signature in signatures:
                continue
            if distinct_source and item.source_doc in sources:
                continue
            selected.append(item)
            sources.add(item.source_doc)
            signatures.add(signature)
            selected_ids.add(item.chunk_id)
            if len(selected) == k:
                return selected
    return selected
