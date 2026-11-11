"""用 BGE-reranker-v2-m3 对候选证据重新排序。"""

from __future__ import annotations

from functools import cache

import numpy as np

from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import configure_model_cache, get_retrieval_config


@cache
def load_reranker() -> object:
    """按需加载并复用同一个 BGE 重排模型。"""

    configure_model_cache()
    from FlagEmbedding import FlagReranker

    config = get_retrieval_config()
    return FlagReranker(
        config.rerank_model,
        query_max_length=config.rerank_query_max_length,
        max_length=config.rerank_passage_max_length,
        batch_size=config.rerank_batch_size,
        use_fp16=config.rerank_use_fp16,
    )


def rerank(
    query: str, docs: list[EvidenceChunk], top_k: int | None = None
) -> list[EvidenceChunk]:
    """用声明与候选证据的成对相关性重排并保留前 k 条。"""

    limit = get_retrieval_config().final_top_k if top_k is None else top_k
    if limit <= 0:
        raise ValueError("重排 top_k 必须大于零")
    if not docs:
        return []
    pairs = [(query, doc.text) for doc in docs]
    scores = np.atleast_1d(
        load_reranker().compute_score(pairs, normalize=True)
    ).astype(float)
    if len(scores) != len(docs) or not np.isfinite(scores).all():
        raise ValueError("重排模型返回的分数数量或数值无效")
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))[:limit]
    return [docs[i].model_copy(update={"score": float(scores[i])}) for i in order]
