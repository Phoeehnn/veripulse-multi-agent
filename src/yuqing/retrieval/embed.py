"""BGE-M3 的稠密向量编码入口。"""

from __future__ import annotations

from functools import cache

import numpy as np

from src.yuqing.settings import configure_model_cache, get_retrieval_config


@cache
def load_embedding_model() -> object:
    """按需加载并复用同一个 BGE-M3 模型实例。"""

    configure_model_cache()
    from FlagEmbedding import BGEM3FlagModel

    config = get_retrieval_config()
    return BGEM3FlagModel(config.embedding_model, use_fp16=config.embedding_use_fp16)


def embed(texts: list[str]) -> np.ndarray:
    """批量编码文本并返回可用于 Chroma 的稠密向量矩阵。"""

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    config = get_retrieval_config()
    output = load_embedding_model().encode(
        texts,
        batch_size=config.embedding_batch_size,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("BGE-M3 返回的向量形状与输入条数不一致")
    return vectors
