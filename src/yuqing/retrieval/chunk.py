"""按中文字符长度切分证据文本。"""

from __future__ import annotations

from src.yuqing.settings import get_retrieval_config


def chunk(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """按字符数切片并保留相邻片段的重叠上下文。"""

    config = get_retrieval_config()
    size = config.chunk_size if size is None else size
    overlap = config.chunk_overlap if overlap is None else overlap
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("切分参数必须满足 size > 0 且 0 <= overlap < size")
    if not text:
        return []

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return pieces
