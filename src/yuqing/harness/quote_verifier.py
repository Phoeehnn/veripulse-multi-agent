"""按合同在已交付正文内匹配引文并映射回原文坐标。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Literal


@dataclass(frozen=True)
class QuoteLocation:
    """引文匹配状态及可复核的原文半开区间。"""

    status: Literal["matched", "not_found", "offset_unresolved"]
    normalized_text: str
    start: int | None = None
    end: int | None = None
    ambiguous: bool = False


def normalize_quote_text(text: str) -> str:
    """统一兼容字符和连续 Unicode 空白，不改标点或否定词。"""
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized)


def locate_quote(raw_text: str, quote: str, *, read_end: int) -> QuoteLocation:
    """查找规范化引文并返回已交付原文中的最早可复检范围。"""
    if read_end < 0 or read_end > len(raw_text):
        return QuoteLocation(status="offset_unresolved", normalized_text=normalize_quote_text(quote))
    visible = raw_text[:read_end]
    normalized_quote = normalize_quote_text(quote)
    normalized_body = normalize_quote_text(visible)
    if not normalized_quote or normalized_quote not in normalized_body:
        return QuoteLocation(status="not_found", normalized_text=normalized_quote)
    offsets = _prefix_normalized_lengths(visible)
    matches, unresolved = _raw_matches(visible, normalized_quote, normalized_body, offsets)
    if matches:
        start, end = min(matches)
        return QuoteLocation(status="matched", normalized_text=normalized_quote,
                             start=start, end=end, ambiguous=len(matches) > 1)
    status: Literal["not_found", "offset_unresolved"] = "offset_unresolved" if unresolved else "not_found"
    return QuoteLocation(status=status, normalized_text=normalized_quote)


def _prefix_normalized_lengths(text: str) -> tuple[int, ...]:
    """预先计算原文边界对应的规范化前缀长度。"""
    return tuple(len(normalize_quote_text(text[:index])) for index in range(len(text) + 1))


def _raw_matches(text: str, quote: str, body: str, offsets: tuple[int, ...]) -> tuple[list[tuple[int, int]], bool]:
    """复检规范化命中的原始区间，并收集坐标歧义。"""
    matches: list[tuple[int, int]] = []
    unresolved = False
    cursor = 0
    while (position := body.find(quote, cursor)) != -1:
        pairs = _candidate_raw_ranges(text, quote, position, offsets)
        if pairs:
            matches.extend(pairs)
        else:
            unresolved = True
        cursor = position + 1
    return matches, unresolved


def _candidate_raw_ranges(text: str, quote: str, position: int,
                          offsets: tuple[int, ...]) -> list[tuple[int, int]]:
    """只接受切片重新规范化后等于原引文的原文范围。"""
    end_position = position + len(quote)
    starts = [index for index, length in enumerate(offsets) if length == position]
    ends = [index for index, length in enumerate(offsets) if length == end_position]
    return [(start, end) for start in starts for end in ends
            if start < end and normalize_quote_text(text[start:end]) == quote]
