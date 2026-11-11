"""确定性传播数据统计、评论抽样与检索。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.yuqing.schemas import PropagationRaw, RepostNode
from src.yuqing.settings import get_propagation_tools_config


class CommentItem(BaseModel):
    """传播树中可引用的一条文字节点。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    comment_id: str
    text: str
    date: datetime


class SpreadStats(BaseModel):
    """可用于传播研判的原始快照与截止时刻衍生统计。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    has_data: bool
    reposts: int | None
    comments: int | None
    likes: int | None
    root_followers: int | None
    root_verified: bool | None
    nodes_1h: int | None
    nodes_6h: int | None
    nodes_24h: int | None
    peak_nodes_per_hour: int | None
    max_depth: int | None
    direct_reposts: int | None
    text_repost_ratio: float | None
    visible_comments: int | None
    visible_nodes: int
    unavailable_fields: tuple[str, ...]


def _validate_time(value: datetime, label: str) -> None:
    """要求时间带时区，避免不同口径的时间比较。"""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label}_timezone_required")


def _visible_nodes(raw: PropagationRaw | None, as_of: datetime | None) -> list[tuple[int, RepostNode]]:
    """先按原始位置编号，再按时间过滤并保持来源顺序。"""

    if as_of is not None:
        _validate_time(as_of, "as_of")
    if raw is None:
        return []
    _validate_time(raw.root_time, "root_time")
    visible: list[tuple[int, RepostNode]] = []
    for index, node in enumerate(raw.nodes, start=1):
        _validate_time(node.date, "node_date")
        if as_of is None or node.date <= as_of:
            visible.append((index, node))
    return visible


def _validate_k(k: int, maximum: int) -> None:
    """拒绝非整数、零值及超出规格的结果数。"""

    if type(k) is not int or k <= 0 or k > maximum:
        raise ValueError("comment_limit_out_of_range")


def _comment_items(nodes: list[tuple[int, RepostNode]], *, truncate: bool = True) -> list[CommentItem]:
    """只暴露有正文的节点，并按原始位置保留 C 编号。"""

    limit = get_propagation_tools_config().comment_preview_chars
    return [CommentItem(comment_id=f"C{index}", text=node.text[:limit], date=node.date)
            if truncate else CommentItem(comment_id=f"C{index}", text=node.text, date=node.date)
            for index, node in nodes if node.text.strip()]


def _node_depths(nodes: list[tuple[int, RepostNode]]) -> dict[str, int] | None:
    """对完整、无环的可见父子树计算深度；异常拓扑返回未知。"""

    by_id = {node.mid: node for _, node in nodes}
    if len(by_id) != len(nodes):
        return None
    depths: dict[str, int] = {}
    for mid in by_id:
        path: list[str] = []
        current = mid
        while current not in depths:
            if current in path or current not in by_id:
                return None
            path.append(current)
            parent = by_id[current].parent
            if not parent:
                depth = 1
                break
            current = parent
        else:
            depth = depths[current] + 1
        for item in reversed(path):
            depths[item] = depth
            depth += 1
    return depths


def _hour_counts(raw: PropagationRaw, nodes: list[tuple[int, RepostNode]]) -> list[int]:
    """按原帖后的整点小时桶统计可见节点。"""

    counts: Counter[int] = Counter()
    for _, node in nodes:
        elapsed = node.date - raw.root_time
        if elapsed >= timedelta(0):
            counts[int(elapsed.total_seconds() // 3600)] += 1
    return list(counts.values())


def get_spread_stats(raw: PropagationRaw | None, *, as_of: datetime | None = None) -> SpreadStats:
    """计算可见传播树统计，并隐藏无法按截止时间回溯的快照字段。"""

    visible = _visible_nodes(raw, as_of)
    if raw is None:
        return SpreadStats(has_data=False, reposts=None, comments=None, likes=None,
                           root_followers=None, root_verified=None, nodes_1h=None,
                           nodes_6h=None, nodes_24h=None, peak_nodes_per_hour=None,
                           max_depth=None, direct_reposts=None, text_repost_ratio=None,
                           visible_comments=None, visible_nodes=0,
                           unavailable_fields=("reposts", "comments", "likes", "root_followers",
                                               "root_verified", "nodes_1h", "nodes_6h", "nodes_24h",
                                               "peak_nodes_per_hour", "max_depth", "direct_reposts",
                                               "text_repost_ratio", "visible_comments"))
    config = get_propagation_tools_config()
    elapsed = [node.date - raw.root_time for _, node in visible]
    windows = [sum(timedelta(0) <= delta <= timedelta(hours=hours) for delta in elapsed)
               for hours in config.window_hours]
    hours = _hour_counts(raw, visible)
    depths = _node_depths(visible)
    nonempty = sum(bool(node.text.strip()) for _, node in visible)
    invalid_topology = depths is None
    temporal_aggregate = as_of is not None
    unavailable = ["reposts", "comments", "likes", "root_followers", "root_verified"] if temporal_aggregate else []
    if invalid_topology:
        unavailable.append("max_depth")
    return SpreadStats(
        has_data=True,
        reposts=None if temporal_aggregate else raw.reposts,
        comments=None if temporal_aggregate else raw.comments,
        likes=None if temporal_aggregate else raw.likes,
        root_followers=None if temporal_aggregate else raw.root_followers,
        root_verified=None if temporal_aggregate else raw.root_verified,
        nodes_1h=windows[0], nodes_6h=windows[1], nodes_24h=windows[2],
        peak_nodes_per_hour=max(hours, default=0),
        max_depth=None if invalid_topology else max(depths.values(), default=0),
        direct_reposts=sum(not node.parent for _, node in visible),
        text_repost_ratio=None if not visible else nonempty / len(visible),
        visible_comments=nonempty, visible_nodes=len(visible),
        unavailable_fields=tuple(unavailable),
    )


def sample_comments(raw: PropagationRaw | None, *, sample_id: str,
                    order: Literal["earliest", "latest", "seeded_random"], k: int,
                    as_of: datetime | None = None) -> list[CommentItem]:
    """确定性抽取可见文字节点，编号始终对应原始节点位置。"""

    _validate_k(k, get_propagation_tools_config().sample_max_k)
    if not sample_id.strip():
        raise ValueError("sample_id_required")
    if order not in ("earliest", "latest", "seeded_random"):
        raise ValueError("comment_order_invalid")
    items = _comment_items(_visible_nodes(raw, as_of))
    if order == "earliest":
        ordered = sorted(items, key=lambda item: (item.date, int(item.comment_id[1:])))
    elif order == "latest":
        ordered = sorted(items, key=lambda item: (-item.date.timestamp(), int(item.comment_id[1:])))
    else:
        fingerprints = {f"C{index}": f"{node.mid}:{node.uid}:{node.parent}:{node.date.isoformat()}:{node.text}"
                        for index, node in _visible_nodes(raw, as_of)}
        ordered = sorted(items, key=lambda item: sha256(
            f"{sample_id}:{item.comment_id}:{fingerprints[item.comment_id]}".encode()).digest())
    return ordered[:k]


def _query_terms(query: str) -> Counter[str]:
    """将中文查询拆成相邻字词片段，不依赖外部分词器。"""

    compact = "".join(query.split()).casefold()
    if len(compact) < 2:
        return Counter([compact])
    return Counter(compact[index:index + 2] for index in range(len(compact) - 1))


def search_comments(raw: PropagationRaw | None, *, query: str, k: int,
                    as_of: datetime | None = None) -> list[CommentItem]:
    """按确定性关键词覆盖排序检索本案可见评论。"""

    config = get_propagation_tools_config()
    _validate_k(k, config.search_max_k)
    if not isinstance(query, str) or not query.strip() or len(query) > config.search_query_max_chars:
        raise ValueError("search_query_out_of_range")
    terms = _query_terms(query)
    scored: list[tuple[int, int, CommentItem]] = []
    for item in _comment_items(_visible_nodes(raw, as_of), truncate=False):
        text = item.text.casefold()
        overlap = sum(min(count, text.count(term)) for term, count in terms.items())
        if overlap:
            scored.append((int(query.casefold() in text), overlap, item))
    scored.sort(key=lambda row: (-row[0], -row[1], int(row[2].comment_id[1:])))
    limit = config.comment_preview_chars
    return [item.model_copy(update={"text": item.text[:limit]}) for _, _, item in scored[:k]]
