"""证据检索的可复现指标。"""

from __future__ import annotations

from typing import Literal

from src.yuqing.schemas import EvidenceChunk
from src.yuqing.settings import get_retrieval_config


def hit_at_k(
    retrieved: list[EvidenceChunk], gold_evidence_ids: list[str], k: int | None = None
) -> float:
    """前 k 个片段中只要有一个来自 gold 证据文档就记为命中。"""

    limit = get_retrieval_config().final_top_k if k is None else k
    if limit <= 0:
        raise ValueError("hit@k 的 k 必须大于零")
    if not gold_evidence_ids:
        raise ValueError("计算 hit@k 需要至少一个 gold 证据 ID")
    gold = set(gold_evidence_ids)
    return float(any(item.source_doc in gold for item in retrieved[:limit]))


def recall_at_k(
    retrieved: list[EvidenceChunk], gold_evidence_ids: list[str], k: int
) -> float:
    """前 k 个返回片段按来源文档去重，计算已定位 gold 文档的找回比例。"""

    if k <= 0:
        raise ValueError("recall@k 的 k 必须大于零")
    if not gold_evidence_ids:
        raise ValueError("计算 recall@k 需要至少一个 gold 证据 ID")
    gold = set(gold_evidence_ids)
    found = {item.source_doc for item in retrieved[:k]}
    return len(found & gold) / len(gold)


def reciprocal_rank_at_k(
    retrieved: list[EvidenceChunk], gold_evidence_ids: list[str], k: int
) -> float:
    """按前 k 个返回片段中首个 gold 来源的位置算倒数排名，未命中为零。"""

    if k <= 0:
        raise ValueError("倒数排名的 k 必须大于零")
    if not gold_evidence_ids:
        raise ValueError("计算倒数排名需要至少一个 gold 证据 ID")
    gold = set(gold_evidence_ids)
    return next((1 / rank for rank, item in enumerate(retrieved[:k], start=1)
                 if item.source_doc in gold), 0.0)


def mean_hit_at_k(
    rows: list[dict[str, str | float | int]],
    method: Literal["vector", "bm25", "hybrid", "rerank"],
) -> float:
    """从逐题结果计算指定检索方法的平均 hit@k。"""

    if not rows:
        raise ValueError("计算平均 hit@k 需要至少一条评测行")
    field = f"{method}_hit_at_k"
    if any(row[field] == "" for row in rows):
        raise ValueError("存在不可计分题目；请显式报告覆盖率后选择可计分子集")
    return sum(float(row[field]) for row in rows) / len(rows)


def fact_check_summary(rows: list[dict[str, str | float | int]]) -> dict[str, int | float | None]:
    """固定三类计算 Macro-F1，保留失败题并区分已知费用小计与完整费用。"""

    if not rows:
        raise ValueError("没有事实核查结果可汇总")
    f1s = []
    for label in ("支持", "反驳", "信息不足"):
        tp = sum(row["gold_label"] == label and row["predicted_label"] == label for row in rows)
        fp = sum(row["gold_label"] != label and row["predicted_label"] == label for row in rows)
        fn = sum(row["gold_label"] == label and row["predicted_label"] != label for row in rows)
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    known_costs = [float(row["cost_cny"]) for row in rows if row["cost_cny"] not in ("", None)]
    unknown = len(rows) - len(known_costs)
    subtotals = [float(row["known_cost_subtotal_cny"])
                 if row.get("known_cost_subtotal_cny") not in ("", None)
                 else float(row["cost_cny"]) if row["cost_cny"] not in ("", None) else 0.0
                 for row in rows]
    return {"total": len(rows), "correct": sum(int(row["correct"]) for row in rows),
            "macro_f1": sum(f1s) / len(f1s),
            "cost_cny": None if unknown else sum(known_costs),
            "known_cost_subtotal_cny": sum(subtotals), "unknown_cost_rows": unknown}
