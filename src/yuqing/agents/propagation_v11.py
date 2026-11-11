"""传播Agent v1.1的受限Schema与模型可见回执。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.yuqing.features.propagation import SpreadStats
from src.yuqing.schemas import PropagationReport

SPREAD_BASIS_FIELDS = (
    "reposts", "comments", "likes", "root_followers", "root_verified", "nodes_1h",
    "nodes_6h", "nodes_24h", "peak_nodes_per_hour", "max_depth", "direct_reposts",
    "text_repost_ratio", "visible_comments",
)
SpreadBasisValue = Literal[SPREAD_BASIS_FIELDS]


class PropagationStatsReceipt(BaseModel):
    """只含本次已交付可引用统计和非证据性可用性说明。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    values: dict[str, int | float | bool]
    allowed_spread_basis: list[str]
    availability_note: str = Field(min_length=1)


class PropagationReportV11(PropagationReport):
    """v1.1提交Schema：统计依据字段名必须来自固定业务字段集合。"""

    spread_basis: list[SpreadBasisValue] = Field(default_factory=list)


def stats_receipt(stats: SpreadStats) -> PropagationStatsReceipt:
    """把内部统计拆成允许引用的值、字段名和一句可用性说明。"""

    values = {name: getattr(stats, name) for name in available_fields(stats)}
    if not stats.has_data:
        note = "本案没有传播资料覆盖。"
    elif stats.visible_comments == 0:
        note = "本案有传播统计，但当前快照没有可用评论文本。"
    else:
        note = "统计按当前可见快照计算；评论是否可引用以已返回的评论编号为准。"
    return PropagationStatsReceipt(values=values, allowed_spread_basis=list(values),
                                   availability_note=note)


def available_fields(stats: SpreadStats | None) -> tuple[str, ...]:
    """从本案已交付统计计算新版展示、Schema和反馈共用的业务字段。"""
    if stats is None or not stats.has_data:
        return ()
    return tuple(name for name in SPREAD_BASIS_FIELDS
                 if name not in stats.unavailable_fields and getattr(stats, name) is not None)


def empty_comment_note(*, has_coverage: bool, has_visible_comments: bool) -> str:
    """解释空评论工具回执，区分无覆盖、无评论与关键词未命中。"""

    if not has_coverage:
        return "本案没有传播资料覆盖。"
    if not has_visible_comments:
        return "本案当前快照没有可用评论文本。"
    return "本案存在可用评论，但本次检索没有匹配结果。"


def v11_submission_feedback(values: list[str], available: tuple[str, ...]) -> str:
    """指出统计依据中具体非法位置及本案允许值，不自动修正报告。"""

    allowed = set(available)
    invalid = [index for index, value in enumerate(values) if value not in allowed]
    choices = "、".join(available) if available else "无"
    location_limit = len(SPREAD_BASIS_FIELDS)
    locations = "、".join(f"spread_basis[{index}]" for index in invalid[:location_limit]) or "spread_basis"
    more = "；还有其他非法项" if len(invalid) > location_limit else ""
    return f"{locations}{more}：字段名不属于本次已交付且非空可用项；本案允许值：{choices}。请自行修正。"


def v11_emotion_feedback(index: int, allowed_sources: tuple[str, ...]) -> str:
    """定位不合格情绪引文并列出可用来源，不复述外部引文。"""

    allowed = "、".join(allowed_sources) if allowed_sources else "无"
    return (f"emotion_signals[{index}].source/quote 与可用原文不匹配；"
            f"请从允许来源重新连续摘录，当前允许来源：{allowed}。")


def submission_model(available: tuple[str, ...]) -> type[BaseModel]:
    """按本案已交付字段动态限制spread_basis枚举。"""

    if available:
        basis_type = list[Literal[available]]
        field = (basis_type, Field(default_factory=list))
    else:
        field = (list[str], Field(default_factory=list, max_length=0))
    return create_model("PropagationReportV11Submission", __base__=PropagationReportV11,
                        spread_basis=field)
