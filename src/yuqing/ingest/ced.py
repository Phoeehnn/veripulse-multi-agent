"""把 CED 的原帖与传播记录转换成统一样本。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.yuqing.schemas import PropagationRaw, RepostNode, Sample

CHINA_TIME = timezone(timedelta(hours=8))


def _parse_root_time(value: object) -> datetime:
    """将 CED 原帖的 Unix 秒数转换为东八区时间。"""

    return datetime.fromtimestamp(int(value), tz=CHINA_TIME)


def _parse_repost_time(value: object) -> datetime:
    """将 CED 转发文本时间按数据集使用的东八区解释。"""

    parsed = datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=CHINA_TIME)


def load_ced_sample(root: Path, filename: str, *, rumor: bool) -> Sample:
    """从指定类别目录读取一条 CED 原帖和传播树。"""

    if Path(filename).name != filename:
        raise ValueError("CED 文件名只能是单个文件名")
    category = "rumor" if rumor else "non_rumor"
    folder = "rumor-repost" if rumor else "non-rumor-repost"
    original_path = root / "original-microblog" / filename
    repost_path = root / folder / filename
    original = json.loads(original_path.read_text(encoding="utf-8"))
    raw_nodes = json.loads(repost_path.read_text(encoding="utf-8"))
    if not isinstance(original, dict) or not isinstance(raw_nodes, list):
        raise ValueError(f"CED 记录格式无效：{filename}")
    claim = original.get("text")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(f"CED 原帖正文为空：{filename}")
    root_time = _parse_root_time(original.get("time"))
    user = original.get("user") or {}
    nodes = [RepostNode(mid=str(item["mid"]), uid=str(item["uid"]),
                        parent=str(item.get("parent", "")), text=str(item.get("text", "")),
                        date=_parse_repost_time(item["date"])) for item in raw_nodes]
    propagation = PropagationRaw(
        root_time=root_time, root_followers=int(user.get("followers", 0)),
        root_verified=bool(user.get("verified", False)), comments=int(original.get("comments", 0)),
        reposts=int(original.get("reposts", 0)), likes=int(original.get("likes", 0)), nodes=nodes,
    )
    return Sample(sample_id=f"ced:{category}:{Path(filename).stem}", source="ced", claim=claim,
                  published_at=root_time, gold_veracity="假" if rumor else "真", propagation=propagation)
