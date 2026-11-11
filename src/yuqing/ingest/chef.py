"""将 CHEF 原始记录转换为项目统一的样本模型。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from src.yuqing.schemas import Sample
from src.yuqing.schemas import EvidenceChunk
from src.yuqing.retrieval.chunk import chunk


def load_chef(path: Path) -> list[Sample]:
    """读取一个 CHEF 数据分片，并为每条声明生成稳定且唯一的样本与证据 ID。"""

    label_map = {0: "真", 1: "假", 2: "无法判定"}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"CHEF 文件顶层必须是列表：{path}")

    samples: list[Sample] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"CHEF 第 {index} 条记录必须是对象")
        label = record.get("label")
        if type(label) is not int or label not in label_map:
            raise ValueError(f"CHEF 第 {index} 条记录的标签无效：{label!r}")
        evidence = record.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"CHEF 第 {index} 条记录的 evidence 必须是对象")

        # claimId 在原始数据中会重复，分片名与行号共同定位样本。
        sample_id = f"chef:{path.stem}:{index}"
        raw_date = record.get("publish_date", "")
        published_at = (
            datetime(*map(int, re.split(r"[/-]", raw_date))) if raw_date else None
        )
        samples.append(
            Sample(
                sample_id=sample_id,
                source="chef",
                claim=record["claim"],
                published_at=published_at,
                gold_veracity=label_map[label],
                evidence_ids=[f"{sample_id}:evidence:{key}" for key in evidence],
            )
        )
    return samples


def load_chef_evidence(path: Path) -> list[EvidenceChunk]:
    """只将 CHEF 候选证据切成带来源 ID 与位置的检索片段。"""

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"CHEF 文件顶层必须是列表：{path}")

    chunks: list[EvidenceChunk] = []
    for row_index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("evidence"), dict):
            raise ValueError(f"CHEF 第 {row_index} 条记录的 evidence 无效")
        for key, item in record["evidence"].items():
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise ValueError(f"CHEF 第 {row_index} 条的证据 {key} 缺少文本")
            source_doc = f"chef:{path.stem}:{row_index}:evidence:{key}"
            for position, piece in enumerate(chunk(item["text"])):
                chunks.append(
                    EvidenceChunk(
                        chunk_id=f"{source_doc}:chunk:{position}",
                        text=piece,
                        source_doc=source_doc,
                        position=position,
                    )
                )
    return chunks
