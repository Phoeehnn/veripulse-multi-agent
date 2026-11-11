"""为 M5 生成不改变可信材料和引用资格的紧凑仲裁数据视图。"""

from __future__ import annotations

from src.yuqing.collab.arbiter_material_v2 import ArbiterMaterialV2


def m5_fact_packet(material: ArbiterMaterialV2) -> dict[str, object]:
    """保留语义材料，以显式文本别名消除完全相同的已读正文重复。"""
    packet = material.model_dump(mode="json", exclude={"proposal_origins", "text_sources",
        "fact_import_provenance", "claim_sha256"})
    for report in packet["fact_reports"]:
        for key in ("snapshot_sha256", "report_sha256", "trace_sha256"):
            report.pop(key)
    texts: dict[tuple[str, str, int], str] = {}
    for row in packet["evidence"]:
        identity = (row["text"], row["body_sha256"], row["delivered_chars"])
        if identity in texts:
            row["text_same_as"] = texts[identity]
            row.pop("text")
        else:
            texts[identity] = row["evidence_ref"]
    return packet
