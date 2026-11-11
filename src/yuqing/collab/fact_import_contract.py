"""M4专用历史事实导入的不可变来源合同。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class M4FactImportBindingV1(BaseModel):
    """把一个固定历史答卷绑定到新分配的M4案件身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1, pattern=r"\S")
    case_id: str = Field(min_length=1, pattern=r"\S")
    fact_round_no: int = Field(gt=0, strict=True)
    experiment_id: str = Field(min_length=1, pattern=r"\S")
    condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    group_id: Literal["iterative"]
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_path: str = Field(min_length=1, pattern=r"\S")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_manifest_path: str = Field(min_length=1, pattern=r"\S")
    initial_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class M4FactImportProvenanceV1(BaseModel):
    """仲裁材料中的程序侧导入来源摘要，不代表原始Trace曾记录M4身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_kind: Literal["m4_historical_import"] = "m4_historical_import"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_id: str = Field(min_length=1, pattern=r"\S")
    case_id: str = Field(min_length=1, pattern=r"\S")
    fact_round_no: int = Field(gt=0, strict=True)
    experiment_id: str = Field(min_length=1, pattern=r"\S")
    condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    initial_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
