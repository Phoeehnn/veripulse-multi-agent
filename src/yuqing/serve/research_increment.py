"""Offline sample-only increment archive, separate from full investigation execution."""
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json
import re
from src.yuqing.serve.research_contracts import ResearchSnapshot, ResearchRequest
from src.yuqing.serve.research_public import PublicSampleSession, project_public_sample
from src.yuqing.serve.research_service import save_research_snapshot
from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask
from src.yuqing.retrieval.public_material_contracts import PublicSourceDescriptor
from src.yuqing.retrieval.public_materials import adapt_public_material, archive_public_material
from src.yuqing.retrieval.web_contracts import WebMaterialBatch
from src.yuqing.serve.research_public_activation import APPROVED_URLS, APPROVED_PROMPT_SHA


def source_case(root: Path, identity: str) -> Path:
    """Resolve an existing run strictly inside this history root."""
    if not re.fullmatch(r"[0-9a-f]{32}", identity):
        raise ValueError("invalid_increment_source")
    case=(root/identity).resolve()
    if case.parent != root.resolve():
        raise ValueError("increment_source_escape")
    return case


def prepare_increment(root: Path, identity: str) -> tuple[Path, ResearchSnapshot, PublicPropagationTask]:
    """Copy provenance only; bind freshly read W material to a new offline attempt identity."""
    source=source_case(root,identity)
    if (source/"sample-increment.json").exists():
        raise ValueError("increment_requires_original_main_report")
    snapshot_bytes=(source/"snapshot.json").read_bytes()
    snapshot=ResearchSnapshot.model_validate_json(snapshot_bytes)
    if snapshot.status != "completed":
        raise ValueError("increment_requires_completed_main_report")
    request=ResearchRequest.model_validate_json((source/"request.json").read_bytes())
    reuse=json.loads((source/"reuse.json").read_text()) if (source/"reuse.json").exists() else {}
    material_source=source_case(root,reuse.get("source_case",identity))
    original=ResearchRequest.model_validate_json((material_source/"request.json").read_bytes())
    if request != original:
        raise ValueError("increment_claim_mismatch")
    material_bytes=(material_source/"retrieval/evidence/materials.json").read_bytes()
    batch=WebMaterialBatch.model_validate_json(material_bytes)
    if batch.query != request.input_value:
        raise ValueError("increment_material_claim_mismatch")
    descriptors=tuple(PublicSourceDescriptor(url=p.url,source_kind="news_report",source_label="已批准新闻报道",
        classification_basis="人工确认的首批报道URL") for p in batch.pages if str(p.url) in APPROVED_URLS)
    if not descriptors:
        raise ValueError("increment_no_approved_material")
    new_id=uuid4().hex
    task=PublicPropagationTask(sample_id=new_id,claim=request.input_value,
        batch=adapt_public_material(batch,descriptors,case_id=new_id,claim=request.input_value))
    case=root/new_id
    case.mkdir()
    (case/"request.json").write_text(request.model_dump_json(),encoding="utf-8")
    archive_public_material(task.batch,case/"public-material")
    manifest={"mode":"offline_sample_increment","main_source":identity,"material_source":material_source.name,
        "main_snapshot_sha256":sha256(snapshot_bytes).hexdigest(),"material_file_sha256":sha256(material_bytes).hexdigest(),
        "approved_prompt_sha256":APPROVED_PROMPT_SHA,"execution_prompt_sha256":None,"fact_and_arbiter":"reused_not_executed"}
    (case/"sample-increment.json").write_text(json.dumps(manifest,ensure_ascii=False),encoding="utf-8")
    return case,snapshot,task


def archive_sample_increment(root: Path, identity: str, *, job: Callable,
                             cost_receipt: Callable[[str], dict[str, object]]) -> str:
    """Execute only an explicitly injected offline sample job, preserving failures and unknown costs."""
    case,base,task=prepare_increment(root,identity)
    started=datetime.now(timezone.utc)
    session=PublicSampleSession(task,lambda:job(task,case))
    session.round1({})
    base=base.model_copy(update={"research_id":case.name,"execution_mode":"mock","last_seq":0})
    snapshot=project_public_sample(base,session,sample_id=task.sample_id)
    snapshot=snapshot.model_copy(update={"status":"completed" if session.status=="completed" else "failed",
        "report":"离线样本增量 · 主报告复用，事实与仲裁未重跑\n\n"+(snapshot.report or "")})
    try:
        cost=cost_receipt(task.sample_id)
    except Exception:
        cost={"known_cost_cny":None,"cost_status":"unknown","cumulative_known_cny":None,"cumulative_limit_cny":"5"}
    (case/"cost.json").write_text(json.dumps(cost),encoding="utf-8")
    snapshot=snapshot.model_copy(update={"cost_status":cost["cost_status"],"known_cost_cny":cost.get("known_cost_cny")})
    save_research_snapshot(case,snapshot)
    (case/"sample-result.json").write_text(session.result.model_dump_json() if session.result else "null",encoding="utf-8")
    rows=[{"time":started.isoformat(),"stage":"仅样本离线增量（主报告复用）","status":"running"},
          {"time":datetime.now(timezone.utc).isoformat(),"stage":"仅样本离线增量","status":session.status}]
    (case/"stages.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows),encoding="utf-8")
    return case.name
