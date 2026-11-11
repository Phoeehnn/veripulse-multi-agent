"""仅准备已核验的658联合演示；不包含Provider执行或正式钱包入口。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.yuqing import settings
from experiments.archive.m5_inputs import _runtime_code_files
from experiments.archive.m4_five_runner import _prompt_dev_permission
from src.yuqing.eval.preflight import validate_propagation_startup, validate_runtime_config
from src.yuqing.graph.composition import CaseComposition, CaseInputs
from src.yuqing.graph.fact_adapter import FactNodeAdapter
from src.yuqing.harness.initial_evidence import InitialEvidenceItemV2, InitialEvidenceManifestV2
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.retrieval.bm25 import build_bm25_index
from src.yuqing.retrieval.chunk import chunk
from src.yuqing.retrieval.search import search
from src.yuqing.retrieval.vector_store import index_chunks, validate_vector_corpus
from src.yuqing.schemas import AgentSpec, EvidenceChunk, Sample

MATERIAL_PATH = "data/demo_cases/joint-ced-658-v1.json"
MATERIAL_SHA256 = "ec21d04a76545dd1adc269ec5ecf2f641be9de5702b22d45598c71141859d2e9"
INVENTORY_PATH = "docs/cards/M9-J1_sources.json"
STORE_PATH = "data/processed/joint-ced-658-v1"
SAMPLE_ID = "ced:rumor:658_zmcLozTdA_2275337915"
CLAIM_PATH = "traces/joint-ced-658-v1.claim"


class JointSource(BaseModel):
    """精确时间和仅知日期分开保存，text只含连续原句摘录。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_key: Literal["S1", "S3"]
    title: str
    url: str
    publisher: str
    published_at: datetime | None
    published_on: date | None
    timestamp_precision: Literal["minute", "day_only"]
    author_verified: Literal[True]
    text: str
    text_sha256: str
    material_scope: str


class JointMaterial(BaseModel):
    """固定同案外壳，不接受Gold、预期结论或任意覆盖字段。"""
    model_config = ConfigDict(extra="forbid", frozen=True)
    manifest_version: Literal["joint-ced-658-v1"]
    sample_id: Literal["ced:rumor:658_zmcLozTdA_2275337915"]
    claim: str
    original_sha256: str
    propagation_sha256: str
    inventory_sha256: str
    is_demo: Literal[True]
    analysis_mode: Literal["retrospective"]
    as_of: None
    sources: tuple[JointSource, ...]


class JointSource2508(JointSource):
    """C06区分可知发布日期与论文署文日期，不推测网站上线时间。"""
    source_key: Literal["S1", "S2", "S3"]
    timestamp_precision: Literal["day_only", "document_date_only"]
    document_on: date | None = None


class JointMaterial2508(JointMaterial):
    """只接受作者核验过的C06身份，旧658合同不变。"""
    manifest_version: Literal["joint-ced-2508-v1"]
    sample_id: Literal["ced:rumor:2508_kDhyoY_1093974672"]
    sources: tuple[JointSource2508, ...]


class JointSource407(JointSource):
    """C07只使用两篇作者核验过的报道。"""
    source_key: Literal["S1", "S2"]


class JointMaterial407(JointMaterial):
    """固定C07 v2身份，来源摘录自带主语和适用范围。"""
    manifest_version: Literal["joint-ced-407-v2"]
    sample_id: Literal["ced:rumor:407_zgzvExfXe_2195600811"]
    sources: tuple[JointSource407, ...]


@dataclass(frozen=True)
class JointCaseFiles:
    """固定案例的资料定位与精确指纹，不接受外部传入任意路径。"""
    material: str
    inventory: str
    store: str
    material_sha256: str
    sample_id: str
    schema: type[JointMaterial]
    source_keys: tuple[str, ...]


def _case_files(case: str) -> JointCaseFiles:
    """旧入口默认658；新增案例只开放固定身份的离线准备。"""
    if case == "658":
        return JointCaseFiles(MATERIAL_PATH, INVENTORY_PATH, STORE_PATH,
                              MATERIAL_SHA256, SAMPLE_ID, JointMaterial, ("S1", "S3"))
    if case == "2508":
        return JointCaseFiles("data/demo_cases/joint-ced-2508-v1.json",
            "docs/cards/X15-b-C06_sources.json", "data/processed/joint-ced-2508-v1",
            "bbb03f05d7527f71673ed3699013dad0c1a7ce9f3338a1b89b7df52d97586603",
            "ced:rumor:2508_kDhyoY_1093974672", JointMaterial2508, ("S1", "S2", "S3"))
    if case == "407":
        return JointCaseFiles("data/demo_cases/joint-ced-407-v2.json",
            "docs/cards/X15-b-C07_sources-v2.json", "data/processed/joint-ced-407-v2",
            "1ee898452d3b88a9e44eefee7521e50c7b67cefb7d7cefebbd4c63c9919db55d",
            "ced:rumor:407_zgzvExfXe_2195600811", JointMaterial407, ("S1", "S2"))
    raise ValueError("joint_case_not_approved")


def _sha(path: Path) -> str:
    """计算已存在文件的原字节指纹。"""
    return sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    """计算不依赖字典顺序的结构指纹。"""
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _store_path(root: Path, case: str = "658") -> Path:
    """固定D盘项目子目录，拒绝通过目录链接复用别处的证据库。"""
    expected = root.resolve() / _case_files(case).store
    if expected.resolve() != expected:
        raise ValueError("joint_index_path_alias")
    return expected


def load_joint_material(root: Path, case: str = "658") -> tuple[JointMaterial, Sample]:
    """先验证许可、同案身份和原文，再提供不含Gold的推理样本。"""
    files = _case_files(case)
    if _sha(root / files.material) != files.material_sha256:
        raise ValueError("joint_material_drift")
    material = files.schema.model_validate_json((root / files.material).read_bytes())
    if _sha(root / files.inventory) != material.inventory_sha256:
        raise ValueError("joint_inventory_drift")
    inventory = json.loads((root / files.inventory).read_bytes())
    known = {item["source_key"]: item for item in inventory["sources"]}
    selected = list(files.source_keys)
    if (inventory.get("author_source_review_complete") is not True
            or inventory.get("selected_source_keys") != selected
            or [item.source_key for item in material.sources] != selected):
        raise ValueError("joint_sources_not_author_verified")
    for source in material.sources:
        ref = known[source.source_key]
        if (not source.text.strip() or sha256(source.text.encode()).hexdigest() != source.text_sha256
                or ref.get("author_verified") is not True
                or any(getattr(source, key) != ref[key] for key in ("url", "publisher", "title"))):
            raise ValueError("joint_source_invalid")
        if case != "658" and source.model_dump(mode="json") != ref:
            raise ValueError("joint_source_inventory_mismatch")
        if case == "658" and source.source_key == "S3" and (source.published_at is not None
                or source.published_on != date(2013, 3, 8) or source.timestamp_precision != "day_only"):
            raise ValueError("joint_source_time_precision_invalid")
    if case != "658":
        if _sha(root / inventory["shortlist_path"]) != inventory["shortlist_sha256"]:
            raise ValueError("joint_shortlist_drift")
        for item in inventory["protected_pool_checks"]:
            if _sha(root / item["path"]) != item["sha256"]:
                raise ValueError("joint_protected_pool_drift")
    stem = files.sample_id.rsplit(":", 1)[-1] + ".json"
    data = root / "data/raw/CED_Dataset"
    if (_sha(data / "original-microblog" / stem) != material.original_sha256
            or _sha(data / "rumor-repost" / stem) != material.propagation_sha256):
        raise ValueError("joint_ced_source_drift")
    sample = load_ced_sample(data, stem, rumor=True)
    if sample.claim != material.claim or sample.sample_id != material.sample_id:
        raise ValueError("joint_case_identity_mismatch")
    return material, sample.model_copy(update={"gold_veracity": None, "gold_risk": None})


def material_chunks(material: JointMaterial) -> list[EvidenceChunk]:
    """按既有切片器处理所有选定正文，不加答案或补充别案材料。"""
    identity = material.sample_id.rsplit(":", 1)[-1].split("_")[0]
    return [EvidenceChunk(chunk_id=f"joint-{identity}:{source.source_key}:{index}",
        source_doc=source.url, position=index, text=text)
        for source in material.sources for index, text in enumerate(chunk(source.text))]


def _corpus_signature(material: JointMaterial, chunks: list[EvidenceChunk],
                      case: str = "658") -> dict[str, object]:
    """绑定正文、切片、模型版本和编码参数，缺本地版本即停止。"""
    config = settings.get_retrieval_config()
    revision = (settings.get_settings().model_cache_dir / "huggingface/hub" /
        ("models--" + config.embedding_model.replace("/", "--")) / "refs/main")
    if not revision.is_file():
        raise ValueError("joint_local_embedding_missing_no_download")
    return {"material_sha256": _case_files(case).material_sha256,
        "inventory_sha256": material.inventory_sha256,
        "chunks_sha256": _digest([c.model_dump(exclude={"score"}) for c in chunks]),
        "embedding_revision": revision.read_text(encoding="utf-8").strip(),
        "retrieval_config_sha256": _digest(config.model_dump())}


def _initial_manifest(sample: Sample, ranked: list[EvidenceChunk],
                      signature: dict[str, object]) -> InitialEvidenceManifestV2:
    """把实际检索排名转换为既有V2初始交付合同。"""
    entries = [InitialEvidenceItemV2(chunk_id=c.chunk_id, source_doc=c.source_doc,
        position=c.position, sources=[c], text=c.text, text_sha256=sha256(c.text.encode()).hexdigest(),
        initial_rank=i + 1, read_start=0, read_end=len(c.text)) for i, c in enumerate(ranked)]
    return InitialEvidenceManifestV2(manifest_version="fact-v2-initial-v1",
        sample_id=sample.sample_id, claim_sha256=sha256(sample.claim.encode()).hexdigest(),
        retrieval_sha256=str(signature["retrieval_config_sha256"]),
        corpus_sha256=str(signature["chunks_sha256"]),
        limits_sha256=_digest(settings.get_fact_v2_evaluation_config().model_dump()),
        entries_sha256=_digest([e.model_dump(mode="json") for e in entries]),
        retrieval_status="success" if entries else "empty", original_candidate_count=len(ranked),
        delivered_count=len(entries), entries=entries)


def build_joint_corpus(root: Path = settings.PROJECT_ROOT, case: str = "658") -> Path:
    """显式构建独立小库；复用必须先验证，部分旧目录不擅自删除续建。"""
    material, sample = load_joint_material(root, case)
    chunks = material_chunks(material)
    signature = _corpus_signature(material, chunks, case)
    target = _store_path(root, case)
    if target.exists():
        preflight_joint_demo(root, case)
        return target
    target.mkdir(parents=True)
    index_chunks(chunks, target)
    validate_vector_corpus(chunks, target)
    ranked = search(sample.claim, build_bm25_index(chunks),
                    k=settings.get_harness_config().search_k, path=target)
    manifest = _initial_manifest(sample, ranked, signature)
    (target / "initial.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    signature["initial_sha256"] = _sha(target / "initial.json")
    (target / "corpus.json").write_text(json.dumps(signature, sort_keys=True), encoding="utf-8")
    (target / "READY").write_text(_digest(signature), encoding="ascii")
    preflight_joint_demo(root, case)
    return target


def _validate_joint_store(root: Path, material: JointMaterial,
                           chunks: list[EvidenceChunk], case: str = "658") -> Path:
    """不以目录存在代替完整性；核对清单、初始正文及实际库内容。"""
    target = _store_path(root, case)
    for name in ("READY", "corpus.json", "initial.json", "chroma.sqlite3"):
        if not (target / name).is_file() or (target / name).resolve().parent != target:
            raise ValueError("joint_index_incomplete:" + name)
    signature = _corpus_signature(material, chunks, case)
    signature["initial_sha256"] = _sha(target / "initial.json")
    if (json.loads((target / "corpus.json").read_bytes()) != signature
            or (target / "READY").read_text(encoding="ascii") != _digest(signature)):
        raise ValueError("joint_index_signature_drift")
    manifest = InitialEvidenceManifestV2.model_validate_json((target / "initial.json").read_bytes())
    expected = {item.chunk_id: item for item in chunks}
    if (manifest.sample_id != material.sample_id
            or manifest.claim_sha256 != sha256(material.claim.encode()).hexdigest()
            or manifest.corpus_sha256 != signature["chunks_sha256"]
            or manifest.retrieval_sha256 != signature["retrieval_config_sha256"]
            or manifest.limits_sha256 != _digest(settings.get_fact_v2_evaluation_config().model_dump())
            or manifest.entries_sha256 != _digest([entry.model_dump(mode="json") for entry in manifest.entries])
            or manifest.delivered_count != len(manifest.entries)
            or any(entry.chunk_id not in expected or entry.text != expected[entry.chunk_id].text
                   or entry.source_doc != expected[entry.chunk_id].source_doc
                   or entry.position != expected[entry.chunk_id].position
                   or entry.text_sha256 != sha256(entry.text.encode()).hexdigest()
                   for entry in manifest.entries)):
        raise ValueError("joint_initial_manifest_drift")
    validate_vector_corpus(chunks, target)
    return target


def _conditions(root: Path, target: Path, case: str = "658") -> dict[str, str]:
    """覆盖实际图依赖、Prompt、配置、原始资料和独立索引指纹。"""
    files = _case_files(case)
    paths = {*_runtime_code_files(root), files.material, files.inventory,
        "experiments/archive/joint_demo.py", "scripts/run_joint_demo.py",
        "experiments/archive/joint_run.py", ".gitattributes",
        "src/yuqing/agents/propagation_v11.py", "config/weights.yaml", "config/fact_v2_eval.yaml",
        "config/rules_compliance.yaml", "prompts/fact_check/investigator_v2.md",
        "prompts/fact_check/v2_manifest.json", "prompts/propagation/v1.md",
        "prompts/propagation/contracts.json", "prompts/arbiter/v2.md", "prompts/arbiter/contracts.json"}
    values = {name: _sha(root / name) for name in sorted(paths)}
    inventory = json.loads((root / files.inventory).read_bytes())
    if case != "658":
        paths.add(inventory["shortlist_path"])
        paths.update(item["path"] for item in inventory["protected_pool_checks"])
        values.update({name: _sha(root / name) for name in paths if name not in values})
    values.update({name: _sha(root / name) for name in
                   (inventory["original_path"], inventory["propagation_path"])})
    values.update({name: _sha(target / name) for name in ("READY", "corpus.json", "initial.json")})
    config, prices = settings.get_settings(), settings.get_cost_config()
    values.update(model=config.deepseek_model, base_url_sha256=sha256(config.deepseek_base_url.encode()).hexdigest(),
                  prices_sha256=_digest(prices.model_dump()), timeout=str(config.llm_timeout_s),
                  thinking=str(config.llm_thinking), propagation_version="v1.1", analysis_mode="retrospective")
    return values


@dataclass(frozen=True)
class PreparedJointDemo:
    """已核验离线依赖；本对象本身不创建钱包或授予模型调用权限。"""
    root: Path
    sample: Sample
    chunks: tuple[EvidenceChunk, ...]
    store: Path
    conditions: dict[str, str]
    propagation_spec: AgentSpec
    arbiter_spec: AgentSpec
    case: str = "658"

    def composition(self, wallet: object, trace_root: Path) -> CaseComposition:
        """给已有、由调用者管理的钱包装配同案资源，默认仍不允许invoke。"""
        current = _validated_joint_demo(self.root, self.case)
        if current.conditions != self.conditions:
            raise ValueError("joint_conditions_changed")
        traces = (self.root / "traces").resolve()
        if trace_root.resolve() == traces or not trace_root.resolve().is_relative_to(traces):
            raise ValueError("joint_trace_outside_project")
        fact_scope, propagation_scope = EvidenceWorkerScope(), EvidenceWorkerScope()
        inputs = CaseInputs(case_id="joint_demo:" + self.case + ":" + trace_root.name, sample=self.sample,
            coverage="demo", trace_root=trace_root, wallet=wallet,
            fact_worker_scope=fact_scope, propagation_worker_scope=propagation_scope,
            index=build_bm25_index(list(self.chunks)), store_path=self.store,
            initial_manifest_path=self.store / "initial.json", propagation_spec=self.propagation_spec,
            arbiter_spec=self.arbiter_spec, as_of=None, group_id="joint_demo", variant="fact-v2_prop-v1.1",
            propagation_contract_version="v1.1")
        try:
            return CaseComposition(inputs)
        except Exception:
            fact_scope.close()
            propagation_scope.close()
            raise

    def state(self, case_id: str) -> dict[str, object]:
        """生成明确演示、无Gold的图初始状态，不接受外部覆盖标志。"""
        return {"case_id": case_id, "sample": self.sample.model_dump(mode="json"),
                "ticket": {"claim": self.sample.claim}, "is_demo": True, "as_of": None}


def preflight_joint_demo(root: Path = settings.PROJECT_ROOT, case: str = "658") -> PreparedJointDemo:
    """核对资料与许可，拒绝重复认领；不联网、不建正式钱包、不认领实验。"""
    _case_files(case)
    if case == "658" and (root / CLAIM_PATH).exists():
        raise ValueError("joint_attempt_already_claimed")
    return _validated_joint_demo(root, case)


def _validated_joint_demo(root: Path, case: str = "658") -> PreparedJointDemo:
    """复验同案依赖；认领后的调用者另须核对本场claim，不解除执行权限制。"""
    material, sample = load_joint_material(root, case)
    validate_runtime_config()
    FactNodeAdapter(settings.get_graph_config().fact_node_version, root=root)
    propagation = validate_propagation_startup(root / "prompts/propagation/v1.md", offline=True)
    if not _prompt_dev_permission(root, root / "prompts/arbiter/v2.md",
                                   normalize_checkout_newlines=True):
        raise ValueError("joint_arbiter_prompt_not_licensed")
    chunks = material_chunks(material)
    target = _validate_joint_store(root, material, chunks, case)
    return PreparedJointDemo(root, sample, tuple(chunks), target, _conditions(root, target, case),
        AgentSpec(agent="propagation_analyst", system_prompt=propagation.text),
        AgentSpec(agent="arbiter", system_prompt=(root / "prompts/arbiter/v2.md").read_text(encoding="utf-8")), case)
