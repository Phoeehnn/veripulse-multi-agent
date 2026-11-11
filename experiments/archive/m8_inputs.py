"""Prepare the fixed M8 60-CHEF/40-CED inputs without exposing evaluation labels."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.yuqing.schemas import Sample
from src.yuqing.settings import PROJECT_ROOT


SELECTION_PATH = "eval/label_pool_100.jsonl"
LABELS_PATH = "data/labels_100.jsonl"
ARCHIVE_VERSION = "m8-inputs-v1"
_CODE_PATHS = (
    "scripts/prepare_fact_v2.py", "src/yuqing/eval/fact_v2_conditions.py",
    "src/yuqing/eval/fact_v2_experiment.py", "src/yuqing/eval/dataset.py",
    "src/yuqing/graph/composition.py", "src/yuqing/graph/runtime.py",
    "src/yuqing/graph/fact_adapter.py", "src/yuqing/agents/propagation.py",
    "src/yuqing/agents/arbiter_loop.py", "src/yuqing/harness/initial_evidence.py",
    "src/yuqing/retrieval/search.py", "src/yuqing/retrieval/corpus.py",
    "src/yuqing/settings.py", "src/yuqing/schemas.py", "src/yuqing/llm.py",
    "config/weights.yaml", "config/fact_v2_eval.yaml",
    "prompts/fact_check/single_v4.md", "prompts/fact_check/investigator_v2.md",
    "prompts/arbiter/v2.md", "prompts/propagation/v1.md",
)


@dataclass(frozen=True)
class M8CaseInput:
    """One ordered model-facing sample and its source fingerprints."""

    order: int
    sample_id: str
    source: str
    coverage: str
    sample_path: Path
    sample_sha256: str
    initial_path: Path | None
    initial_sha256: str | None


@dataclass(frozen=True)
class M8Inputs:
    """Verified fixed-100 archive; risk labels remain separate from model inputs."""

    archive: Path
    archive_sha256: str
    cases: tuple[M8CaseInput, ...]
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class _BuildContext:
    """Local, validated inputs needed to prepare each fixed case."""

    config: Any
    test_store: Path
    chef_samples: dict[str, Sample]
    index: Any
    prompts: dict[str, str]
    retrieval_sha: str
    corpus_sha: str
    limits_sha: str


def _sha(data: bytes) -> str:
    """Return a lowercase SHA-256 for exact file bytes."""
    return sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    """Encode stable UTF-8 JSON for fingerprints."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read strict object-per-line JSONL and reject blank or malformed rows."""
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            raise ValueError("m8_input_jsonl_blank_row")
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError("m8_input_jsonl_row_not_object")
        rows.append(item)
    return rows


def _model_input_sample(sample: Sample) -> bytes:
    """Serialize only fields that may safely be delivered to the model."""
    return _json_bytes(sample.model_dump(mode="json", exclude={
        "gold_veracity", "gold_risk", "evidence_ids"})) + b"\n"


def _validate_pool(pool: list[dict[str, Any]], labels: list[dict[str, Any]]) -> None:
    """Require fixed 60/40 identities and separate valid risk labels in the same order."""
    if len(pool) != 100 or len(labels) != 100:
        raise ValueError("m8_input_pool_count_invalid")
    ids = [str(row.get("sample_id", "")) for row in pool]
    label_ids = [str(row.get("sample_id", "")) for row in labels]
    if not all(ids) or len(set(ids)) != 100 or ids != label_ids:
        raise ValueError("m8_input_identity_order_mismatch")
    if sum(row.get("source") == "chef" for row in pool) != 60 or sum(
            row.get("source") == "ced" for row in pool) != 40:
        raise ValueError("m8_input_source_counts_invalid")
    if any(isinstance(row.get("risk_level"), bool)
            or not isinstance(row.get("risk_level"), int)
            or row["risk_level"] not in {1, 2, 3, 4} for row in labels):
        raise ValueError("m8_input_risk_label_invalid")


def _source_paths(root: Path, pool: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return source/config/code paths, including only selected CED source records."""
    paths = {SELECTION_PATH, LABELS_PATH, "data/CHEF/test.json",
        "config/fact_v2_eval.yaml", "config/weights.yaml", *_CODE_PATHS}
    for row in pool:
        if row["source"] != "ced":
            continue
        sample_id = str(row["sample_id"])
        filename = sample_id.rsplit(":", 1)[1] + ".json"
        suffix = "rumor" if ":rumor:" in sample_id else "non_rumor"
        repost = "rumor-repost" if suffix == "rumor" else "non-rumor-repost"
        paths.add(f"data/raw/CED_Dataset/original-microblog/{filename}")
        paths.add(f"data/raw/CED_Dataset/{repost}/{filename}")
    result = tuple(sorted(paths))
    missing = [name for name in result if not (root / name).is_file()]
    if missing:
        raise ValueError("m8_input_source_missing:" + missing[0])
    return result


def _stable_corpus_files(store: Path) -> tuple[Path, ...]:
    """Fingerprint the corpus manifest and immutable index files, not Chroma's query ledger."""
    return tuple(sorted(path for path in store.rglob("*") if path.is_file()
        and path.name != "chroma.sqlite3" and not path.name.startswith("chroma.sqlite3-")))


def _source_shas(root: Path, pool: list[dict[str, Any]]) -> dict[str, str]:
    """Hash every file that can affect selection, input rendering, or evaluation."""
    files = {name: (root / name) for name in _source_paths(root, pool)}
    from src.yuqing.settings import get_settings

    store = get_settings().test_store_path.resolve(strict=True)
    for path in _stable_corpus_files(store):
        files["corpus/" + path.relative_to(store).as_posix()] = path
    if not any(name.endswith("/READY") for name in files):
        raise ValueError("m8_test_corpus_not_ready")
    return {name: _sha(path.read_bytes()) for name, path in sorted(files.items())}


def _validate_source_snapshot(current: dict[str, str], recorded: dict[str, str]) -> None:
    """Check runtime sources while retaining generation-time docs and mutable SQLite hashes."""
    provenance_only = {"docs/cards/M8_plan.md", "scripts/prepare_m8_inputs.py",
        "experiments/archive/m8_inputs.py", "corpus/chroma.sqlite3"}
    if not isinstance(recorded, dict):
        raise ValueError("m8_input_source_manifest_invalid")
    missing = set(current) - set(recorded)
    extra = set(recorded) - set(current)
    if (missing or any(not _is_provenance_source(name, provenance_only) for name in extra)
            or any(recorded[name] != digest for name, digest in current.items())):
        raise ValueError("m8_input_source_drift")


def _is_provenance_source(name: str, exact: set[str]) -> bool:
    """Allow only known preparation metadata and Chroma's mutable SQLite bookkeeping."""
    return name in exact or name.startswith("corpus/chroma.sqlite3-")


def _load_case(archive: Path, row: dict[str, Any], order: int) -> M8CaseInput:
    """Load one stripped sample and verify its input files against the manifest row."""
    folder = archive / f"case-{order:03d}"
    if not folder.resolve(strict=True).is_relative_to(archive.resolve(strict=True)):
        raise ValueError("m8_input_case_path_invalid")
    sample_path = folder / "sample.json"
    raw = _read_archive_file(sample_path, archive)
    if _sha(raw) != row["sample_sha256"]:
        raise ValueError("m8_input_sample_digest_mismatch")
    model_fields = json.loads(raw)
    if {"gold_veracity", "gold_risk", "evidence_ids"}.intersection(model_fields):
        raise ValueError("m8_input_model_fields_contain_gold")
    sample = Sample.model_validate_json(raw, strict=True)
    if (sample.sample_id != row["sample_id"] or sample.source != row["source"]
            or sample.gold_veracity is not None or sample.gold_risk is not None):
        raise ValueError("m8_input_sample_identity_or_label_leak")
    initial = folder / "initial-evidence.json" if row["source"] == "chef" else None
    if row["source"] == "chef" and (not row.get("initial_sha256")
            or not row.get("initial_audit_sha256")):
        raise ValueError("m8_input_initial_evidence_metadata_missing")
    if row["source"] == "ced" and any(row.get(name) for name in (
            "initial_sha256", "initial_audit_sha256")):
        raise ValueError("m8_input_ced_must_not_have_fact_evidence")
    if initial is not None:
        from src.yuqing.harness.initial_evidence import load_initial_evidence_manifest

        audit = folder / "initial-evidence-audit.json"
        if _sha(_read_archive_file(audit, archive)) != row.get("initial_audit_sha256"):
            raise ValueError("m8_input_evidence_audit_digest_mismatch")
        initial_raw = _read_archive_file(initial, archive)
        if _sha(initial_raw) != row["initial_sha256"]:
            raise ValueError("m8_input_evidence_digest_mismatch")
        manifest_path = archive / f"case-{order:03d}" / "initial-evidence.json"
        load_initial_evidence_manifest(manifest_path, sample.sample_id, sample.claim)
    return M8CaseInput(order, sample.sample_id, sample.source, row["coverage"], sample_path,
        row["sample_sha256"], initial, row.get("initial_sha256"))


def _read_archive_file(path: Path, archive: Path) -> bytes:
    """Read a regular archive file only after resolving symlinks within the archive."""
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(archive.resolve(strict=True)) or not resolved.is_file():
        raise ValueError("m8_input_archive_member_invalid")
    return resolved.read_bytes()


def prepare_m8_inputs(root: Path, archive: Path) -> Path:
    """Create a new D-drive fixed-100 input archive; never write labels into case inputs."""
    target, traces = archive.resolve(), (root / "traces/review-packages").resolve(strict=True)
    if target.exists() or target == traces or not target.is_relative_to(traces):
        raise ValueError("m8_input_archive_path_invalid_or_exists")
    pool_path, labels_path = root / SELECTION_PATH, root / LABELS_PATH
    pool, labels = _read_jsonl(pool_path), _read_jsonl(labels_path)
    _validate_pool(pool, labels)
    sources = _source_shas(root, pool)
    return _prepare_archive(root, target, pool, sources)


def _build_context(root: Path) -> _BuildContext:
    """Validate the full test corpus and construct the same retrieval dependencies as V2."""
    from src.yuqing.ingest.chef import load_chef
    from src.yuqing.retrieval.bm25 import build_bm25_index
    from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus
    from src.yuqing.settings import get_settings
    from scripts.prepare_fact_v2 import (_limits_fingerprint, _prepare_one_initial,
        _retrieval_fingerprints)
    from src.yuqing.eval.fact_v2_experiment import PROMPT_FILES

    config = get_settings()
    test_store = config.test_store_path.resolve(strict=True)
    if not (test_store / "READY").is_file() or not (test_store / "corpus.json").is_file():
        raise ValueError("m8_test_corpus_not_ready")
    chef_samples = {sample.sample_id: sample for sample in load_chef(config.chef_test_path)}
    corpus = load_eval_corpus(config.chef_test_path, [], corpus_name="chef_test_full")
    validate_eval_corpus(corpus, test_store, corpus_name="chef_test_full")
    index = build_bm25_index(corpus)
    prompts = {role: (root / name).read_text(encoding="utf-8")
        for role, name in PROMPT_FILES.items()}
    retrieval_sha, corpus_sha = _retrieval_fingerprints(root, test_store)
    limits_sha = _limits_fingerprint(root, prompts["single"], prompts["loop"])
    return _BuildContext(config, test_store, chef_samples, index, prompts,
        retrieval_sha, corpus_sha, limits_sha)


def _validate_test_corpus_content() -> None:
    """Recheck stored evidence text and source metadata without hashing mutable SQLite bytes."""
    from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus
    from src.yuqing.settings import get_settings

    config = get_settings()
    corpus = load_eval_corpus(config.chef_test_path, [], corpus_name="chef_test_full")
    validate_eval_corpus(corpus, config.test_store_path, corpus_name="chef_test_full")


def _prepare_case(folder: Path, selected: dict[str, Any], order: int,
                  context: _BuildContext) -> dict[str, Any]:
    """Write one label-free model input and its optional CHEF initial evidence."""
    from src.yuqing.eval.dataset import RetrievalQuestion
    from src.yuqing.ingest.ced import load_ced_sample
    from scripts.prepare_fact_v2 import _prepare_one_initial

    sample, coverage, initial_sha = None, "none", None
    if selected["source"] == "chef":
        sample = context.chef_samples.get(selected["sample_id"])
        if sample is None or (sample.claim, sample.gold_veracity) != (
                selected["claim"], selected["gold_veracity"]):
            raise ValueError("m8_chef_source_mismatch:" + str(selected["sample_id"]))
        question = RetrievalQuestion(sample_id=sample.sample_id, claim=sample.claim,
            gold_veracity=sample.gold_veracity, gold_evidence_ids=[])
        manifest, audit = _prepare_one_initial(question, context.index, context.test_store,
            context.prompts["single"], context.prompts["loop"], context.retrieval_sha,
            context.corpus_sha, context.limits_sha)
        initial = folder / "initial-evidence.json"
        initial.write_bytes(manifest.model_dump_json().encode("utf-8"))
        audit_path = folder / "initial-evidence-audit.json"
        audit_path.write_bytes(_json_bytes(audit) + b"\n")
        initial_sha, coverage = _sha(initial.read_bytes()), "full"
        sample = sample.model_copy(update={"gold_veracity": None,
            "gold_risk": None, "evidence_ids": []})
    else:
        sample = _load_ced_sample(context.config.ced_dataset_path, selected)
    if sample is None:
        raise ValueError("m8_input_sample_unavailable")
    raw_sample = _model_input_sample(sample)
    (folder / "sample.json").write_bytes(raw_sample)
    audit_path = folder / "initial-evidence-audit.json"
    return {"order": order, "sample_id": sample.sample_id, "source": sample.source,
        "coverage": coverage, "sample_sha256": _sha(raw_sample),
        "initial_sha256": initial_sha,
        "initial_audit_sha256": _sha(audit_path.read_bytes()) if audit_path.exists() else None}


def _load_ced_sample(dataset: Path, selected: dict[str, Any]) -> Sample:
    """Load the registered CED original/reposts and reject changed or absent records."""
    from src.yuqing.ingest.ced import load_ced_sample

    sample_id = str(selected["sample_id"])
    filename = sample_id.rsplit(":", 1)[1] + ".json"
    sample = load_ced_sample(dataset, filename, rumor=":rumor:" in sample_id)
    if sample.claim != selected["claim"] or sample.sample_id != sample_id:
        raise ValueError("m8_ced_source_mismatch:" + sample_id)
    if sample.propagation is None or sample.propagation.model_dump(
            mode="json") != selected["propagation"]:
        raise ValueError("m8_ced_propagation_mismatch:" + sample_id)
    return sample.model_copy(update={"gold_veracity": None,
        "gold_risk": None, "evidence_ids": []})


def _prepare_archive(root: Path, target: Path, pool: list[dict[str, Any]],
                     sources: dict[str, str]) -> Path:
    """Build all 100 model inputs in a private staging directory, then publish once."""
    context = _build_context(root)
    staging = target.with_name(target.name + "." + uuid4().hex + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        rows = []
        for order, selected in enumerate(pool, start=1):
            folder = staging / f"case-{order:03d}"
            folder.mkdir()
            rows.append(_prepare_case(folder, selected, order, context))
        if _source_shas(root, pool) != sources:
            raise ValueError("m8_input_source_drift_during_prepare")
        summary = {"version": ARCHIVE_VERSION,
            "selection_sha256": sources[SELECTION_PATH],
            "risk_labels_sha256": sources[LABELS_PATH], "sources": sources,
            "cases": rows, "sample_count": len(rows),
            "source_counts": {"chef": 60, "ced": 40},
            "conditions_sha256": _sha(_json_bytes({"sources": sources, "cases": rows}))}
        (staging / "inputs.json").write_bytes(_json_bytes(summary) + b"\n")
        summary["archive_sha256"] = _sha((staging / "inputs.json").read_bytes())
        (staging / "summary.json").write_bytes(_json_bytes(summary) + b"\n")
        staging.rename(target)
        return target / "summary.json"
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_m8_inputs(root: Path, summary_path: Path,
                   expected_summary_sha256: str) -> M8Inputs:
    """Verify the archive summary, source bytes, ordered samples, and CHEF evidence manifests."""
    actual_summary = summary_path.resolve(strict=True)
    archive = actual_summary.parent
    traces = (root / "traces/review-packages").resolve(strict=True)
    if (not archive.is_relative_to(traces) or actual_summary.name != "summary.json"
            or not actual_summary.is_file()):
        raise ValueError("m8_input_archive_path_invalid")
    summary_raw = _read_archive_file(actual_summary, archive)
    if _sha(summary_raw) != expected_summary_sha256:
        raise ValueError("m8_input_summary_digest_mismatch")
    summary = json.loads(summary_raw)
    archive_raw = _read_archive_file(archive / "inputs.json", archive)
    if _sha(archive_raw) != summary.get("archive_sha256"):
        raise ValueError("m8_input_manifest_digest_mismatch")
    manifest = json.loads(archive_raw)
    projected = {key: value for key, value in summary.items() if key != "archive_sha256"}
    if manifest != projected or manifest.get("version") != ARCHIVE_VERSION:
        raise ValueError("m8_input_manifest_mismatch")
    if summary.get("conditions_sha256") != _sha(_json_bytes({
            "sources": summary.get("sources"), "cases": summary.get("cases")})):
        raise ValueError("m8_input_conditions_digest_mismatch")
    pool = _read_jsonl(root / SELECTION_PATH)
    labels = _read_jsonl(root / LABELS_PATH)
    _validate_pool(pool, labels)
    _validate_test_corpus_content()
    sources = _source_shas(root, pool)
    _validate_source_snapshot(sources, summary.get("sources", {}))
    rows = summary.get("cases")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("m8_input_case_count_invalid")
    cases = []
    for expected_order, (selected, row) in enumerate(zip(pool, rows, strict=True), start=1):
        if (row.get("order"), row.get("sample_id"), row.get("source")) != (
                expected_order, selected["sample_id"], selected["source"]):
            raise ValueError("m8_input_order_or_identity_mismatch")
        case = _load_case(archive, row, expected_order)
        sample = Sample.model_validate_json(_read_archive_file(case.sample_path, archive), strict=True)
        expected_coverage = "full" if selected["source"] == "chef" else "none"
        if sample.claim != selected["claim"] or case.coverage != expected_coverage:
            raise ValueError("m8_input_claim_or_coverage_mismatch")
        if selected["source"] == "ced":
            if sample.propagation is None or sample.propagation.model_dump(
                    mode="json") != selected["propagation"]:
                raise ValueError("m8_input_propagation_mismatch")
        cases.append(case)
    return M8Inputs(archive, summary["archive_sha256"], tuple(cases), sources)
