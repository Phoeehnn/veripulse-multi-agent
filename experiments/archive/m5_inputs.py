"""构建并核验M5固定十题的逐案输入归档。"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from src.yuqing import settings
from src.yuqing.schemas import Sample

SELECTION = "docs/cards/M5-B-selection-v1.json"
LEGACY_SUMMARY = "docs/cards/M5-B-inputs-v1.json"
PREVIOUS_SUMMARY = "docs/cards/M5-B-inputs-v2.json"
V3_SUMMARY = "docs/cards/M5-B-inputs-v3.json"
SOURCE_SUMMARY = "docs/cards/M5-B-inputs-v4.json"
SUMMARY = "docs/cards/M5-B-inputs-v5.json"
ARCHIVE_VERSION = "m5-inputs-v5"
FIXED_CAP = Decimal("6")
_CODE_FILES = (
    "experiments/archive/m5_checkpoint.py", "experiments/archive/m5_inputs.py",
    "experiments/archive/m5_checkpoint_audit.py", "src/yuqing/eval/dual_budget.py",
    "src/yuqing/graph/composition.py",
    "src/yuqing/graph/runtime.py", "src/yuqing/graph/fact_adapter.py",
    "src/yuqing/graph/build.py", "src/yuqing/graph/nodes.py",
    "src/yuqing/graph/fact_selection.py",
    "src/yuqing/agents/propagation.py", "src/yuqing/agents/arbiter_loop.py",
    "src/yuqing/agents/arbiter.py", "src/yuqing/agents/compliance.py",
    "src/yuqing/eval/preflight.py", "src/yuqing/settings.py", "src/yuqing/schemas.py",
    "src/yuqing/llm.py", "src/yuqing/harness/fact_v2_contract.py",
    "src/yuqing/harness/initial_evidence.py", "src/yuqing/harness/inherited_evidence.py",
    "src/yuqing/harness/context.py", "src/yuqing/harness/ledger.py",
    "src/yuqing/harness/verifier.py", "src/yuqing/harness/tools.py",
    "src/yuqing/harness/loop.py", "src/yuqing/harness/policy.py",
    "src/yuqing/harness/trace.py",
    "src/yuqing/harness/feedback.py", "src/yuqing/harness/budget.py",
    "src/yuqing/collab/fact_archive.py", "src/yuqing/collab/fact_material.py",
    "src/yuqing/collab/arbiter_budget_v2.py", "src/yuqing/collab/dispatchability.py",
    "src/yuqing/collab/validate.py", "src/yuqing/collab/board.py",
    "src/yuqing/collab/arbiter_material_v2.py",
    "src/yuqing/ingest/chef.py", "src/yuqing/ingest/ced.py",
    "src/yuqing/retrieval/corpus.py", "src/yuqing/retrieval/bm25.py",
    "src/yuqing/eval/experiment_budget.py",
    "src/yuqing/eval/experiment_budget_recovery.py",
    "src/yuqing/eval/experiment_budget_store.py",
    "src/yuqing/report/case_usage.py",
    "experiments/archive/m8_cache.py",
    "experiments/archive/m8_cache_validation.py",
    "experiments/archive/m8_runner.py",
    "scripts/run_m5_checkpoint.py",
    "src/yuqing/eval/experiment_session.py", "src/yuqing/eval/persistent_experiment_budget.py",
    "src/yuqing/__init__.py", "src/yuqing/agents/__init__.py",
    "src/yuqing/agents/base.py", "src/yuqing/agents/fact_check.py",
    "src/yuqing/collab/__init__.py", "src/yuqing/features/__init__.py",
    "src/yuqing/graph/__init__.py", "src/yuqing/harness/__init__.py",
    "src/yuqing/ingest/__init__.py", "src/yuqing/report/__init__.py",
    "src/yuqing/collab/arbiter_view.py", "src/yuqing/collab/fact_delivery.py",
    "src/yuqing/collab/fact_import_contract.py", "src/yuqing/collab/m4_fact_history_import.py",
    "src/yuqing/collab/m4_history.py", "src/yuqing/collab/m4_readiness.py",
    "src/yuqing/collab/progress.py", "src/yuqing/collab/proposals.py",
    "src/yuqing/eval/__init__.py", "src/yuqing/eval/dataset.py",
    "experiments/archive/e2_artifacts.py", "experiments/archive/e2_checkpoint.py",
    "src/yuqing/eval/fact_v2_conditions.py", "src/yuqing/eval/fact_v2_experiment.py",
    "src/yuqing/eval/label_consistency.py", "experiments/archive/m4_five_runner.py",
    "src/yuqing/eval/metrics.py", "src/yuqing/eval/propagation_ced10.py",
    "src/yuqing/eval/propagation_eval.py", "src/yuqing/eval/run_eval.py",
    "src/yuqing/features/propagation.py", "src/yuqing/graph/fork.py",
    "src/yuqing/graph/state.py", "src/yuqing/harness/fact_v2_messages.py",
    "src/yuqing/harness/quote_verifier.py", "src/yuqing/report/brief.py",
    "src/yuqing/retrieval/__init__.py", "src/yuqing/retrieval/chunk.py",
    "src/yuqing/retrieval/embed.py", "src/yuqing/retrieval/fusion.py",
    "src/yuqing/retrieval/rerank.py", "src/yuqing/retrieval/search.py",
    "src/yuqing/retrieval/vector_store.py",
    "scripts/prepare_fact_v2.py",
)


@dataclass(frozen=True)
class M5CaseInput:
    """One selected, source-verified case with its exact initial evidence artifact."""

    order: int
    sample: Sample
    coverage: str
    sample_sha256: str
    initial_path: Path | None
    initial_sha256: str | None


@dataclass(frozen=True)
class M5Inputs:
    """The immutable ten-case order and the on-disk archive identity."""

    archive: Path
    archive_sha256: str
    cases: tuple[M5CaseInput, ...]
    source_sha256: dict[str, str]


def _sha(data: bytes) -> str:
    """Return the lowercase SHA-256 of exact bytes."""
    return sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    """Encode deterministic UTF-8 JSON for manifests and per-case archives."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _runtime_code_files(root: Path) -> tuple[str, ...]:
    """Include existing Python package initializers above every registered source."""
    files = set(_CODE_FILES)
    source_root = (root / "src").resolve()
    for name in _CODE_FILES:
        source = (root / name).resolve()
        if not source.is_relative_to(source_root):
            continue
        parent = source.parent
        while parent.is_relative_to(source_root):
            initializer = parent / "__init__.py"
            if initializer.is_file():
                files.add(initializer.relative_to(root.resolve()).as_posix())
            if parent == source_root:
                break
            parent = parent.parent
    return tuple(sorted(files))


def _source_shas(root: Path) -> dict[str, str]:
    """Bind data, manifests, prompts, configuration and the M5 runtime sources."""
    paths = [SELECTION, "docs/cards/M5-B-CHEF-selection-v1.json",
        "docs/cards/F1-CED10-selection-v1.json", "docs/cards/M4-five-sample-manifest.json",
        "eval/questions_30.jsonl", "eval/dev_extra_60.jsonl", "eval/dev90_initial_evidence_v2.jsonl",
        "eval/dev90_initial_evidence_v2_audit.jsonl",
        "eval/dev_val_120.jsonl", "eval/dev_val_120_v2.jsonl", "eval/test_60.jsonl",
        "eval/label_pool_100.jsonl",
        "data/CHEF/dev.json", "config/weights.yaml", "config/fact_v2_eval.yaml",
        "config/rules_compliance.yaml",
        "prompts/fact_check/investigator_v2.md", "prompts/fact_check/v2_manifest.json",
        "prompts/propagation/v1.md", "prompts/propagation/contracts.json",
        "prompts/arbiter/v2.md", "prompts/arbiter/contracts.json",
        *_runtime_code_files(root)]
    result = {}
    for name in paths:
        path = root / name
        if not path.is_file():
            raise ValueError("m5_source_file_missing:" + name)
        result[name] = _sha(path.read_bytes())
    return result


def _archive_shas(root: Path, archive: Path) -> dict[str, str]:
    """Hash the actual ten-case archive and selected raw CED sources on every check."""
    from src.yuqing.settings import get_settings

    summary = json.loads((root / SUMMARY).read_text(encoding="utf-8"))
    rows = summary.get("cases")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("m5_input_archive_case_manifest_invalid")
    paths, expected = [archive / "inputs.json"], {}
    for row in rows:
        order = row.get("order")
        folder = archive / f"case-{order:02d}"
        sample, audit = folder / "sample.json", folder / "initial-evidence-audit.json"
        paths.append(sample)
        expected[sample] = row.get("sample_sha256")
        if row.get("initial_audit_sha256") is not None:
            paths.append(audit)
            expected[audit] = row["initial_audit_sha256"]
        if row.get("coverage") != "none":
            initial = folder / "initial-evidence.json"
            paths.append(initial)
            expected[initial] = row.get("initial_sha256")
    if any(not path.is_file() for path in paths):
        raise ValueError("m5_input_archive_file_missing")
    result = {"archive:" + path.relative_to(archive).as_posix(): _sha(path.read_bytes())
        for path in paths}
    if any(result["archive:" + path.relative_to(archive).as_posix()] != digest
            for path, digest in expected.items()):
        raise ValueError("m5_input_archive_case_changed")
    unsigned = dict(summary)
    registration_sha = unsigned.pop("archive_summary_sha256", None)
    if _sha(_json_bytes(unsigned) + b"\n") != registration_sha:
        raise ValueError("m5_input_summary_changed")
    if (summary.get("archive") != archive.relative_to(root).as_posix()
            or _sha((archive / "inputs.json").read_bytes()) != summary.get("archive_sha256")):
        raise ValueError("m5_input_archive_fingerprint_mismatch")
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    for item in selection["samples"]:
        if item["source"] != "CED":
            continue
        filename = item["sample_id"].rsplit(":", 1)[1] + ".json"
        rumor = ":rumor:" in item["sample_id"]
        result.update({"ced:" + name: value for name, value in _ced_source_shas(
            get_settings().ced_dataset_path, filename, rumor).items()})
    return result


def _verify_committed_runtime(root: Path) -> dict[str, str]:
    """Require every runtime file to match its Git HEAD blob, ignoring unrelated dirty docs."""
    import subprocess

    digests = {}
    runtime_files = _runtime_code_files(root)
    for name in runtime_files:
        path = root / name
        if not path.is_file():
            raise ValueError("m5_runtime_file_missing:" + name)
        digests[name] = _sha(path.read_bytes())
    try:
        tracked = subprocess.run(["git",
            "ls-files", "--error-unmatch", *runtime_files], cwd=root, check=True,
            capture_output=True, text=True).stdout.splitlines()
        if set(tracked) != set(runtime_files):
            raise ValueError("m5_runtime_file_not_committed")
        subprocess.run(["git",
            "diff", "--quiet", "HEAD", "--", *runtime_files], cwd=root, check=True,
            capture_output=True)
    except ValueError:
        raise
    except subprocess.CalledProcessError as exc:
        raise ValueError("m5_runtime_file_not_committed") from exc
    except OSError as exc:
        raise ValueError("m5_runtime_git_status_unavailable") from exc
    return digests


def rebind_m5_inputs_v5(root: Path, archive: Path) -> Path:
    """Copy fixed v4 case bytes into a new v5 archive with complete package fingerprints."""
    import shutil

    old, old_archive, old_cases = _verified_v4_registration(root)
    target, traces = archive.resolve(), (root / "traces/review-packages").resolve()
    if target.exists() or not target.is_relative_to(traces) or target == traces:
        raise ValueError("m5_v4_archive_path_invalid_or_exists")
    _verify_committed_runtime(root)
    for row in old_cases:
        folder = old_archive / f"case-{row['order']:02d}"
        _verify_v4_case_bytes(root, folder, row)
    target.mkdir(parents=True)
    for row in old_cases:
        source, dest = old_archive / f"case-{row['order']:02d}", target / f"case-{row['order']:02d}"
        shutil.copytree(source, dest)
    selection_sha = _sha((root / SELECTION).read_bytes())
    manifest = {"version": ARCHIVE_VERSION, "selection_sha256": selection_sha,
        "sources": _source_shas(root), "cases": old_cases}
    raw = _json_bytes(manifest) + b"\n"
    (target / "inputs.json").write_bytes(raw)
    summary = {"archive": target.relative_to(root).as_posix(),
        "archive_sha256": _sha(raw), "selection_sha256": selection_sha,
        "source_sha256": manifest["sources"], "cases": old_cases}
    summary["archive_summary_sha256"] = _sha(_json_bytes(summary) + b"\n")
    path = root / SUMMARY
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return path


def _verified_v4_registration(root: Path) -> tuple[dict, Path, list[dict]]:
    """Verify v4 registration and source bytes before creating an immutable v5 copy."""
    old = json.loads((root / SOURCE_SUMMARY).read_text(encoding="utf-8"))
    unsigned = dict(old)
    registered_sha = unsigned.pop("archive_summary_sha256", None)
    if _sha(_json_bytes(unsigned) + b"\n") != registered_sha:
        raise ValueError("m5_v4_summary_fingerprint_mismatch")
    selection = (root / SELECTION).read_bytes()
    if _sha(selection) != old.get("selection_sha256"):
        raise ValueError("m5_v4_selection_changed")
    archive = (root / old["archive"]).resolve()
    raw = (archive / "inputs.json").read_bytes()
    manifest = json.loads(raw)
    if (_sha(raw) != old.get("archive_sha256") or manifest.get("version") != "m5-inputs-v4"
            or manifest.get("sources") != old.get("source_sha256")
            or manifest.get("cases") != old.get("cases")):
        raise ValueError("m5_v4_archive_fingerprint_mismatch")
    for name, digest in old["source_sha256"].items():
        path = root / name
        if name != "experiments/archive/m5_inputs.py" and (
                not path.is_file() or _sha(path.read_bytes()) != digest):
            raise ValueError("m5_v4_source_changed:" + name)
    return old, archive, old["cases"]


def _verify_v4_case_bytes(root: Path, folder: Path, row: dict[str, object]) -> None:
    """Verify every archived sample, evidence artifact and selected CED source before copy."""
    expected = {"sample.json": row["sample_sha256"],
        "initial-evidence.json": row.get("initial_sha256"),
        "initial-evidence-audit.json": row.get("initial_audit_sha256")}
    for name, digest in expected.items():
        path = folder / name
        if (digest is None) != (not path.exists()) or (path.exists() and _sha(path.read_bytes()) != digest):
            raise ValueError("m5_v4_case_byte_mismatch:" + str(row["sample_id"]))
    if str(row["sample_id"]).startswith("ced:"):
        from src.yuqing.settings import get_settings

        sample_id = str(row["sample_id"])
        current = _ced_source_shas(get_settings().ced_dataset_path,
            sample_id.rsplit(":", 1)[1] + ".json", ":rumor:" in sample_id)
        if current != row["source_sha256"]:
            raise ValueError("m5_v4_ced_source_changed:" + sample_id)


def _questions(root: Path) -> dict[str, object]:
    """Index raw CHEF dev claims; Gold is carried only for the retrieval input type."""
    from src.yuqing.eval.dataset import RetrievalQuestion
    from src.yuqing.ingest.chef import load_chef

    rows = load_chef(root / "data/CHEF/dev.json")
    indexed = {row.sample_id: RetrievalQuestion(sample_id=row.sample_id, claim=row.claim,
        gold_veracity=row.gold_veracity) for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("m5_chef_question_identity_duplicate")
    return indexed


def _initial_rows(root: Path) -> dict[str, object]:
    """Read and validate each frozen initial-evidence line without changing it."""
    from src.yuqing.harness.initial_evidence import InitialEvidenceManifestV2

    rows = {}
    for line in (root / "eval/dev90_initial_evidence_v2.jsonl").read_bytes().splitlines():
        item = InitialEvidenceManifestV2.model_validate_json(line, strict=True)
        if item.sample_id in rows:
            raise ValueError("m5_initial_manifest_identity_duplicate")
        rows[item.sample_id] = item
    return rows


def _make_missing_initial(root: Path, sample_ids: set[str]) -> dict[str, tuple[object, dict[str, object]]]:
    """Prepare only the two missing CHEF cases with the existing full-corpus search path."""
    from scripts.prepare_fact_v2 import (_limits_fingerprint, _prepare_one_initial,
        _retrieval_fingerprints)
    from src.yuqing.eval.fact_v2_experiment import PROMPT_FILES
    from src.yuqing.eval.dataset import RetrievalQuestion
    from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus
    from src.yuqing.retrieval.bm25 import build_bm25_index

    selected = _questions(root)
    questions = [selected[sample_id] for sample_id in sorted(sample_ids)]
    source, store = settings.get_settings().chef_dev_path, settings.get_settings().dev_store_path
    chunks = load_eval_corpus(source, questions, corpus_name="chef_dev_full")
    validate_eval_corpus(chunks, store, corpus_name="chef_dev_full")
    index = build_bm25_index(chunks)
    prompt_text = {role: (root / name).read_text(encoding="utf-8")
        for role, name in PROMPT_FILES.items()}
    retrieval_sha, corpus_sha = _retrieval_fingerprints(root, store)
    limits_sha = _limits_fingerprint(root, prompt_text["single"], prompt_text["loop"])
    prepared = {}
    for question in questions:
        manifest, audit = _prepare_one_initial(question, index, store,
            prompt_text["single"], prompt_text["loop"], retrieval_sha, corpus_sha, limits_sha)
        if manifest.retrieval_status != "success" or not manifest.entries or audit["capacity_exceeded"]:
            raise ValueError("m5_initial_evidence_preparation_failed:" + question.sample_id)
        prepared[question.sample_id] = (manifest, audit)
    return prepared


def prepare_m5_inputs(root: Path, archive: Path) -> Path:
    """Create a new D-drive archive for the ten selected cases; never overwrite."""
    selection_path, selection, target, source_shas, questions, initial, initial_audit, prepared = _prepare_sources(root, archive)
    target.mkdir(parents=True)
    rows = [_write_case_input(root, target, item, source_shas, questions, initial, initial_audit, prepared)
            for item in selection["samples"]]
    return _write_input_manifests(root, target, selection_path, rows, source_shas)


def _prepare_sources(root: Path, archive: Path) -> tuple[Path, dict, Path, dict, dict, dict, dict, dict]:
    """Validate frozen inputs and prepare only the two missing CHEF initial-evidence rows."""
    selection_path = root / SELECTION
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if _sha(selection_path.read_bytes()) != "4e79796fe1916480f306e4a01af4bee3c70bc800e6b2dae63751c8d39171ea76":
        raise ValueError("m5_selection_manifest_changed")
    target, traces = archive.resolve(), (root / "traces/review-packages").resolve()
    if target == traces or not target.is_relative_to(traces) or target.exists():
        raise ValueError("m5_input_archive_path_invalid_or_exists")
    if any((root / name).exists() for name in
            (SUMMARY, SOURCE_SUMMARY, V3_SUMMARY, PREVIOUS_SUMMARY, LEGACY_SUMMARY)):
        raise ValueError("m5_input_summary_already_exists")
    questions, initial = _questions(root), _initial_rows(root)
    initial_audit = _initial_audit_rows(root)
    missing = {item["sample_id"] for item in selection["samples"][:5]} - initial.keys()
    if missing != {"chef:dev:260", "chef:dev:484"}:
        raise ValueError("m5_frozen_initial_evidence_coverage_changed")
    return (selection_path, selection, target, _source_shas(root), questions, initial,
        initial_audit, _make_missing_initial(root, missing))


def _initial_audit_rows(root: Path) -> dict[str, dict[str, object]]:
    """Index the frozen dev90 retrieval audit without modifying its original bytes."""
    path = root / "eval/dev90_initial_evidence_v2_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    indexed = {str(row["sample_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("m5_initial_audit_identity_duplicate")
    return indexed


def _write_case_input(root: Path, target: Path, item: dict[str, object], source_shas: dict,
                      questions: dict, initial: dict, initial_audit: dict,
                      prepared: dict) -> dict[str, object]:
    """Write one selected input without its Gold label and retain source hashes."""
    order, sample_id = item["order"], item["sample_id"]
    folder = target / f"case-{order:02d}"
    folder.mkdir()
    initial_sha, initial_path, audit_sha = None, None, None
    if sample_id.startswith("chef:"):
        sample = Sample(sample_id=sample_id, source="chef", claim=questions[sample_id].claim)
        if sample_id in initial:
            manifest, audit = initial[sample_id], initial_audit[sample_id]
        else:
            manifest, audit = prepared[sample_id]
        raw = manifest.model_dump_json().encode("utf-8")
        initial_path = folder / "initial-evidence.json"
        initial_path.write_bytes(raw)
        initial_sha = _sha(raw)
        audit_raw = json.dumps(audit, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        (folder / "initial-evidence-audit.json").write_bytes(audit_raw)
        audit_sha = _sha(audit_raw)
        coverage = settings.get_evidence_sources_config()["chef_dev"].coverage
        sources = {"chef_dev": source_shas["data/CHEF/dev.json"], "manifest": initial_sha}
    else:
        sample, coverage, sources = _ced_input(sample_id)
    sample_bytes = sample.model_dump_json().encode("utf-8")
    (folder / "sample.json").write_bytes(sample_bytes)
    return {"order": order, "sample_id": sample_id, "coverage": coverage,
        "sample_sha256": _sha(sample_bytes), "initial_sha256": initial_sha,
        "initial_audit_sha256": audit_sha,
        "source_sha256": sources}


def _ced_input(sample_id: str) -> tuple[Sample, str, dict[str, str]]:
    """Read CED original plus repost data and strip dataset truth from model-facing input."""
    from src.yuqing.ingest.ced import load_ced_sample

    rumor = ":rumor:" in sample_id
    filename = sample_id.rsplit(":", 1)[1] + ".json"
    source_root = settings.get_settings().ced_dataset_path
    sample = load_ced_sample(source_root, filename, rumor=rumor).model_copy(
        update={"gold_veracity": None, "evidence_ids": []})
    return sample, "none", _ced_source_shas(source_root, filename, rumor)


def _write_input_manifests(root: Path, target: Path, selection_path: Path,
                           rows: list[dict[str, object]], source_shas: dict) -> Path:
    """Write archive and portable summary with exclusive creation semantics."""
    raw = {"version": ARCHIVE_VERSION, "selection_sha256": _sha(selection_path.read_bytes()),
        "sources": source_shas, "cases": rows}
    payload = _json_bytes(raw) + b"\n"
    (target / "inputs.json").write_bytes(payload)
    summary = {"archive": target.relative_to(root).as_posix(), "archive_sha256": _sha(payload),
        "selection_sha256": raw["selection_sha256"], "source_sha256": source_shas, "cases": rows}
    summary["archive_summary_sha256"] = _sha(_json_bytes(summary) + b"\n")
    summary_path = root / SUMMARY
    with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return summary_path


def _ced_source_shas(root: Path, filename: str, rumor: bool) -> dict[str, str]:
    """Hash only the selected CED original and its corresponding repost tree."""
    suffix = "rumor-repost" if rumor else "non-rumor-repost"
    paths = [root / "original-microblog" / filename, root / suffix / filename]
    if any(not path.is_file() for path in paths):
        raise ValueError("m5_ced_source_missing:" + filename)
    return {str(path.relative_to(root).as_posix()): _sha(path.read_bytes()) for path in paths}


def load_m5_inputs(root: Path) -> M5Inputs:
    """Verify the fixed manifest, archive bytes, sample identities and all source hashes."""
    path = root / SUMMARY
    summary_raw = path.read_bytes()
    summary = json.loads(summary_raw)
    unsigned = dict(summary)
    registration_sha = unsigned.pop("archive_summary_sha256", None)
    if _sha(_json_bytes(unsigned) + b"\n") != registration_sha:
        raise ValueError("m5_input_summary_changed")
    archive = (root / summary["archive"]).resolve()
    traces = (root / "traces/review-packages").resolve()
    if archive == traces or not archive.is_relative_to(traces) or _sha((archive / "inputs.json").read_bytes()) != summary["archive_sha256"]:
        raise ValueError("m5_input_archive_fingerprint_mismatch")
    archive_manifest = json.loads((archive / "inputs.json").read_text(encoding="utf-8"))
    if archive_manifest.get("version") != ARCHIVE_VERSION:
        raise ValueError("m5_input_archive_version_invalid")
    selection_bytes = (root / SELECTION).read_bytes()
    selection = json.loads(selection_bytes)
    if summary["selection_sha256"] != _sha(selection_bytes):
        raise ValueError("m5_selection_fingerprint_mismatch")
    if (archive_manifest.get("selection_sha256") != summary["selection_sha256"]
            or archive_manifest.get("sources") != summary["source_sha256"]
            or archive_manifest.get("cases") != summary["cases"]):
        raise ValueError("m5_input_summary_archive_mismatch")
    if summary["source_sha256"] != _source_shas(root):
        raise ValueError("m5_source_conditions_changed")
    cases, questions = [], _questions(root)
    for expected, row in zip(selection["samples"], summary["cases"], strict=True):
        if (expected["order"], expected["sample_id"]) != (row["order"], row["sample_id"]):
            raise ValueError("m5_fixed_sample_order_mismatch")
        cases.append(_load_archived_case(root, archive, expected, row, questions))
    if len(cases) != 10 or len({item.sample.sample_id for item in cases}) != 10:
        raise ValueError("m5_fixed_ten_invalid")
    return M5Inputs(archive, summary["archive_sha256"], tuple(cases), summary["source_sha256"])


def _load_archived_case(root: Path, archive: Path, expected: dict[str, object],
                        row: dict[str, object], questions: dict[str, object]) -> M5CaseInput:
    """Verify one selected sample, its source-specific coverage and its initial-evidence bundle."""
    from src.yuqing.harness.initial_evidence import load_initial_evidence_manifest

    folder = archive / f"case-{row['order']:02d}"
    sample_path = folder / "sample.json"
    sample_bytes = sample_path.read_bytes()
    if _sha(sample_bytes) != row["sample_sha256"]:
        raise ValueError("m5_sample_artifact_changed:" + row["sample_id"])
    sample = Sample.model_validate_json(sample_bytes, strict=True)
    if sample.sample_id != row["sample_id"] or sample.gold_veracity is not None:
        raise ValueError("m5_sample_identity_or_gold_leak")
    _validate_case_source(expected, row, sample, questions, root, archive)
    initial_path = folder / "initial-evidence.json" if row["coverage"] != "none" else None
    if initial_path is not None:
        if _sha(initial_path.read_bytes()) != row["initial_sha256"]:
            raise ValueError("m5_initial_artifact_changed:" + row["sample_id"])
        load_initial_evidence_manifest(initial_path, sample.sample_id, sample.claim)
        audit_path = folder / "initial-evidence-audit.json"
        if _sha(audit_path.read_bytes()) != row["initial_audit_sha256"]:
            raise ValueError("m5_initial_audit_changed:" + row["sample_id"])
    return M5CaseInput(row["order"], sample, row["coverage"], row["sample_sha256"],
        initial_path, row["initial_sha256"])


def _validate_case_source(expected: dict[str, object], row: dict[str, object], sample: Sample,
                          questions: dict[str, object], root: Path, archive: Path) -> None:
    """Tie the selected identity to CHEF claims or CED original/repost source bytes."""
    if expected["source"] == "CHEF":
        question = questions.get(sample.sample_id)
        if (sample.source != "chef" or question is None or sample.claim != question.claim
                or _sha(sample.claim.encode("utf-8")) != expected["claim_sha256"]
                or row["coverage"] != settings.get_evidence_sources_config()["chef_dev"].coverage):
            raise ValueError("m5_chef_claim_or_coverage_mismatch:" + sample.sample_id)
        return
    if expected["source"] != "CED" or sample.source != "ced" or row["coverage"] != "none":
        raise ValueError("m5_ced_source_or_coverage_mismatch:" + sample.sample_id)
    _verify_ced_archive_source(root, archive, expected, row)


def _verify_ced_archive_source(root: Path, archive: Path, expected: dict[str, object],
                               row: dict[str, object]) -> None:
    """Refuse a stale archived CED body when either original or repost file changed."""
    from src.yuqing.ingest.ced import load_ced_sample

    sample_id = str(expected["sample_id"])
    filename = sample_id.rsplit(":", 1)[1] + ".json"
    rumor = ":rumor:" in sample_id
    suffix = "rumor-repost" if rumor else "non-rumor-repost"
    source_root = settings.get_settings().ced_dataset_path
    current = _ced_source_shas(source_root, filename, rumor)
    expected_sources = {
        f"original-microblog/{filename}": expected["original_sha256"],
        f"{suffix}/{filename}": expected["repost_sha256"],
    }
    if row["source_sha256"] != expected_sources:
        raise ValueError("m5_ced_manifest_source_mismatch:" + sample_id)
    if current != row["source_sha256"]:
        raise ValueError("m5_ced_source_changed:" + sample_id)
    parsed = load_ced_sample(source_root, filename, rumor=rumor).model_copy(
        update={"gold_veracity": None, "evidence_ids": []})
    archived = Sample.model_validate_json((archive / f"case-{row['order']:02d}" / "sample.json").read_bytes(), strict=True)
    if _sha(parsed.model_dump_json().encode()) != _sha(archived.model_dump_json().encode()):
        raise ValueError("m5_ced_sample_reparse_mismatch:" + sample_id)


