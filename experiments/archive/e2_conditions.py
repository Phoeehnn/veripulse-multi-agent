"""固定dev实验输入与非密钥环境，不调用模型、不授予提示词许可。"""

from hashlib import sha256
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
from typing import Sequence

from src.yuqing import llm
from src.yuqing.eval.dataset import RetrievalQuestion, load_questions
from experiments.archive.e2_continuation import ContinuationParent, inspect_continuation_parent, continuation_conditions
from src.yuqing.eval.preflight import validate_runtime_config
from src.yuqing.retrieval.corpus import load_eval_corpus, validate_eval_corpus
from src.yuqing.schemas import FactVerdict, CitationRepairV2, ToolRequest, AgentResult
from src.yuqing.settings import get_settings, get_evaluation_config, get_retrieval_config

logger = logging.getLogger(__name__)


def _json(value: object) -> str:
    """规范JSON只用于条件记录，不隐式转换任意对象。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _files(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    """只登记项目内明确文件的相对路径与原字节指纹，缺失不跳过。"""
    result: dict[str, str] = {}
    for path in paths:
        actual = path.resolve()
        if not actual.is_relative_to(root):
            raise ValueError("condition_file_outside_project")
        result["file:" + actual.relative_to(root).as_posix()] = sha256(actual.read_bytes()).hexdigest()
    return result


def _paths(root: Path, question_paths: Sequence[Path], single_prompt: Path,
            loop_prompt: Path, store: Path) -> list[Path]:
    """枚举全部业务源码与明确输入，不扫描环境变量、秘密或运行产物。"""
    settings = get_settings()
    code = sorted((root / "src/yuqing").rglob("*.py"))
    if not code:
        raise ValueError("missing_experiment_source")
    return [*question_paths, single_prompt, loop_prompt, settings.chef_dev_path, settings.weights_path,
        get_evaluation_config().gold_review_path, root / "prompts/fact_check/contracts.json",
        store / "corpus.json", store / "READY", *code,
        *sorted(root.glob("requirements*.txt")), *sorted(root.glob("pyproject.toml"))]


def _runtime() -> dict[str, str]:
    """显式白名单记录运行字段；密钥和完整环境配置从不序列化。"""
    settings = get_settings()
    fields = ("deepseek_model", "llm_timeout_s", "llm_max_retries", "llm_validation_retries", "llm_thinking")
    data = {name: _json(getattr(settings, name)) for name in fields}
    data["endpoint_sha256"] = sha256(settings.deepseek_base_url.encode("utf-8")).hexdigest()
    data["python"] = platform.python_version()
    data["platform"] = _json([platform.system(), platform.machine()])
    for name in ("openai", "pydantic", "chromadb", "sentence-transformers", "transformers", "torch", "numpy"):
        data["package:" + name] = version(name)
    config = get_retrieval_config()
    for name in ("embedding_model", "rerank_model"):
        folder = settings.model_cache_dir / "huggingface/hub" / ("models--" + getattr(config, name).replace("/", "--"))
        ref = folder / "refs/main"
        data[name + ":revision"] = ref.read_text(encoding="utf-8").strip() if ref.exists() else "missing"
    return data


def _identity(question: RetrievalQuestion) -> tuple[str, str, str]:
    """对齐可更新证据字段，但不得改变冻结题的题号、声明或标准标签。"""
    return question.sample_id, question.claim, question.gold_veracity


def _suffix_conditions(parent: ContinuationParent | None, result: dict[str, str], actual: Sequence[RetrievalQuestion]) -> dict[str, str]:
    """核对完整题目条件；运行提交号与合同版本由入口另行冻结和检查。"""
    if parent is None:
        return result
    original = json.loads(parent.state.conditions["questions"])
    full = result | {"questions": _json(original[:parent.finished_pairs] + [q.model_dump() for q in actual])}
    full.update({k: v for k, v in parent.state.conditions.items() if k in {"code_revision", "loop_contract_version"}})
    return result | continuation_conditions(parent, full)


def collect_e2_conditions(questions: Sequence[RetrievalQuestion], *, question_paths: Sequence[Path],
                          single_prompt: Path, loop_prompt: Path, store_path: Path,
                          pilot_manifest: Path | None = None, continuation_parent: Path | None = None) -> dict[str, str]:
    """固定dev冻结前缀、对齐后题目与输入文件，复用语料校验并检查采集中漂移。"""
    validate_runtime_config()
    settings = get_settings()
    root, store = settings.project_root.resolve(), store_path.resolve()
    expected = [root / "eval/questions_30.jsonl", root / "eval/dev_extra_60.jsonl"]
    if [p.resolve() for p in question_paths] != expected or store != settings.dev_store_path.resolve():
        raise ValueError("conditions_require_dev90_full_corpus")
    if settings.chef_dev_path.stem != "dev" or get_retrieval_config().corpus != "chef_dev_full":
        raise ValueError("conditions_require_dev_split")
    for path in (single_prompt, loop_prompt):
        if not path.resolve().is_relative_to(root / "prompts"):
            raise ValueError("condition_prompt_outside_prompts")
    paths = _paths(root, question_paths, single_prompt, loop_prompt, store)
    extra = [root / "prompts/fact_check/loop_contracts.json", root / "scripts/run_dev_e2.py"]
    paths += [path for path in extra if path.is_file()]
    if pilot_manifest is not None:
        paths.append(pilot_manifest)
    before, runtime = _files(root, paths), _runtime()
    fixed = [q for path in question_paths for q in load_questions(path)]
    actual = [RetrievalQuestion.model_validate(q.model_dump()) for q in questions]
    selected = load_pilot_manifest(pilot_manifest, fixed, question_paths) if pilot_manifest is not None else fixed[:len(actual)]
    parent = inspect_continuation_parent(continuation_parent, [q.sample_id for q in fixed]) if continuation_parent else None
    if parent and pilot_manifest is not None:
        raise ValueError("continuation_not_pilot")
    selected = fixed[parent.finished_pairs:] if parent else selected
    if not actual or len(actual) > len(fixed) or len({q.sample_id for q in fixed}) != len(fixed) or (
        [_identity(q) for q in actual] != [_identity(q) for q in selected]):
        raise ValueError("questions_not_frozen_prefix")
    if not (store / "chroma.sqlite3").is_file():
        raise ValueError("missing_experiment_index")
    chunks = load_eval_corpus(settings.chef_dev_path, actual, corpus_name="chef_dev_full")
    validate_eval_corpus(chunks, store, corpus_name="chef_dev_full")
    single, loop = single_prompt.read_bytes().decode("utf-8"), loop_prompt.read_bytes().decode("utf-8")
    if not single.strip() or not loop.strip():
        raise ValueError("empty_condition_prompt")
    system = {"single_system": llm.build_structured_system(single, FactVerdict),
        "repair_system": llm.build_structured_system(single, CitationRepairV2), "loop_system": loop}
    if before != _files(root, paths) or runtime != _runtime():
        raise ValueError("conditions_changed_during_collection")
    result = before | {"runtime:" + key: value for key, value in runtime.items()} | {
        "split": "dev", "corpus": "chef_dev_full", "questions": _json([q.model_dump() for q in actual]),
        "prompt_roles": _json({"single": single_prompt.resolve().relative_to(root).as_posix(),
            "loop": loop_prompt.resolve().relative_to(root).as_posix()}),
        **{name + "_sha256": sha256(text.encode("utf-8")).hexdigest() for name, text in system.items()},
        "schemas": _json({model.__name__: model.model_json_schema() for model in
            (FactVerdict, CitationRepairV2, ToolRequest, AgentResult)})}
    return _suffix_conditions(parent, result, actual)


def assert_e2_conditions_unchanged(expected: dict[str, str], questions: Sequence[RetrievalQuestion], *,
                                   question_paths: Sequence[Path], single_prompt: Path,
                                   loop_prompt: Path, store_path: Path, pilot_manifest: Path | None = None,
                                   continuation_parent: Path | None = None) -> None:
    """重新采集并比较条件，仅报告变化键，不记录文件正文或敏感值。"""
    current = collect_e2_conditions(questions, question_paths=question_paths, single_prompt=single_prompt,
        loop_prompt=loop_prompt, store_path=store_path, pilot_manifest=pilot_manifest, continuation_parent=continuation_parent)
    changed = sorted(key for key in set(expected) | set(current) if expected.get(key) != current.get(key))
    if changed:
        logger.error("实验条件变化 changed_keys=%s", changed)
        raise ValueError("experiment_conditions_changed: " + ", ".join(changed))


def select_pilot_questions(fixed: Sequence[RetrievalQuestion]) -> list[RetrievalQuestion]:
    """按冻结全局顺序选每类前4/3/3题，数量只取配置规定配置。"""
    counts = get_evaluation_config().fact_check_counts
    remaining, selected = dict(counts), []
    if len({question.sample_id for question in fixed}) != len(fixed):
        raise ValueError("duplicate_pilot_pool_id")
    for question in fixed:
        if remaining[question.gold_veracity] > 0:
            selected.append(question)
            remaining[question.gold_veracity] -= 1
    if any(remaining.values()):
        raise ValueError("pilot_pool_too_small")
    return selected


def _pilot_data(fixed: Sequence[RetrievalQuestion], paths: Sequence[Path]) -> dict[str, object]:
    """清单固定题号、规则和父题文件内容，不带模型成绩。"""
    return {"kind": "manifest", "split": "dev", "selection_rule": "first_per_label_in_frozen_dev90_order",
        "label_counts": get_evaluation_config().fact_check_counts,
        "sample_ids": [q.sample_id for q in select_pilot_questions(fixed)],
        "source_sha256": [sha256(path.read_bytes()).hexdigest() for path in paths]}


def write_pilot_manifest(path: Path, fixed: Sequence[RetrievalQuestion], question_paths: Sequence[Path]) -> Path:
    """独占新增清单；不能覆盖旧清单，也不能写到项目正式结果目录以外。"""
    root = get_settings().project_root.resolve()
    if path.resolve().parent != root / "results":
        raise ValueError("pilot_manifest_outside_results")
    import re
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_manifest_[0-9a-f]{32}\.jsonl", path.name):
        raise ValueError("pilot_manifest_name_invalid")
    data = _pilot_data(fixed, question_paths)
    data["sha256"] = sha256(_json(data).encode()).hexdigest()
    with path.open("xb") as stream:
        stream.write((_json(data) + "\n").encode("utf-8"))
    return path


def load_pilot_manifest(path: Path, fixed: Sequence[RetrievalQuestion], question_paths: Sequence[Path]) -> list[RetrievalQuestion]:
    """重新推导固定规则，只接受完全对应清单，不放开任意子集或test题。"""
    root = get_settings().project_root.resolve()
    if path.resolve().parent != root / "results":
        raise ValueError("pilot_manifest_outside_results")
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or len(raw.splitlines()) != 1:
        raise ValueError("pilot_manifest_incomplete")
    expected = _pilot_data(fixed, question_paths)
    expected["sha256"] = sha256(_json(expected).encode()).hexdigest()
    if raw != (_json(expected) + "\n").encode("utf-8"):
        raise ValueError("pilot_manifest_changed_or_wrong_selection")
    return select_pilot_questions(fixed)
