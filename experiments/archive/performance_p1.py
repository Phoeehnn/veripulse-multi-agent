"""P1离线原型与真实传播校验链的Provider关闭Mock Runner。"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from src.yuqing import llm
from src.yuqing.eval.experiment_session import ExperimentSession
from src.yuqing.harness.tools import EvidenceWorkerScope
from src.yuqing.harness.trace import RunContext
from src.yuqing.schemas import AgentSpec, Sample
from src.yuqing.settings import (CostConfig, EvaluationConfig, PropagationHarnessConfig,
    PROJECT_ROOT, get_cost_config, get_evaluation_config, get_propagation_harness_config)

P1_SELECTION_SHA256 = "b5cbd9a882639bdc568e173f969d2afb203d838e1af8bec8540b43adbe73f1f5"
P1_PROMPT_SHA256 = "715cc189f1bc1f24e1fb573c6d52244ecec52018af4511ed1d9cf5dd936b3d02"
P1_OBSERVATION_SHA256 = "0a50e8e46df3ba34736a1ceca98f90518445dad8fc79fe1a79d2e966c31c7b40"


@dataclass(frozen=True)
class P1GroupProfile:
    """A/B/C实验档案；金额仅作用于显式离线P1入口。"""

    group: Literal["A", "B", "C"]
    slot_cap_cny: Decimal
    correction_enabled: bool


P1_GROUPS = {
    "A": P1GroupProfile("A", Decimal("0.08"), False),
    "B": P1GroupProfile("B", Decimal("0.12"), False),
    "C": P1GroupProfile("C", Decimal("0.12"), True),
}


class P1SelectionItem(BaseModel):
    """冻结开发题单中的一个CED样本与三源指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    category: Literal["rumor", "non_rumor"]
    category_rank_order: int
    input_sha256: str
    original_sha256: str
    propagation_sha256: str
    order: tuple[Literal["A", "B", "C"], ...]


@dataclass(frozen=True)
class P1MockRun:
    """一条真实Agent校验链的Mock执行记录，不提供准确率或语义质量判定。"""

    group: Literal["A", "B", "C"]
    sample_id: str
    qualified_submission: bool
    program_validation_passed: bool
    citation_validation: Literal["passed", "failed", "not_reached"]
    requests: int
    responses_supplied: int
    responses_consumed: int
    request_tools: tuple[tuple[str, ...], ...]
    cost_cny: float | None
    elapsed_s: float
    stop_reason: str
    trace_path: Path
    output_path: Path
    paid_ready: bool = False


class P1OfflineReplay(llm.OfflineCompletionReplay):
    """仅接受内存中的ChatCompletion回执，并记录完整生产网关请求。"""

    def __init__(self, responses: Sequence[object]) -> None:
        """校验响应对象类型后建立不可联网的有限回放队列。"""
        self.requests: list[dict[str, object]] = []
        self.responses_supplied = len(responses)
        super().__init__(responses)

    def create(self, **request: object) -> object:
        """保存请求体后消费下一条本地回执，队列耗尽时绝不联网回退。"""
        self.requests.append(request)
        return super().create(**request)


@dataclass
class P1OfflineSession:
    """持有冻结条件、共享持久钱包和三个原子组额度的离线场次。"""

    session: ExperimentSession
    selection: tuple[P1SelectionItem, ...]
    prompt: str
    group_caps: Mapping[str, Decimal]
    experiment_root: Path
    conditions: Mapping[str, str]


@dataclass
class OfflineWallet:
    """离线模拟单场钱包，未知费用保留预占并永久停止。"""

    cap: Decimal
    spent: Decimal = Decimal("0")
    pending: Decimal = Decimal("0")
    halted: bool = False

    def reserve(self, amount: Decimal) -> None:
        """调用前预占，拒绝超额、并行未结算及非正数。"""
        if self.halted or self.pending or amount <= 0 or self.spent + amount > self.cap:
            raise ValueError("wallet_denied")
        self.pending = amount

    def settle(self, actual: Decimal | None) -> None:
        """已知费用替换预占，未知或超预占保留额度并停场。"""
        if not self.pending:
            raise ValueError("no_pending")
        if actual is None or actual < 0 or actual > self.pending:
            self.halted = True
            raise ValueError("usage_unknown_or_exceeded")
        self.spent += actual
        self.pending = Decimal("0")


@dataclass
class OfflineSlot:
    """模拟三组阶段及双层费用边界；结果由外部Mock校验器给出。"""

    group: Literal["A", "B", "C"]
    wallet: OfflineWallet
    spent: Decimal = Decimal("0")
    ordinary: int = 0
    closing: int = 0
    terminal: bool = False
    pending: Decimal = Decimal("0")
    phase: str = "ordinary"
    events: list[dict[str, object]] = field(default_factory=list)

    def request(self, phase: str, tools: tuple[str, ...], bound: Decimal) -> None:
        """执行前限制阶段、工具、调用次数及单案和整场预留。"""
        cap = Decimal("0.08") if self.group == "A" else Decimal("0.12")
        close_count = 2 if self.group == "C" else 1
        if self.terminal or self.pending or bound <= 0 or bound > Decimal("0.036"):
            raise ValueError("slot_denied")
        if phase == "ordinary":
            if self.closing or self.ordinary >= 4:
                raise ValueError("ordinary_closed")
            reserve = Decimal("0.036") * close_count
        elif phase == "submit":
            if self.closing >= close_count or tools != ("submit_propagation_report",):
                raise ValueError("submission_only")
            reserve = Decimal("0.036") * (close_count - self.closing - 1)
        else:
            raise ValueError("invalid_phase")
        if self.spent + bound + reserve > cap:
            raise ValueError("node_cost_limit")
        self.wallet.reserve(bound)
        self.pending, self.phase = bound, phase
        self.ordinary += int(phase == "ordinary")
        self.closing += int(phase == "submit")

    def finish(self, actual: Decimal | None, *, accepted: bool) -> None:
        """只记录Mock校验结论；合格即停，C退卷最多再提交一次。"""
        if not self.pending:
            raise ValueError("no_pending")
        self.wallet.settle(actual)
        assert actual is not None
        self.spent += actual
        self.pending = Decimal("0")
        self.terminal = accepted or (self.closing >= (2 if self.group == "C" else 1))
        self.events.append({"phase": self.phase, "accepted": accepted,
                            "cost_cny": str(actual), "mock_only": True})


def run_offline(group: Literal["A", "B", "C"], wallet: OfflineWallet,
                responses: tuple[tuple[Decimal, bool], ...]) -> OfflineSlot:
    """用预先给定Mock回执演练提交及纠错，禁止隐式付费回调。"""
    if group not in {"A", "B", "C"}:
        raise ValueError("invalid_group")
    slot = OfflineSlot(group, wallet)
    for actual, accepted in responses:
        slot.request("submit", ("submit_propagation_report",), Decimal("0.036"))
        slot.finish(actual, accepted=accepted)
    return slot


def group_profile(group: str) -> P1GroupProfile:
    """返回唯一登记的P1组档案，不允许调用方拼造费用或纠错能力。"""
    try:
        return P1_GROUPS[group]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown_p1_group") from exc


def validate_group_profile(profile: object, config: PropagationHarnessConfig) -> P1GroupProfile:
    """核对Agent档案、费用数值和C组唯一纠错资格。"""
    if type(profile) is not P1GroupProfile or group_profile(profile.group) != profile:
        raise ValueError("invalid_p1_profile")
    if config.max_cost_cny != float(profile.slot_cap_cny):
        raise ValueError("p1_profile_cost_mismatch")
    return profile


def config_for_group(base: PropagationHarnessConfig, profile: P1GroupProfile) -> PropagationHarnessConfig:
    """仅复制传播单案费用上限，其余默认步数、工具、token配置保持相同。"""
    validate_group_profile(profile, base.model_copy(update={"max_cost_cny": float(profile.slot_cap_cny)}))
    values = base.model_dump() | {"max_cost_cny": float(profile.slot_cap_cny)}
    return PropagationHarnessConfig.model_validate(values)


def load_frozen_p1_selection(root: Path = PROJECT_ROOT) -> tuple[P1SelectionItem, ...]:
    """核验固定题单文件、分层、三组顺序及不可付费标记。"""
    path = root / "docs/cards/P1_selection.json"
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != P1_SELECTION_SHA256:
        raise ValueError("p1_selection_sha_drift")
    data = json.loads(raw)
    items = tuple(P1SelectionItem.model_validate(row) for row in data.get("selected", []))
    _validate_selection_envelope(data, items)
    return items


def _validate_selection_envelope(data: object, items: tuple[P1SelectionItem, ...]) -> None:
    """检查冻结清单的样本数、来源平衡、执行顺序和许可字段。"""
    conditions = data.get("experiment_conditions") if isinstance(data, dict) else None
    valid = (isinstance(data, dict) and data.get("status") == "frozen_offline"
        and data.get("version") == "p1-propagation-dev12-v1" and len(items) == 12
        and len({item.sample_id for item in items}) == len(items)
        and sum(item.category == "rumor" for item in items) == 6
        and sum(item.category == "non_rumor" for item in items) == 6
        and isinstance(conditions, dict) and conditions.get("provider_calls_authorized") is False
        and conditions.get("paid_execution_authorized") is False
        and conditions.get("prompt_sha256") == P1_PROMPT_SHA256)
    orders = [item.order for item in items]
    if not valid or any(orders.count(order) != 4 for order in (("A", "B", "C"),
            ("B", "C", "A"), ("C", "A", "B"))):
        raise ValueError("p1_selection_contract_invalid")
    if any(sorted(item.order) != ["A", "B", "C"] for item in items):
        raise ValueError("p1_selection_order_invalid")


def p1_group_caps(selection: Sequence[P1SelectionItem]) -> Mapping[str, Decimal]:
    """从冻结槽数推导各组额度，三组之和就是整场上限。"""
    if not selection or len({item.sample_id for item in selection}) != len(selection):
        raise ValueError("p1_selection_not_unique")
    from types import MappingProxyType

    return MappingProxyType({name: profile.slot_cap_cny * len(selection)
                             for name, profile in P1_GROUPS.items()})


def load_frozen_p1_samples(root: Path = PROJECT_ROOT) -> tuple[tuple[P1SelectionItem, ...], tuple[Sample, ...]]:
    """重算12案的原帖、传播树和真实Agent输入SHA后才交付样本。"""
    from src.yuqing.eval.propagation_eval import _input_fingerprint
    from src.yuqing.ingest.ced import load_ced_sample

    items, samples = load_frozen_p1_selection(root), []
    data_root = root / "data/raw/CED_Dataset"
    for item in items:
        sample = _load_p1_sample(data_root, item, _input_fingerprint, load_ced_sample)
        samples.append(sample)
    return items, tuple(samples)


def _load_p1_sample(data_root: Path, item: P1SelectionItem,
                    fingerprint: Callable[[Sample], str],
                    loader: Callable[..., Sample]) -> Sample:
    """逐源核对冻结SHA并使用现有CED解析器建立样本。"""
    stem = item.sample_id.rsplit(":", 1)[-1]
    if not stem or Path(stem).name != stem:
        raise ValueError("p1_sample_path_invalid")
    original = data_root / "original-microblog" / f"{stem}.json"
    folder = "rumor-repost" if item.category == "rumor" else "non-rumor-repost"
    propagation = data_root / folder / f"{stem}.json"
    if _sha_file(original) != item.original_sha256 or _sha_file(propagation) != item.propagation_sha256:
        raise ValueError("p1_sample_source_sha_drift")
    sample = loader(data_root, f"{stem}.json", rumor=item.category == "rumor")
    if sample.sample_id != item.sample_id or fingerprint(sample) != item.input_sha256:
        raise ValueError("p1_sample_input_sha_drift")
    return sample


def _sha_file(path: Path) -> str:
    """计算冻结CED源文件的SHA-256。"""
    return sha256(path.read_bytes()).hexdigest()


def observation_as_of(sample: Sample) -> datetime:
    """验证冻结全历史观察时间与预期材料量，拒绝发布时间截断或输入漂移。"""
    from src.yuqing.eval.propagation_eval import _input_fingerprint
    from src.yuqing.features.propagation import get_spread_stats

    path = PROJECT_ROOT / "docs/cards/P1-observation-v2.json"
    if _sha_file(path) != P1_OBSERVATION_SHA256:
        raise ValueError("p1_observation_manifest_drift")
    manifest = json.loads(path.read_bytes())
    rows = [row for row in manifest["cases"] if row["sample_id"] == sample.sample_id]
    if len(rows) != 1 or rows[0]["input_sha256"] != _input_fingerprint(sample):
        raise ValueError("p1_observation_input_drift")
    row, raw = rows[0], sample.propagation
    cutoff = datetime.fromisoformat(row["observed_at"])
    if raw is None or cutoff != max([raw.root_time] + [node.date for node in raw.nodes]):
        raise ValueError("p1_observation_time_mismatch")
    stats = get_spread_stats(raw, as_of=cutoff)
    if (stats.visible_nodes != row["visible_nodes"] or stats.visible_nodes != len(raw.nodes)
            or stats.visible_comments != row["visible_comments"]):
        raise ValueError("p1_observation_material_mismatch")
    return cutoff


def load_frozen_p1_prompt(root: Path = PROJECT_ROOT) -> str:
    """加载既有v1提示词正文；v1.1合同沿用同一提示词字节。"""
    from src.yuqing.eval.preflight import validate_propagation_offline_prompt

    path = root / "prompts/propagation/v1.md"
    prompt, fingerprint = validate_propagation_offline_prompt(path)
    if fingerprint != P1_PROMPT_SHA256:
        raise ValueError("p1_prompt_sha_drift")
    return prompt


def create_p1_offline_session(experiment_root: Path, *, root: Path = PROJECT_ROOT,
                              prices: CostConfig | None = None) -> P1OfflineSession:
    """验证全题单、建新持久钱包并固定Prompt、Schema、代码和组额度指纹。"""
    selection, samples = load_frozen_p1_samples(root)
    for sample in samples:
        observation_as_of(sample)
    prompt = load_frozen_p1_prompt(root)
    caps, shared_cap = _p1_caps_and_total(selection)
    target = _new_experiment_root(experiment_root)
    conditions = _p1_conditions(root, selection, prompt, caps, shared_cap)
    session = _create_p1_wallet(target, conditions, shared_cap, prices or get_cost_config())
    _write_unique(target / "conditions.json", {"conditions": conditions, "paid_ready": False})
    _write_unique(target / "observation.json", json.loads(
        (root / "docs/cards/P1-observation-v2.json").read_bytes()))
    return P1OfflineSession(session, selection, prompt, caps, target, conditions)


def _p1_caps_and_total(selection: Sequence[P1SelectionItem]) -> tuple[Mapping[str, Decimal], Decimal]:
    """计算各组和整场严格相加的单案费用硬上限。"""
    caps = p1_group_caps(selection)
    return caps, sum(caps.values(), Decimal("0"))


def _new_experiment_root(path: Path) -> Path:
    """确保实验输出目录是新建目录且位于项目traces之下。"""
    target, traces = path.resolve(), (PROJECT_ROOT / "traces").resolve()
    if target == traces or not target.is_relative_to(traces):
        raise ValueError("p1_output_outside_project_traces")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _p1_conditions(root: Path, selection: Sequence[P1SelectionItem], prompt: str,
                   caps: Mapping[str, Decimal], shared_cap: Decimal) -> dict[str, str]:
    """固定每个影响组间比较的输入指纹，付费许可始终关闭。"""
    fingerprints = _p1_fingerprints(root, prompt)
    cap_hash = _sha_value({key: str(value) for key, value in caps.items()})
    return {"selection_sha256": P1_SELECTION_SHA256, "prompt_sha256": P1_PROMPT_SHA256,
        "observation_sha256": P1_OBSERVATION_SHA256,
        "runtime_config_sha256": _sha_value({name: config_for_group(
            get_propagation_harness_config(), profile).model_dump(mode="json")
            for name, profile in P1_GROUPS.items()}),
        "weights_sha256": _sha_file(root / "config/weights.yaml"),
        "settings_sha256": _sha_file(root / "src/yuqing/settings.py"),
        "code_sha256": fingerprints["code"], "schema_sha256": fingerprints["schema"],
        "group_caps_sha256": cap_hash, "slots_per_group": str(len(selection)),
        "shared_cap_cny": str(shared_cap), "provider_calls_authorized": "false",
        "paid_execution_authorized": "false", "contract_version": "v1.1"}


def _p1_fingerprints(root: Path, prompt: str) -> dict[str, str]:
    """计算当前代码和同一传播报告Schema的指纹，不写或批准Prompt。"""
    from src.yuqing.agents.propagation_v11 import PropagationReportV11
    from src.yuqing.schemas import PropagationReport

    code_paths = ("src/yuqing/agents/propagation.py", "src/yuqing/agents/propagation_v11.py",
        "experiments/archive/performance_p1.py", "src/yuqing/eval/dual_budget.py",
        "src/yuqing/eval/experiment_session.py", "src/yuqing/eval/persistent_experiment_budget.py",
        "src/yuqing/harness/tools.py", "src/yuqing/harness/budget.py", "src/yuqing/llm.py",
        "src/yuqing/harness/feedback.py", "src/yuqing/harness/verifier.py", "src/yuqing/schemas.py")
    code = {name: _sha_file(root / name) for name in code_paths}
    schema = {"report": PropagationReport.model_json_schema(),
              "v1_1": PropagationReportV11.model_json_schema()}
    return {"code": _sha_value(code), "schema": _sha_value(schema),
            "prompt": sha256(prompt.encode("utf-8")).hexdigest()}


def _sha_value(value: object) -> str:
    """对条件结构做稳定JSON序列化并返回SHA-256。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _create_p1_wallet(root: Path, conditions: Mapping[str, str], shared_cap: Decimal,
                       prices: CostConfig) -> ExperimentSession:
    """用现有整场账本创建仅Mock Runner可用的新钱包。"""
    base = get_evaluation_config()
    values = base.model_dump() | {"max_run_cost_cny": float(shared_cap),
                                  "pilot_max_run_cost_cny": float(shared_cap)}
    config = EvaluationConfig.model_validate(values)
    return ExperimentSession.create(root / "wallet.sqlite", "p1-propagation-dev12-v1",
        conditions, prices, config, mode="full", max_run_cost_cny=shared_cap)


def _write_unique(path: Path, value: object) -> None:
    """以独占方式写入可复核JSON，已存在的实验产物不覆盖。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def run_p1_mock_slot(session: P1OfflineSession, item: P1SelectionItem, sample: Sample,
                     group: Literal["A", "B", "C"], replay: P1OfflineReplay,
                     *, isolated_probe: bool = False) -> P1MockRun:
    """经真实Propagation Agent和网关执行一槽本地Mock，不存在Provider回退。"""
    profile = group_profile(group)
    _validate_mock_slot(session, item, sample, profile, replay)
    _bind_runner_mode(session, isolated_probe)
    claims = session.experiment_root / "claims" / ("isolated" if isolated_probe else "formal")
    claims.mkdir(parents=True, exist_ok=True)
    if not isolated_probe:
        plan = [(row.sample_id, name) for row in session.selection for name in row.order]
        completed = len(list(claims.glob("*.json")))
        if completed >= len(plan) or plan[completed] != (item.sample_id, group):
            raise ValueError("p1_slot_out_of_order")
    key = sha256(f"{item.sample_id}:{group}".encode("utf-8")).hexdigest()
    _write_unique(claims / f"{key}.json", {"sample_id": item.sample_id,
        "group": group, "isolated_probe": isolated_probe})
    run_id = uuid4().hex
    slot_root = session.experiment_root / "slots" / f"{group}-{run_id}"
    slot_root.mkdir(parents=True, exist_ok=False)
    context = RunContext(run_id=run_id, sample_id=sample.sample_id,
        agent="propagation_analyst", trace_root=slot_root / "traces")
    started = perf_counter()
    try:
        result = _execute_mock_slot(session, profile, sample, context, replay)
        elapsed = perf_counter() - started
        run_data = _mock_run_data(result, context, replay, elapsed, group)
        output = slot_root / "summary.json"
        _write_unique(output, run_data)
        fields = {key: run_data[key] for key in P1MockRun.__dataclass_fields__}
        return P1MockRun(**(fields | {"trace_path": context.trace_path, "output_path": output}))
    except Exception as exc:
        failure_summary = slot_root / "summary.json"
        if not failure_summary.exists():
            _write_unique(failure_summary, {"sample_id": sample.sample_id,
                "group": group, "execution_mode": "offline_mock", "paid_ready": False,
                "status": "execution_failed", "error_type": type(exc).__name__})
        session.session.halt("p1_slot_execution_failed")
        raise


def _bind_runner_mode(session: P1OfflineSession, isolated_probe: bool) -> None:
    """场次只运行正式槽或isolated探针其中一种，拒绝共享额度。"""
    path = session.experiment_root / "runner_mode.json"
    mode = "isolated_probe" if isolated_probe else "formal"
    try:
        _write_unique(path, {"mode": mode})
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("mode") != mode:
            raise ValueError("p1_runner_mode_mismatch")


def _validate_mock_slot(session: P1OfflineSession, item: P1SelectionItem, sample: Sample,
                        profile: P1GroupProfile, replay: P1OfflineReplay) -> None:
    """确认题目、组顺序、提示词、档案和离线回执源完全匹配。"""
    if type(replay) is not P1OfflineReplay or item not in session.selection:
        raise ValueError("p1_mock_input_unregistered")
    if (session.session.persistent_identity[0] != "p1-propagation-dev12-v1"
            or session.session.snapshot().max_cost_cny != sum(session.group_caps.values(), Decimal("0"))
            or dict(session.group_caps) != dict(p1_group_caps(session.selection))):
        raise ValueError("p1_mock_session_mismatch")
    if profile.group not in item.order or sample.sample_id != item.sample_id:
        raise ValueError("p1_mock_slot_identity_mismatch")
    if sha256((PROJECT_ROOT / "docs/cards/P1_selection.json").read_bytes()).hexdigest() != P1_SELECTION_SHA256:
        raise ValueError("p1_selection_sha_drift")
    if _sha_file(PROJECT_ROOT / "prompts/propagation/v1.md") != P1_PROMPT_SHA256:
        raise ValueError("p1_prompt_sha_drift")
    current = _p1_conditions(PROJECT_ROOT, session.selection, session.prompt,
        session.group_caps, sum(session.group_caps.values(), Decimal("0")))
    if dict(current) != dict(session.conditions):
        raise ValueError("p1_mock_condition_drift")
    from src.yuqing.eval.propagation_eval import _input_fingerprint

    if _input_fingerprint(sample) != item.input_sha256 or not _p1_sources_match(item):
        raise ValueError("p1_mock_sample_fingerprint_mismatch")
    if sha256(session.prompt.encode("utf-8")).hexdigest() != P1_PROMPT_SHA256:
        raise ValueError("p1_mock_prompt_changed")


def _p1_sources_match(item: P1SelectionItem) -> bool:
    """每槽重新核对原帖和传播树，防止钱包创建后输入文件漂移。"""
    stem = item.sample_id.rsplit(":", 1)[-1]
    folder = "rumor-repost" if item.category == "rumor" else "non-rumor-repost"
    data_root = PROJECT_ROOT / "data/raw/CED_Dataset"
    return (_sha_file(data_root / "original-microblog" / f"{stem}.json") == item.original_sha256
            and _sha_file(data_root / folder / f"{stem}.json") == item.propagation_sha256)


def _execute_mock_slot(session: P1OfflineSession, profile: P1GroupProfile, sample: Sample,
                       context: RunContext, replay: P1OfflineReplay) -> object:
    """绑定单案配置、同一持久钱包和离线回放，再调用正式传播循环。"""
    from src.yuqing.agents import propagation

    config = config_for_group(get_propagation_harness_config(), profile)
    identity = {"sample_id": sample.sample_id, "run_id": context.run_id,
                "group_id": profile.group, "variant": f"p1-{profile.group}"}
    scope = EvidenceWorkerScope()
    try:
        with llm.use_offline_completion_replay(replay):
            return propagation._run_propagation_p1(AgentSpec(agent="propagation_analyst",
                system_prompt=session.prompt), sample, context, node_config=config,
                worker_scope=scope, p1_profile=profile,
                coordinator_factory=session.session.coordinator_factory(group_caps=session.group_caps),
                experiment_identity=identity)
    finally:
        scope.close()


def _mock_run_data(result: object, context: RunContext, replay: P1OfflineReplay,
                   elapsed: float, group: Literal["A", "B", "C"]) -> dict[str, object]:
    """从Agent结果和唯一run_outcome事件生成不含准确率的Mock记录。"""
    events = [json.loads(line) for line in context.trace_path.read_text(encoding="utf-8").splitlines()]
    outcomes = [event["data"] for event in events if event.get("type") == "run_outcome"]
    outcome = outcomes[0] if len(outcomes) == 1 else {}
    tools = tuple(tuple(item["function"]["name"] for item in request.get("tools", []))
                  for request in replay.requests)
    citation_status = _mock_citation_status(result, events, context)
    return {"group": group, "sample_id": context.sample_id, "qualified_submission": result is not None,
        "program_validation_passed": result is not None, "citation_validation": citation_status,
        "requests": len(replay.requests),
        "responses_supplied": replay.responses_supplied, "responses_consumed": replay.consumed,
        "request_tools": tools, "cost_cny": outcome.get("cost_cny"), "elapsed_s": elapsed,
        "execution_mode": "offline_mock", "cost_basis": "mock_usage_not_provider_billed",
        "stop_reason": str(outcome.get("reason", "run_outcome_missing")), "trace_path": str(context.trace_path),
        "output_path": str(context.trace_path.parent.parent.parent / "summary.json"), "paid_ready": False,
        "content_quality": None, "accuracy": None,
        "report": result.report.model_dump(mode="json") if result is not None else None}


def _mock_citation_status(result: object, events: list[dict[str, object]],
                          context: RunContext) -> Literal["passed", "failed", "not_reached"]:
    """只报告程序引文/依据验证结果，不把它解释成语义准确率。"""
    if result is not None:
        return "passed"
    codes = {"emotion_quote_invalid", "lead_quote_invalid", "narrative_basis_invalid",
             "narrative_basis_required", "spread_basis_invalid", "spread_basis_required"}
    for event in events:
        data = event.get("data", {})
        archive = data.get("archive") if data.get("tool") == "submit_propagation_report" else None
        if not isinstance(archive, dict) or not isinstance(archive.get("path"), str):
            continue
        payload_path = context.trace_root / archive["path"]
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        metadata = payload.get("payload", {}).get("program_metadata", {})
        if codes.intersection(metadata.get("detailed_feedback_issues", [])):
            return "failed"
    return "not_reached"
