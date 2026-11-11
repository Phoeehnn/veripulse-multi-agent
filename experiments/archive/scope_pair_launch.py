"""固定S/M真实启动许可与全项目单次claim，支持相同门卫的离线验证。"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from src.yuqing import llm
from experiments.archive.scope_pair_runner import ScopePairRunner, proposed_caps, slot_order
from experiments.archive.scope_pair_inputs import initial_messages
from src.yuqing.settings import PROJECT_ROOT, get_settings

def shared_claim_root() -> Path:
    """由Git公共目录定位同一仓库的claim根，worktree不能重新领取。"""
    marker = PROJECT_ROOT / ".git"
    if marker.is_file():
        gitdir = Path(marker.read_text(encoding="utf-8").strip().removeprefix("gitdir: "))
        gitdir = (PROJECT_ROOT / gitdir).resolve()
        common = (gitdir / (gitdir / "commondir").read_text(encoding="utf-8").strip()).resolve()
        return common.parent / "traces/scope-pair-claims"
    return PROJECT_ROOT / "traces/scope-pair-claims"


CLAIMS = shared_claim_root()


class LaunchPermit(BaseModel):
    """一次实验的显式字节许可；旧P1许可不能解析为本合同。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    authorization_id: str
    expires_at: str
    approved_for_dev: bool
    prompts_approved: bool
    prices_approved: bool
    paid_execution_approved: bool
    provider_base_url: str
    provider_model: str
    conditions_sha256: str
    max_cost_cny: str
    slots: int
    max_requests: int
    carry_old_authorization_id: str | None = None


def _check_carry(permit: LaunchPermit) -> None:
    """仅承接作者列名的DNS失败场；哈希不符时拒绝，不改原账本。"""
    if permit.carry_old_authorization_id is None:
        return
    if permit.carry_old_authorization_id != "785b35bfdb964d229bff432724c8c8a0":
        raise ValueError("scope_carry_not_authorized")
    files = {
        "traces/sm-live-20261105-785b35bf/wallet.sqlite": "b11a475cb3df930c882a1d373746956c95c4b23a629399ff210bcbbc03ffb8f8",
        "traces/sm-live-20261105-785b35bf/summary.json": "0e08b8a4bb13b09f53632f826f5860d461aeb6a260c10ef6888b72760d8292e0",
        "traces/scope-pair-claims/785b35bfdb964d229bff432724c8c8a0.json": "1b5eb8ff9781d366e9cdadef2a0b94b2d00132a560599a8ab6fed7c46deb305c"}
    for path, expected in files.items():
        if sha256((PROJECT_ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError("scope_carry_history_drift")


def conditions_sha(runner: ScopePairRunner) -> str:
    """绑定实际参数、价格、材料/Schema/Prompt、工具与启动器代码。"""
    payload = [runner._runtime_fingerprint(), list(runner.inputs.fingerprints), slot_order(runner.inputs),
        sha256(Path(__file__).read_bytes()).hexdigest()]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def preflight(runner: ScopePairRunner, permit: LaunchPermit) -> None:
    """只读检查，失败不会领取claim或建立真实客户端。"""
    if not isinstance(permit, LaunchPermit):
        raise ValueError("scope_permission_missing")
    runner._check()
    _check_carry(permit)
    if not all((permit.approved_for_dev, permit.prompts_approved, permit.prices_approved, permit.paid_execution_approved)):
        raise ValueError("scope_permission_missing")
    if UUID(permit.authorization_id).hex != permit.authorization_id:
        raise ValueError("scope_authorization_id_invalid")
    expiry = datetime.fromisoformat(permit.expires_at)
    if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
        raise ValueError("scope_permission_expired")
    settings = get_settings()
    if (permit.provider_base_url != "https://api.deepseek.com" or permit.provider_model != "deepseek-flash"
            or settings.deepseek_base_url.rstrip("/") != permit.provider_base_url
            or settings.deepseek_model != permit.provider_model or settings.llm_thinking):
        raise ValueError("scope_destination_mismatch")
    if (permit.conditions_sha256 != conditions_sha(runner) or runner.caps != proposed_caps()
            or permit.max_cost_cny != "8.64" or permit.slots != 24 or permit.max_requests != 48):
        raise ValueError("scope_conditions_mismatch")
    definitions = llm._tool_definitions(runner.tools, True)
    for packet in runner.inputs.packets:
        bound = llm.estimate_request_input(initial_messages(runner.inputs, packet), definitions,
            runner.config.input_bound_margin_tokens)
        if bound > runner.config.token_budget_per_call:
            raise ValueError("scope_initial_capacity_exceeded")


def launch(runner: ScopePairRunner, root: Path, permit: LaunchPermit,
           *, replay: llm.OfflineCompletionReplay | None = None) -> dict[str, object]:
    """原子领取一次许可；未确认场次阻止后续启动，故障不会释放额度。"""
    preflight(runner, permit)
    if replay is not None and type(replay) is not llm.OfflineCompletionReplay:
        raise TypeError("invalid_replay")
    if replay is None and (llm._offline_replay.get() is not None or get_settings().deepseek_api_key is None):
        raise ValueError("scope_live_environment_invalid")
    if not root.resolve().is_relative_to((PROJECT_ROOT / "traces").resolve()) or root.exists():
        raise ValueError("scope_output_invalid")
    CLAIMS.mkdir(parents=True, exist_ok=True)
    lock = CLAIMS / "active"
    lock.mkdir()
    try:
        return _claimed_launch(runner, root, permit, replay)
    finally:
        lock.rmdir()


def _claimed_launch(runner: ScopePairRunner, root: Path, permit: LaunchPermit,
                    replay: llm.OfflineCompletionReplay | None) -> dict[str, object]:
    """锁内检查旧场状态并独占写claim，执行后只发布确定的结算状态。"""
    for previous in CLAIMS.glob("*.json"):
        prior = json.loads(previous.read_bytes())
        if permit.carry_old_authorization_id and prior.get("carry_old_authorization_id") == permit.carry_old_authorization_id:
            raise ValueError("scope_carry_already_used")
        if prior["status"] != "settled" and prior["authorization_id"] != permit.carry_old_authorization_id:
            raise ValueError("scope_unresolved_previous_run")
    claim = CLAIMS / f"{permit.authorization_id}.json"
    record = {"authorization_id": permit.authorization_id, "status": "claimed", "root": str(root.resolve()),
        "conditions_sha256": permit.conditions_sha256, "mode": "mock" if replay else "live",
        "carry_old_authorization_id": permit.carry_old_authorization_id}
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(record, stream)
    runner.launch_check = lambda: _check_permit(runner, permit)
    try:
        output = runner._run(root, replay)
        clean = not output["execution_error"] and not output["audit_issues"] and not output["wallet"]["halted"]
        if clean and output["wallet"]["cost_cny"] is not None:
            record["status"] = "settled"
            claim.write_text(json.dumps(record), encoding="utf-8")
        return output
    finally:
        del runner.launch_check


def _check_permit(runner: ScopePairRunner, permit: LaunchPermit) -> None:
    """每次请求前复查许可条件，避免递归调用Runner检查。"""
    _check_carry(permit)
    if permit.conditions_sha256 != conditions_sha(runner) or datetime.fromisoformat(permit.expires_at) <= datetime.now(timezone.utc):
        raise ValueError("scope_permission_drift_or_expired")
