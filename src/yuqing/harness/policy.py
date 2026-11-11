"""根据程序提供的身份和配置判断工具调用权限，异常时拒绝。"""

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.yuqing.harness.tools import RiskTier, ToolRegistry
from src.yuqing.settings import PolicyConfig

logger = logging.getLogger(__name__)


class PolicyDecision(BaseModel):
    """权限决定及可审计理由；决定本身不会执行任何工具。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    action: Literal["allow", "deny", "ask", "dry_run"]
    reason: str
    risk: RiskTier | None = None
    audit_required: bool = False


def decide(registry: ToolRegistry, agent: str, tool_name: str, *,
           caller: Literal["model", "controller"], config: PolicyConfig) -> PolicyDecision:
    """检查登记、名单、风险和发起身份；身份只由可信程序调用处提供。"""
    try:
        spec = registry.get(tool_name)
        if spec is None:
            return PolicyDecision(action="deny", reason="unknown_tool")
        if caller not in ("model", "controller"):
            return PolicyDecision(action="deny", reason="invalid_caller", risk=spec.risk)
        if tool_name not in config.allow.get(agent, []):
            return PolicyDecision(action="deny", reason="not_allowed", risk=spec.risk)
        if config.risk_tier.get(tool_name) != spec.risk:
            return PolicyDecision(action="deny", reason="risk_mismatch", risk=spec.risk)
        if spec.risk == "R2":
            if caller != "controller" or agent != "alert_gate":
                return PolicyDecision(action="deny", reason="r2_requires_alert_gate", risk=spec.risk)
            if config.r2_mode not in ("ask", "deny", "dry_run"):
                return PolicyDecision(action="deny", reason="invalid_r2_mode", risk=spec.risk)
            return PolicyDecision(action=config.r2_mode, reason="r2_" + config.r2_mode,
                                  risk=spec.risk, audit_required=True)
        return PolicyDecision(action="allow", reason="allowlisted", risk=spec.risk,
                              audit_required=spec.risk == "R1")
    except Exception as exc:
        logger.error("权限检查失败并拒绝：%s", type(exc).__name__)
        return PolicyDecision(action="deny", reason="policy_error", audit_required=True)
