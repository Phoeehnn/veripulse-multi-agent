"""单任务预算在调用前预占、响应后结算，收尾包含在总上限内。"""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError

from src.yuqing.llm import TokenUsage, compute_cost_cny
from src.yuqing.settings import (ArbiterHarnessConfig, CostConfig, HarnessConfig,
    PropagationHarnessConfig, PropagationReviewHarnessConfig)
from src.yuqing.schemas import BudgetStatus, ToolLLMResult
from src.yuqing.harness.trace import require_run_context


class BudgetDecision(BaseModel):
    """预算检查结果与机器可读原因，不负责实际模型或工具执行。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    allowed: bool
    reason: str


@dataclass(frozen=True)
class _ReservationCheckpoint:
    """只保存本次命名预占之前的状态，不包含已发生消费的回滚许可。"""

    request_id: str
    steps: int
    tokens: int
    cost_cny: Decimal
    final_used: bool
    correction_used: bool
    response: tuple[int, str, str] | None
    delivered: frozenset[str]
    submission_call: int
    prices: CostConfig


class BudgetTracker:
    """一位专家的一次任务使用一个实例，预占额度不能重复释放。"""

    def __init__(self, config: HarnessConfig | PropagationHarnessConfig | ArbiterHarnessConfig, prices: CostConfig,
                 *, correction_enabled: bool = False) -> None:
        """复制配置并初始化独立计数，未知价格或不足收尾费用拒绝使用。"""
        if type(correction_enabled) is not bool:
            raise TypeError("invalid_correction_flag")
        if type(config) not in {HarnessConfig, PropagationHarnessConfig,
                PropagationReviewHarnessConfig, ArbiterHarnessConfig}:
            raise TypeError("unsupported_harness_config")
        config_type = type(config)
        self.config = config_type.model_validate(config.model_dump())
        self.prices = CostConfig.model_validate(prices.model_dump())
        self.steps = 0
        self.tokens = 0
        self.cost_cny = Decimal("0")
        self.tool_calls: Counter[str] = Counter()
        self.final_used = False
        self.correction_enabled = correction_enabled
        self.correction_used = False
        self._halted = False
        self._halt_reason: str | None = None
        self._pending: tuple[TokenUsage, Decimal] | None = None
        self._checkpoint: _ReservationCheckpoint | None = None
        self._model_dispatched = False
        self._used_request_ids: set[str] = set()
        self._submission_call = -1
        self._response: tuple[int, str, str] | None = None
        self._delivered: frozenset[str] = frozenset()
        self._final_cost = self._cost(TokenUsage(input_tokens=config.token_budget_per_call,
                                               output_tokens=config.max_output_tokens))
        if self._final_cost > Decimal(str(config.max_cost_cny)):
            raise ValueError("总费用预算不足以预留一次收尾")
        if config_type in {PropagationHarnessConfig, PropagationReviewHarnessConfig, ArbiterHarnessConfig}:
            capacity = self.check_model_call(config.token_budget_per_call, config.max_output_tokens)
            if not capacity.allowed:
                raise ValueError(f"传播节点容量不足以容纳普通请求和收尾: {capacity.reason}")

    def _cost(self, usage: TokenUsage) -> Decimal:
        """复用模型网关的统一单价换算，用十进制累计金额。"""
        return Decimal(str(compute_cost_cny(usage, self.prices)))

    def check_model_call(self, input_tokens: int, output_tokens: int, *, final: bool = False,
                         correction: bool = False) -> BudgetDecision:
        """无副作用检查，与正式预占共用同一组token、费用、次数和收尾规则。"""
        if self._halted or (self.final_used and not correction):
            return BudgetDecision(allowed=False, reason="budget_exhausted")
        if correction and (not final or not self.correction_enabled or not self.final_used or self.correction_used):
            return BudgetDecision(allowed=False, reason="correction_unavailable")
        if final and not correction and self.final_used:
            return BudgetDecision(allowed=False, reason="final_already_used")
        if self._pending is not None:
            return BudgetDecision(allowed=False, reason="call_pending")
        try:
            usage = TokenUsage.model_validate({"input_tokens": input_tokens, "output_tokens": output_tokens}, strict=True)
        except ValidationError:
            return BudgetDecision(allowed=False, reason="invalid_usage")
        if input_tokens > self.config.token_budget_per_call or output_tokens > self.config.max_output_tokens:
            return BudgetDecision(allowed=False, reason="per_call_limit")
        if not final and self.steps >= self.config.max_steps:
            return BudgetDecision(allowed=False, reason="step_limit")
        cost = self._cost(usage)
        if correction:
            remaining_closures = 0
        elif final:
            remaining_closures = int(self.correction_enabled and not self.correction_used)
        else:
            remaining_closures = int(not self.final_used) + int(
                self.correction_enabled and not self.correction_used)
        reserve_tokens = self.config.final_token_reserve * remaining_closures
        reserve_cost = self._final_cost * remaining_closures
        if self.tokens + input_tokens + output_tokens + reserve_tokens > self.config.max_total_tokens:
            return BudgetDecision(allowed=False, reason="token_limit")
        if self.cost_cny + cost + reserve_cost > Decimal(str(self.config.max_cost_cny)):
            return BudgetDecision(allowed=False, reason="cost_limit")
        return BudgetDecision(allowed=True, reason="reserved")

    def reserve_model_call(self, input_tokens: int, output_tokens: int, *, final: bool = False,
                           correction: bool = False,
                           request_id: str | None = None) -> BudgetDecision:
        """沿用旧预占接口，命名请求另保存撤回检查点并永久登记编号。"""
        decision = self.check_model_call(input_tokens, output_tokens, final=final, correction=correction)
        if not decision.allowed:
            return decision
        if request_id is not None:
            try:
                if not isinstance(request_id, str) or UUID(request_id).hex != request_id:
                    return BudgetDecision(allowed=False, reason="invalid_request_id")
            except ValueError:
                return BudgetDecision(allowed=False, reason="invalid_request_id")
            if request_id in self._used_request_ids:
                return BudgetDecision(allowed=False, reason="duplicate_request_id")
            self._checkpoint = _ReservationCheckpoint(request_id, self.steps, self.tokens, self.cost_cny,
                self.final_used, self.correction_used, self._response, self._delivered, self._submission_call,
                CostConfig.model_validate(self.prices.model_dump()))
            self._used_request_ids.add(request_id)
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
        cost = self._cost(usage)
        self._pending = (usage, cost)
        self._model_dispatched = False
        self._response = None
        self._delivered = frozenset()
        self.tokens += input_tokens + output_tokens
        self.cost_cny += cost
        if final:
            if correction:
                self.correction_used = True
            else:
                self.final_used = True
        else:
            self.steps += 1
        return BudgetDecision(allowed=True, reason="reserved")

    def pending_cost_cny(self) -> Decimal:
        """返回当前真实预占金额；没有待结算请求时不伪装成零预占。"""
        if self._pending is None:
            raise ValueError("no_pending_model_reservation")
        return self._pending[1]

    def _require_reservation(self, request_id: str) -> _ReservationCheckpoint:
        """只接受当前命名预占的编号，拒绝旧编号、匿名预占和重复操作。"""
        if self._pending is None or self._checkpoint is None or self._checkpoint.request_id != request_id:
            raise ValueError("reservation_identity_mismatch")
        return self._checkpoint

    def mark_model_dispatched(self, request_id: str) -> None:
        """标为可能已发出，此后不可使用未发送撤回；本方法不调用网络。"""
        self._require_reservation(request_id)
        if self._halted or self._model_dispatched:
            raise ValueError("cannot_mark_model_dispatched")
        self._model_dispatched = True

    def cancel_model_reservation(self, request_id: str, *, rollback_attempt: bool) -> None:
        """只撤回匹配且未发送的预占；完整回退或保留尝试，均不解除停止。"""
        if type(rollback_attempt) is not bool:
            raise ValueError("invalid_rollback_attempt")
        before = self._require_reservation(request_id)
        if self._model_dispatched:
            raise ValueError("reservation_already_dispatched")
        self.tokens, self.cost_cny = before.tokens, before.cost_cny
        if rollback_attempt:
            self.steps, self.final_used, self.correction_used = (
                before.steps, before.final_used, before.correction_used)
            self._submission_call = before.submission_call
            if not self._halted:
                self._response, self._delivered = before.response, before.delivered
        self._pending, self._checkpoint, self._model_dispatched = None, None, False

    def settle_model_call(self, input_tokens: int | None, output_tokens: int | None) -> BudgetDecision:
        """实际用量替换预占；缺失或非法用量保留预占并停止后续付费调用。"""
        if self._pending is None:
            raise ValueError("没有待结算请求，禁止重复结算")
        if self._checkpoint is not None and not self._model_dispatched:
            raise ValueError("named_reservation_not_dispatched")
        reserved, cost = self._pending
        prices = self._checkpoint.prices if self._checkpoint is not None else self.prices
        self._pending = None
        self._checkpoint, self._model_dispatched = None, False
        try:
            usage = TokenUsage.model_validate({"input_tokens": input_tokens, "output_tokens": output_tokens}, strict=True)
        except ValidationError:
            self.halt("usage_unknown")
            return BudgetDecision(allowed=False, reason="usage_unknown")
        self.tokens += usage.input_tokens + usage.output_tokens - reserved.input_tokens - reserved.output_tokens
        self.cost_cny += Decimal(str(compute_cost_cny(usage, prices))) - cost
        if usage.input_tokens > reserved.input_tokens or usage.output_tokens > reserved.output_tokens:
            self.halt("reservation_exceeded")
            return BudgetDecision(allowed=False, reason="reservation_exceeded")
        return BudgetDecision(allowed=True, reason="settled")

    def bind_tool_response(self, response: ToolLLMResult, *, delivered: frozenset[str] = frozenset()) -> None:
        """网关在完整解析后绑定当前响应，身份与内容指纹不能由模型自报。"""
        context = require_run_context().model_dump_json(exclude={"step"})
        call = self.steps + int(self.final_used) + int(self.correction_used)
        if self._pending is not None or self._halted or call == 0 or self._response is not None:
            raise ValueError("cannot_bind_tool_response")
        self._response = (call, context, sha256(response.model_dump_json().encode("utf-8")).hexdigest())
        self._delivered = frozenset(delivered)

    def authorize_tool_response(self, response: ToolLLMResult) -> BudgetDecision:
        """拒绝旧响应、跨运行或被改写的响应，不消费工具次数。"""
        context = require_run_context().model_dump_json(exclude={"step"})
        if self._halted or self._pending is not None:
            return BudgetDecision(allowed=False, reason="budget_unavailable")
        expected = (self.steps + int(self.final_used) + int(self.correction_used), context,
                    sha256(response.model_dump_json().encode("utf-8")).hexdigest())
        allowed = self._response is not None and self._response == expected
        return BudgetDecision(allowed=allowed, reason="response_authorized" if allowed else "untrusted_response")

    def delivered_for(self, response: ToolLLMResult) -> frozenset[str]:
        """只返回当前可信响应所绑定的交付快照，不借用后来阅读的资格。"""
        if not self.authorize_tool_response(response).allowed:
            raise ValueError("untrusted_response")
        return self._delivered

    def status(self) -> BudgetStatus:
        """返回独立状态快照，读取不消耗预算也不解除停用。"""
        return BudgetStatus(halted=self._halted, pending=self._pending is not None, halt_reason=self._halt_reason)

    def halt(self, reason: str = "audit_failed") -> None:
        """审计等致命错误后停用预算，不清除已经发生的费用。"""
        if not reason.strip():
            raise ValueError("empty_halt_reason")
        self._halt_reason = self._halt_reason or reason
        self._halted = True
        self._response = None
        self._delivered = frozenset()

    def reserve_tool_call(self, tool_name: str) -> BudgetDecision:
        """工具开始执行即占次数，收尾只允许一次提交，不退款失败尝试。"""
        submission_tool = self.config.submission_tool
        if self._halted or self._pending is not None:
            return BudgetDecision(allowed=False, reason="budget_unavailable")
        if self.final_used and tool_name != submission_tool:
            return BudgetDecision(allowed=False, reason="final_only")
        if tool_name == submission_tool:
            call = self.steps + int(self.final_used) + int(self.correction_used)
            if call == 0 or self._submission_call == call:
                return BudgetDecision(allowed=False, reason="tool_limit")
            self._submission_call = call
        else:
            limit = self.config.max_tool_calls.get(tool_name)
            if limit is None or self.tool_calls[tool_name] >= limit:
                return BudgetDecision(allowed=False, reason="tool_limit")
        self.tool_calls[tool_name] += 1
        return BudgetDecision(allowed=True, reason="tool_reserved")
