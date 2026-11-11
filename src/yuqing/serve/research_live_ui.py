"""Lightweight, explicit live action boundary; no wallet, credentials or Agent imports."""
from collections.abc import Callable
from dataclasses import dataclass, asdict
from hashlib import sha256
from html import escape
import json


@dataclass(frozen=True)
class LiveUIPlan:
    """Read-only presentation of a backend-pinned sample plan, never execution permission."""

    source_id: str
    main_sha256: str
    material_urls: tuple[str, ...]
    prompt_sha256: str
    model: str
    price_fingerprint: str
    remaining_cny: str
    node_limit_cny: str

    def fingerprint(self) -> str:
        """Bind confirmation to all previewed source, model and cost fields."""
        return sha256(json.dumps(asdict(self),sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def markdown(self) -> str:
        """Display backend metadata as escaped text, without executable URLs or JSON dumps."""
        rows=["**仅样本增量计划 · 尚未执行**", "原事实/仲裁报告只读复用，总体规模/情绪仍unavailable。",
            "主报告："+self.source_id, "主报告SHA："+self.main_sha256,
            "Prompt SHA："+self.prompt_sha256,"模型："+self.model,"价格指纹："+self.price_fingerprint,
            "累计保守余量（元）："+self.remaining_cny,"节点上限（元）："+self.node_limit_cny,
            "材料："+"；".join(self.material_urls)]
        return "\n\n".join(escape(row).replace("[","\\[").replace("]","\\]") for row in rows)


class LiveUIBinding:
    """Separate read-only previews from an explicitly installed backend executor."""

    def __init__(self, preview: Callable[[str], LiveUIPlan], execute: Callable[[str], str]) -> None:
        """Store callbacks only; construction never reads a wallet or executes a task."""
        self._preview,self._execute=preview,execute

    def preview(self, identity: str) -> tuple[str, tuple[str, str]]:
        """Read and bind the exact plan displayed for the selected main report."""
        plan=self._preview(identity)
        if plan.source_id!=identity:
            raise ValueError("live_ui_source_mismatch")
        return plan.markdown(),(identity,plan.fingerprint())

    def execute(self, identity: str, confirmed: bool, token: tuple[str, str] | None) -> str:
        """Require confirmation of the unchanged preview before delegating one explicit action."""
        if confirmed is not True or token is None or token[0]!=identity:
            raise ValueError("live_ui_preview_and_confirmation_required")
        plan=self._preview(identity)
        if plan.source_id!=identity or plan.fingerprint()!=token[1]:
            raise ValueError("live_ui_plan_changed_preview_again")
        return self._execute(identity)
