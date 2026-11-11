"""In-process graph registry and explicit Mock case assembly for M7."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any
from uuid import uuid4
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.yuqing.graph.runtime import (CaseRuntimeAssembly, CaseRuntimeContext,
    assemble_case_runtime, build_case_dry_run_handlers)
from src.yuqing.settings import get_graph_config


class CaseRegistry:
    """Keep MemorySaver graph objects reachable while this service process lives."""

    def __init__(self) -> None:
        """Create an empty thread-safe registry for paused demo cases."""
        self._items: dict[str, CaseRuntimeAssembly] = {}
        self._lock = RLock()

    def add(self, case_id: str, assembly: CaseRuntimeAssembly) -> None:
        """Register one freshly created case assembly under its trusted identity."""
        with self._lock:
            if case_id in self._items:
                raise ValueError("case_registry_duplicate_identity")
            self._items[case_id] = assembly

    def get(self, case_id: str) -> CaseRuntimeAssembly | None:
        """Return a case assembly while preserving concurrent map access safety."""
        with self._lock:
            return self._items.get(case_id)

    def get_or_create(self, case_id: str,
                      factory: Callable[[], CaseRuntimeAssembly]) -> CaseRuntimeAssembly:
        """Serialize restore construction so concurrent requests share one assembly."""
        with self._lock:
            existing = self._items.get(case_id)
            if existing is not None:
                return existing
            assembly = factory()
            self._items[case_id] = assembly
            return assembly


def build_demo_assembly(case_id: str, ticket: dict[str, Any],
                        checkpointer: BaseCheckpointSaver | None = None) -> CaseRuntimeAssembly:
    """Bind the existing deterministic Mock handlers to a unique API ticket."""
    sample_id = ticket.get("sample_id") or f"demo:{case_id}"
    handlers, base_score_inputs = build_case_dry_run_handlers()

    def score_inputs(state: dict[str, Any]) -> dict[str, Any]:
        """Keep the production scorer while forwarding the declared demo scenario."""
        values = base_score_inputs(state)
        values["scenario"] = ticket.get("scenario") or "general"
        return values

    state = {"case_id": case_id, "sample": {"sample_id": sample_id}, "ticket": ticket,
        "is_demo": True, "scenario": ticket.get("scenario") or "general",
        "collab_config": {"eval_mode": False}}
    context = CaseRuntimeContext(case_id, sample_id,
        thread_id=ticket.get("thread_id") or uuid4().hex)
    return assemble_case_runtime(state, handlers, score_inputs, config=get_graph_config(),
        eval_mode=False, context=context, checkpointer=checkpointer)
