"""为后续消融实验复制案件快照并分配独立图线程身份。"""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from src.yuqing.graph.state import CaseState


def fork_case_state(snapshot: CaseState) -> CaseState:
    """深拷贝第一轮状态；保留案件身份但隔离可变台账和检查点线程。"""
    forked = deepcopy(snapshot)
    forked["thread_id"] = uuid4().hex
    return forked
