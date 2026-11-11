"""逐运行核对成功阅读正文与实际模型输入，本模块尚不执行上下文压缩。"""

from copy import deepcopy
from hashlib import sha256
import json

from src.yuqing.harness.ledger import EvidenceLedger
from src.yuqing.harness.initial_evidence import (InitialEvidenceReceiptItemV2,
    parse_delivery_message)
from src.yuqing.harness.tools import ReadEvidenceArgs
from src.yuqing.harness.trace import record_event, require_run_context
from src.yuqing.schemas import EvidenceToolPayload, InheritedEvidencePlanV2, ToolExecutionResult
from src.yuqing.harness.inherited_evidence import verify_inherited_request_block


class EvidenceDelivery:
    """可信程序暂存工具返回值，只有匹配实际请求的正文才能授予交付资格。"""

    def __init__(self, *, initial_message: str | None = None,
                 initial_evidence_ids: frozenset[str] = frozenset(),
                 initial_expected: tuple[InitialEvidenceReceiptItemV2, ...] = (),
                 program_messages: tuple[str, ...] = (),
                 inherited_plan: InheritedEvidencePlanV2 | None = None) -> None:
        """绑定当前运行身份，禁止跨运行、题目、专家或追踪目录复用。"""
        self._identity = require_run_context().model_dump_json(exclude={"step"})
        self._reads: dict[str, ToolExecutionResult] = {}
        self._initial_message = initial_message
        self._initial_evidence_ids = initial_evidence_ids
        self._initial_expected = initial_expected
        self._program_messages = program_messages
        self._inherited_plan = inherited_plan

    def _require_identity(self) -> None:
        """每次使用均复核可信运行身份。"""
        if require_run_context().model_dump_json(exclude={"step"}) != self._identity:
            raise ValueError("delivery_identity_mismatch")

    def stage_read(self, result: ToolExecutionResult, ledger: EvidenceLedger) -> None:
        """只接成功阅读并核对实际前缀，保存深副本但不授予交付资格。"""
        self._require_identity()
        result = ToolExecutionResult.model_validate_json(result.model_dump_json())
        if result.name != "read_evidence" or result.payload.status != "success" or not result.call_id:
            raise ValueError("not_successful_read")
        if len(result.payload.items) != 1 or result.call_id in self._reads:
            raise ValueError("invalid_read_result")
        item = result.payload.items[0]
        entry = ledger.get(item.evidence_id)
        if (entry is None or not item.text.strip() or item.read_end != len(item.text)
                or item.read_end > entry.read_end or not entry.text.startswith(item.text)
                or item.truncated != (item.read_end < len(entry.text))):
            raise ValueError("read_body_mismatch")
        self._reads[result.call_id] = result

    def assemble(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        """在消息独立副本中补齐待交付正文，所有工具申请及结果必须完整配对。"""
        self._require_identity()
        snapshot = deepcopy(messages)
        present = {message.get("tool_call_id") for message in snapshot if message.get("role") == "tool"}
        for call_id, result in self._reads.items():
            if call_id not in present:
                snapshot.append({"role": "tool", "tool_call_id": call_id,
                                 "content": result.payload.model_dump_json()})
        self.inspect(snapshot)
        return snapshot

    def inspect(self, messages: list[dict[str, object]]) -> frozenset[str]:
        """核对实际输入中的配对、正文与编号，拒绝遗漏、改写及伪装成功的阅读。"""
        self._require_identity()
        pending: dict[str, dict[str, object]] = {}
        seen: set[str] = set()
        delivered: set[str] = set()
        included: set[str] = set()
        initial_seen = False
        initial_count = 0
        seen_program: dict[str, int] = {}
        for message in messages:
            if message.get("role") == "tool":
                self._inspect_tool_result(message, pending, delivered, included)
            else:
                initial_count, initial_seen = self._inspect_message(
                    message, pending, seen, seen_program, initial_count, initial_seen)
        if pending or included != self._reads.keys():
            raise ValueError("missing_read_body")
        if self._initial_message is not None and not initial_seen:
            raise ValueError("missing_initial_evidence_body")
        if initial_count > 1:
            raise ValueError("duplicate_initial_evidence_body")
        if any(seen_program.get(item) != 1 for item in self._program_messages):
            raise ValueError("followup_data_block_missing_or_duplicate")
        if initial_seen:
            delivered.update(self._initial_evidence_ids)
        if self._inherited_plan is not None:
            inherited_message = next((item for item in self._program_messages
                                      if "<inherited_evidence_v2" in item), None)
            if inherited_message is None:
                raise ValueError("inherited_evidence_block_missing")
            verify_inherited_request_block(inherited_message, self._inherited_plan)
            delivered.update(item.evidence_id for item in self._inherited_plan.entries)
        return frozenset(delivered)

    def _inspect_message(self, message: dict[str, object], pending: dict[str, dict[str, object]],
                         seen: set[str], seen_program: dict[str, int],
                         initial_count: int, initial_seen: bool) -> tuple[int, bool]:
        """核对非工具消息内的证据数据块及工具申请配对。"""
        if message.get("role") == "user":
            initial_count, initial_seen = self._inspect_initial(message, initial_count, initial_seen)
            self._inspect_program_blocks(message, seen_program)
        if pending:
            raise ValueError("incomplete_tool_pair")
        self._register_calls(message, pending, seen)
        return initial_count, initial_seen

    def _inspect_initial(self, message: dict[str, object], count: int,
                         seen: bool) -> tuple[int, bool]:
        """确认首次证据消息必须与程序生成的完整清单一致。"""
        if self._initial_message is None:
            return count, seen
        content = message.get("content")
        if content == self._initial_message:
            if parse_delivery_message(content) != self._initial_expected:
                raise ValueError("initial_evidence_manifest_mismatch")
            return count + 1, True
        if "<initial_evidence_v2" in str(content or ""):
            raise ValueError("untrusted_initial_evidence_message")
        return count, seen

    def _inspect_program_blocks(self, message: dict[str, object], counts: dict[str, int]) -> None:
        """要求程序任务与继承材料逐字按独立消息交付。"""
        content = str(message.get("content", ""))
        for expected in self._program_messages:
            if _program_marker(expected) in content:
                if content != expected:
                    raise ValueError("untrusted_followup_data_block")
                counts[expected] = counts.get(expected, 0) + 1

    def _register_calls(self, message: dict[str, object], pending: dict[str, dict[str, object]],
                        seen: set[str]) -> None:
        """登记助手申请的工具调用并拒绝重复或错误角色申请。"""
        for call in message.get("tool_calls", []):
            call_id = call.get("id")
            if (message.get("role") != "assistant" or call.get("type") != "function"
                    or not call_id or call_id in seen):
                raise ValueError("invalid_tool_pair")
            seen.add(call_id)
            pending[call_id] = call["function"]

    def _inspect_tool_result(self, message: dict[str, object], pending: dict[str, dict[str, object]],
                             delivered: set[str], included: set[str]) -> None:
        """核对已登记的工具配对与成功阅读正文。"""
        call_id = message.get("tool_call_id")
        if call_id not in pending:
            raise ValueError("unpaired_tool_result")
        function = pending.pop(call_id)
        if function.get("name") != "read_evidence":
            return
        payload = EvidenceToolPayload.model_validate_json(message.get("content"), strict=True)
        if payload.status != "success":
            return
        read = self._reads.get(call_id)
        args = ReadEvidenceArgs.model_validate_json(function.get("arguments"), strict=True)
        if (read is None or payload != read.payload
                or args.evidence_id != read.payload.items[0].evidence_id):
            raise ValueError("untrusted_read_message")
        delivered.add(args.evidence_id)
        included.add(call_id)

    def confirm(self, request_id: str, messages: list[dict[str, object]],
                *, request_archive: dict[str, str]) -> frozenset[str]:
        """响应可信后审计实际请求正文；审计成功才返回可绑定的交付编号。"""
        delivered = self.inspect(messages)
        data = {"request_id": request_id, "status": "included_in_request",
            "request_archive": dict(request_archive),
            "messages_sha256": sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "evidence": [{"call_id": call_id, "evidence_id": item.evidence_id,
                "read_end": item.read_end, "truncated": item.truncated,
                "returned_sha256": sha256(item.text.encode("utf-8")).hexdigest()}
                for call_id, read in self._reads.items() for item in read.payload.items]}
        if self._initial_message is not None:
            data["initial_evidence_ids"] = sorted(self._initial_evidence_ids)
            data["initial_evidence"] = [{"evidence_id": item.evidence_id, "read_start": 0,
                "read_end": len(item.text), "body_sha256": sha256(item.text.encode("utf-8")).hexdigest()}
                for item in self._initial_expected]
        if self._inherited_plan is not None:
            data["inherited_plan_sha256"] = sha256(
                self._inherited_plan.model_dump_json().encode("utf-8")).hexdigest()
            data["inherited_evidence"] = [{"evidence_id": item.evidence_id,
                "origin": "inherited",
                "origin_refs": list(item.origin_refs), "read_end": item.read_end,
                "body_sha256": item.body_sha256, "prefix_sha256": item.prefix_sha256}
                for item in self._inherited_plan.entries]
        record_event("evidence_delivery", data)
        return delivered


def _program_marker(message: str) -> str:
    """Return the opening marker used to find tampered program-owned JSON blocks."""
    start = message.find("<")
    end = message.find(">", start + 1)
    return message[start:end + 1] if start >= 0 and end > start else message
