"""只读重建 M4 固定样本的历史传播材料与来源链。"""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import stat
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict

from src.yuqing.agents.propagation import PropagationToolPayload, review_propagation_report
from src.yuqing.eval.propagation_ced10 import _file_sha256
from src.yuqing.features.propagation import CommentItem, SpreadStats
from src.yuqing.harness.trace import TraceEvent
from src.yuqing.ingest.ced import load_ced_sample
from src.yuqing.ingest.chef import load_chef
from src.yuqing.schemas import PropagationRaw, PropagationReport
from src.yuqing.settings import PROJECT_ROOT

MANIFEST_PATH = Path("docs/cards/M4-five-sample-manifest.json")
_MANIFEST_SHA256 = "34f83f3e7932e762e1fcb3c8c34869582f983393bc549e48047ef1a8eaa1ca13"
_EXPECTED_IDS = ("chef:dev:0", "chef:dev:1", "chef:dev:10",
                 "ced:non_rumor:2602_AfRpJr42Z_2451549504", "ced:rumor:2421_z6uV4bCT2_2345130274")
_NULLABLE_STATS_FIELDS = ("reposts", "comments", "likes", "root_followers", "root_verified", "nodes_1h",
                          "nodes_6h", "nodes_24h", "peak_nodes_per_hour", "max_depth", "direct_reposts",
                          "text_repost_ratio", "visible_comments")

class M4HistorySourceFile(BaseModel):
    """固定样本依赖的一份原始输入文件及其字节指纹。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str

class M4HistoryManifestItem(BaseModel):
    """固定五样本的一条历史来源记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str
    source: Literal["chef", "ced"]
    coverage: Literal["none", "has_data"]
    result_csv: str
    result_csv_sha256: str
    trace_path: str
    trace_sha256: str
    run_id: str
    claim_sha256: str
    report_sha256: str
    stats_sha256: str
    source_files: tuple[M4HistorySourceFile, ...]

class M4HistoryManifest(BaseModel):
    """M4 历史材料的固定版本与有序样本清单。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["m4-history-propagation-v1"]
    arbiter_prompt_sha256: str
    items: tuple[M4HistoryManifestItem, ...]

class M4HistoricalPropagationInput(BaseModel):
    """由归档重新核验出的传播输入；no_coverage 不暴露历史空报告。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    sample_id: str
    source: Literal["chef", "ced"]
    claim: str
    coverage: Literal["no_coverage", "has_data"]
    report: PropagationReport | None
    stats: SpreadStats | None
    comments: tuple[CommentItem, ...]
    report_sha256: str
    trace_sha256: str
    review_accepted: bool

def load_m4_history_inputs(root: Path = PROJECT_ROOT) -> tuple[M4HistoricalPropagationInput, ...]:
    """校验固定清单及其历史 CSV、Trace、归档和原始样本后返回传播材料。"""
    manifest_bytes = (root / MANIFEST_PATH).read_bytes()
    if _digest(manifest_bytes) != _MANIFEST_SHA256:
        raise ValueError("m4_history_manifest_fingerprint_mismatch")
    manifest = M4HistoryManifest.model_validate_json(manifest_bytes)
    _validate_manifest(manifest)
    chef_samples = ({sample.sample_id: sample for sample in load_chef(root / "data/CHEF/dev.json")}
                    if any(item.source == "chef" for item in manifest.items) else {})
    return tuple(_load_item(root, item, chef_samples) for item in manifest.items)

def _validate_manifest(manifest: M4HistoryManifest) -> None:
    """拒绝样本身份、顺序、来源或提示词许可指纹漂移。"""
    ids = tuple(item.sample_id for item in manifest.items)
    if ids != _EXPECTED_IDS or len(set(ids)) != len(ids):
        raise ValueError("m4_history_sample_manifest_invalid")
    if manifest.arbiter_prompt_sha256 != "298cf72780715713e165c8231e24d7cae86308bb6354a6933e1db0156b0ab8e8":
        raise ValueError("m4_history_prompt_fingerprint_invalid")
    for item in manifest.items:
        expected_source = "chef" if item.sample_id.startswith("chef:") else "ced"
        expected_coverage = "none" if expected_source == "chef" else "has_data"
        if item.source != expected_source or item.coverage != expected_coverage:
            raise ValueError("m4_history_source_coverage_invalid")

def _load_item(root: Path, item: M4HistoryManifestItem, chef_samples: Mapping[str, object]
               ) -> M4HistoricalPropagationInput:
    """重建一条固定样本，并拒绝任何来源或确定性复核差异。"""
    claim, raw = _load_source_sample(root, item, chef_samples)
    if _digest(claim.encode("utf-8")) != item.claim_sha256:
        raise ValueError("m4_history_claim_fingerprint_mismatch")
    row = _read_result_row(root, item)
    events = _read_trace(root, item)
    report, stats, comments = _verified_submission(root, item, events)
    _compare_result_row(row, report, stats, item)
    review = review_propagation_report(report, claim=claim, raw=raw,
                                       returned_comments={c.comment_id: c for c in comments}, stats=stats)
    if not review.accepted:
        raise ValueError("m4_history_submission_recheck_failed")
    if item.coverage == "none":
        if raw is not None or (stats is not None and stats.has_data) or comments:
            raise ValueError("m4_history_no_coverage_mismatch")
        return M4HistoricalPropagationInput(sample_id=item.sample_id, source=item.source,
            claim=claim, coverage="no_coverage", report=None, stats=None, comments=(),
            report_sha256=item.report_sha256, trace_sha256=item.trace_sha256, review_accepted=True)
    if raw is None or stats is None or not stats.has_data:
        raise ValueError("m4_history_data_coverage_missing")
    return M4HistoricalPropagationInput(sample_id=item.sample_id, source=item.source, claim=claim,
        coverage="has_data", report=report, stats=stats, comments=comments,
        report_sha256=item.report_sha256, trace_sha256=item.trace_sha256, review_accepted=True)

def _load_source_sample(root: Path, item: M4HistoryManifestItem,
                        chef_samples: Mapping[str, object]) -> tuple[str, PropagationRaw | None]:
    """验证原始数据文件指纹并重载声明及 CED 传播树。"""
    for source_file in item.source_files:
        if _file_sha256(_under_root(root, source_file.path)) != source_file.sha256:
            raise ValueError("m4_history_source_fingerprint_mismatch")
    if item.source == "chef":
        sample = chef_samples.get(item.sample_id)
        if sample is None:
            raise ValueError("m4_history_chef_sample_missing")
        return sample.claim, None
    stem = item.sample_id.rsplit(":", 1)[1]
    sample = load_ced_sample(root / "data/raw/CED_Dataset", f"{stem}.json",
                              rumor=item.sample_id.startswith("ced:rumor:"))
    return sample.claim, sample.propagation

def _read_result_row(root: Path, item: M4HistoryManifestItem) -> dict[str, str]:
    """只接受固定结果 CSV 中唯一、成功且身份匹配的答卷行。"""
    path = _under_root(root, item.result_csv)
    if _file_sha256(path) != item.result_csv_sha256:
        raise ValueError("m4_history_result_csv_fingerprint_mismatch")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        matches = [row for row in csv.DictReader(stream) if row.get("sample_id") == item.sample_id]
    if len(matches) != 1:
        raise ValueError("m4_history_result_row_count_invalid")
    row = matches[0]
    if (row.get("source") != item.source or row.get("run_id") != item.run_id
            or row.get("status") != "success" or row.get("submission_validation") != "accepted"
            or Path(row.get("trace_path", "")).name != Path(item.trace_path).name):
        raise ValueError("m4_history_result_identity_mismatch")
    return row

def _read_trace(root: Path, item: M4HistoryManifestItem) -> list[TraceEvent]:
    """读取固定 Trace 原始字节，并检查所有事件的运行身份。"""
    path = _under_root(root, item.trace_path)
    raw = path.read_bytes()
    if _digest(raw) != item.trace_sha256:
        raise ValueError("m4_history_trace_fingerprint_mismatch")
    events = [TraceEvent.model_validate_json(line, strict=True)
              for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not events or any((e.run_id, e.sample_id, e.agent) !=
                         (item.run_id, item.sample_id, "propagation_analyst") for e in events):
        raise ValueError("m4_history_trace_identity_mismatch")
    return events

def _verified_submission(root: Path, item: M4HistoryManifestItem,
                         events: list[TraceEvent]) -> tuple[PropagationReport, SpreadStats | None, tuple[CommentItem, ...]]:
    """核对模型调用链、工具回执交付和唯一最终报告。"""
    requests, responses, calls = _verified_calls(root, item, events)
    accepted = [call for call in calls if call[0].data.get("tool") == "submit_propagation_report"
                and call[0].data.get("reason") == "accepted"]
    if len(accepted) != 1:
        raise ValueError("m4_history_accepted_submission_count_invalid")
    event, archived = accepted[0]
    request_id = _request_id_for_step(requests, event.step)
    request_archive = requests[request_id][1]
    response_archive = responses[request_id][1]
    report = PropagationReport.model_validate(archived["request"]["arguments"])
    response_calls = response_archive.get("body", {}).get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
    matching = [call for call in response_calls if call.get("id") == archived["request"].get("call_id")]
    result, accepted_result = _parse_tool_payload(archived["content"])
    if (archived["request"].get("name") != "submit_propagation_report" or len(matching) != 1
            or event.data.get("status") != "success" or result.reason != "accepted"
            or not accepted_result):
        raise ValueError("m4_history_submission_request_invalid")
    if max(value[0].step for value in requests.values()) != event.step:
        raise ValueError("m4_history_submission_not_final")
    stats, comments = _delivered_inputs(calls, request_archive["body"].get("messages", []), event.step)
    if _object_digest(report.model_dump(mode="json")) != item.report_sha256:
        raise ValueError("m4_history_report_fingerprint_mismatch")
    stats_data = stats.model_dump(mode="json") if stats is not None else None
    if _object_digest(stats_data) != item.stats_sha256:
        raise ValueError("m4_history_stats_fingerprint_mismatch")
    if any(call[0].step > event.step for call in calls):
        raise ValueError("m4_history_late_tool_call")
    return report, stats, comments

def _verified_calls(root: Path, item: M4HistoryManifestItem, events: list[TraceEvent]
                    ) -> tuple[dict[str, tuple[TraceEvent, dict[str, object]]],
                               dict[str, tuple[TraceEvent, dict[str, object]]],
                               list[tuple[TraceEvent, dict[str, object]]]]:
    """验证每个请求、模型响应、工具调用与结果归档都能一一对应。"""
    requests, responses, tools = {}, {}, []
    for event in events:
        rid = event.data.get("request_id")
        if event.type == "llm_request":
            if not isinstance(rid, str) or rid in requests:
                raise ValueError("m4_history_request_id_invalid")
            payload = _read_archive(root, item, event.data.get("archive"), "model_request", event.step)
            if payload.get("request_id") != rid or not isinstance(payload.get("body"), dict):
                raise ValueError("m4_history_request_archive_mismatch")
            requests[rid] = (event, payload)
        elif event.type == "model_response_archive":
            if not isinstance(rid, str) or rid in responses:
                raise ValueError("m4_history_response_id_invalid")
            payload = _read_archive(root, item, event.data.get("archive"), "model_response", event.step)
            if payload.get("request_id") != rid or not isinstance(payload.get("body"), dict):
                raise ValueError("m4_history_response_archive_mismatch")
            responses[rid] = (event, payload)
        elif event.type == "tool_call":
            tools.append((event, _read_archive(root, item, event.data.get("archive"), "tool_result", event.step)))
    if requests.keys() != responses.keys() or not requests:
        raise ValueError("m4_history_request_response_mismatch")
    _match_tool_calls(item, requests, responses, tools)
    return requests, responses, tools

def _match_tool_calls(item: M4HistoryManifestItem, requests: Mapping[str, tuple],
                      responses: Mapping[str, tuple], tools: list[tuple]) -> None:
    """确保响应中的每份工具申请恰好对应程序记录的同一步回执。"""
    by_call_id = {str(archive["request"].get("call_id")): (event, archive) for event, archive in tools}
    if len(by_call_id) != len(tools):
        raise ValueError("m4_history_tool_call_id_duplicate")
    expected = set()
    for request_id, (request_event, request_archive) in requests.items():
        response_event, response_archive = responses[request_id]
        if request_event.step != response_event.step:
            raise ValueError("m4_history_request_response_step_mismatch")
        body = response_archive.get("body", {})
        calls = body.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        for call in calls:
            call_id = call.get("id")
            event_archive = by_call_id.get(call_id)
            if event_archive is None:
                raise ValueError("m4_history_tool_response_missing")
            event, archive = event_archive
            function = call.get("function", {})
            arguments = json.loads(function.get("arguments", "{}"))
            payload_request = archive["request"]
            if (event.step != request_event.step or event.data.get("call_id") != call_id
                    or event.data.get("tool") != function.get("name")
                    or payload_request.get("call_id") != call_id
                    or payload_request.get("name") != function.get("name")
                    or payload_request.get("arguments") != arguments):
                raise ValueError("m4_history_tool_correlation_mismatch")
            expected.add(call_id)
    if expected != set(by_call_id):
        raise ValueError("m4_history_orphan_tool_result")

def _delivered_inputs(calls: list[tuple[TraceEvent, dict[str, object]]],
                      final_messages: object, final_step: int) -> tuple[SpreadStats | None, tuple[CommentItem, ...]]:
    """仅重建最终提交请求实际收到的成功统计和评论回执。"""
    if not isinstance(final_messages, list):
        raise ValueError("m4_history_final_messages_missing")
    delivered = {message.get("tool_call_id"): message.get("content") for message in final_messages
                 if isinstance(message, dict) and message.get("role") == "tool"}
    stats, comments = None, {}
    for event, archive in calls:
        if event.step >= final_step or event.data.get("tool") == "submit_propagation_report":
            continue
        content = archive.get("content")
        call_id = archive["request"].get("call_id")
        if not isinstance(content, str) or delivered.get(call_id) != content:
            continue
        payload, _ = _parse_tool_payload(content)
        if payload.status != "success" or payload.reason != "ok":
            continue
        stats = _merge_stats(stats, payload.stats)
        for comment in payload.comments:
            prior = comments.get(comment.comment_id)
            if prior is not None and prior != comment:
                raise ValueError("m4_history_comment_conflict")
            comments[comment.comment_id] = comment
    return stats, tuple(comments[key] for key in sorted(comments))

def _merge_stats(current: SpreadStats | None, candidate: SpreadStats | None) -> SpreadStats | None:
    """拒绝不同工具回执中不一致的传播统计。"""
    if current is not None and candidate is not None and current != candidate:
        raise ValueError("m4_history_stats_conflict")
    return current or candidate

def _parse_tool_payload(content: str) -> tuple[PropagationToolPayload, bool | None]:
    """校验归档回执，同时读取序列化中的计算字段 accepted。"""
    value = json.loads(content)
    feedback = value.get("feedback") if isinstance(value, dict) else None
    accepted = feedback.get("accepted") if isinstance(feedback, dict) else None
    if isinstance(feedback, dict):
        feedback = {key: item for key, item in feedback.items() if key != "accepted"}
        value["feedback"] = feedback
    stats = value.get("stats") if isinstance(value, dict) else None
    if isinstance(stats, dict):
        for name in _NULLABLE_STATS_FIELDS:
            stats.setdefault(name, None)
    return PropagationToolPayload.model_validate(value), accepted if type(accepted) is bool else None

def _compare_result_row(row: Mapping[str, str], report: PropagationReport,
                        stats: SpreadStats | None, item: M4HistoryManifestItem) -> None:
    """把 CSV 可见报告字段与归档的最终提交逐项比较。"""
    report_json_fields = {"emotion_signals", "spread_basis", "narrative_basis", "leads"}
    for key, value in report.model_dump(mode="json").items():
        if key == "confidence":
            # Historical CSVs omit confidence; the pinned full-report digest below covers it.
            continue
        actual = json.loads(row[key]) if key in report_json_fields else row[key]
        if actual != value:
            raise ValueError("m4_history_csv_report_mismatch")
    row_stats = json.loads(row["stats"])
    if row_stats != (stats.model_dump(mode="json") if stats is not None else None):
        raise ValueError("m4_history_csv_stats_mismatch")
    if (row.get("submission_validation") != "accepted"
            or _object_digest(report.model_dump(mode="json")) != item.report_sha256):
        raise ValueError("m4_history_csv_submission_mismatch")

def _read_archive(root: Path, item: M4HistoryManifestItem, reference: object,
                  kind: str, step: int) -> dict[str, object]:
    """读取并核验 Trace 引用的单份安全归档。"""
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("m4_history_archive_reference_invalid")
    relative, expected = reference.get("path"), reference.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("m4_history_archive_reference_invalid")
    path = _under_root(root / "traces", relative)
    raw = path.read_bytes()
    if _digest(raw) != expected:
        raise ValueError("m4_history_archive_sha_mismatch")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("m4_history_archive_identity_mismatch")
    payload = document.get("payload")
    if (document.get("kind") != kind or document.get("run_id") != item.run_id
            or document.get("sample_id") != item.sample_id or document.get("agent") != "propagation_analyst"
            or document.get("step") != step or not isinstance(payload, dict)):
        raise ValueError("m4_history_archive_identity_mismatch")
    return payload

def _under_root(root: Path, relative: str) -> Path:
    """只解析项目内普通文件，拒绝绝对路径、越界和重解析点。"""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative or ":" in relative:
        raise ValueError("m4_history_path_invalid")
    base = root.resolve(strict=True)
    try:
        target = (base / candidate).resolve(strict=True)
    except OSError as exc:
        raise ValueError("m4_history_file_missing") from exc
    if target == base or not target.is_relative_to(base) or not target.is_file():
        raise ValueError("m4_history_path_outside_root")
    current = base
    for part in candidate.parts:
        current /= part
        metadata = current.lstat()
        flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if current.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & flag):
            raise ValueError("m4_history_reparse_point")
    return target

def _request_id_for_step(requests: Mapping[str, tuple], step: int) -> str:
    """查找最终提交所在的唯一模型请求。"""
    matches = [request_id for request_id, (event, _) in requests.items() if event.step == step]
    if len(matches) != 1:
        raise ValueError("m4_history_final_request_invalid")
    return matches[0]

def _digest(payload: bytes) -> str:
    """计算原始字节的 SHA-256。"""
    return sha256(payload).hexdigest()

def _object_digest(value: object) -> str:
    """按稳定 JSON 编码计算对象 SHA-256。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _digest(payload)
