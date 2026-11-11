"""Read-only navigation of locally archived investigation results."""
import ast
from datetime import datetime
from pathlib import Path
import re

from src.yuqing.serve.research_contracts import ResearchRequest, ResearchSnapshot
from src.yuqing.serve.research_frontend import _rows, _cost_label, render_case


class ResearchHistory:
    """Resolve only run IDs under the supplied archive root; never execute investigations."""

    def __init__(self, root: Path) -> None:
        """Bind the archive root without creating directories or opening any wallet."""
        self.root = root.resolve()

    def choices(self) -> tuple[list[tuple[str, str]], str | None]:
        """List successful and failed runs, preferring the newest completed result initially."""
        records = []
        for folder in self.root.iterdir() if self.root.exists() else ():
            if not re.fullmatch(r"[0-9a-f]{32}", folder.name) or folder.resolve().parent != self.root:
                continue
            if not (folder / "snapshot.json").is_file():
                continue
            snapshot = ResearchSnapshot.model_validate_json((folder / "snapshot.json").read_bytes())
            request = ResearchRequest.model_validate_json((folder / "request.json").read_bytes())
            rows = _rows(folder / "stages.jsonl")
            time = str(rows[-1]["time"]) if rows else ""
            status = {"completed": "已完成", "failed": "失败", "interrupted": "中断"}.get(snapshot.status, snapshot.status)
            mode = {"live": "真实", "mock": "Mock", "historical_replay": "回放"}[snapshot.execution_mode]
            mode = "仅样本增量 · " + mode if (folder / "sample-increment.json").exists() else mode
            label = f"{mode} · {status} · {request.input_value[:32]} · {folder.name[:8]}"
            records.append((time, label, folder.name, snapshot.status))
        records.sort(reverse=True)
        selected = next((row[2] for row in records if row[3] == "completed"), records[0][2] if records else None)
        return [(row[1], row[2]) for row in records], selected

    def find_statement(self, text: str) -> str | None:
        """Prefer an existing completed result for the same exact trimmed statement."""
        choices, _ = self.choices()
        matched = []
        for _, identity in choices:
            case = self.root / identity
            if (case / "sample-increment.json").exists():
                continue
            request = ResearchRequest.model_validate_json((case / "request.json").read_bytes())
            if request.input_kind == "text" and request.input_value.strip() == text.strip():
                snapshot = ResearchSnapshot.model_validate_json((case / "snapshot.json").read_bytes())
                if snapshot.status == "completed":
                    return identity
                matched.append(identity)
        return matched[0] if matched else None

    def load(self, research_id: str | None) -> tuple:
        """Load an archived report, quotes and cost without spawning or charging anything."""
        if research_id is None:
            return ("尚无已归档调查结果。", "事实：not_started · 传播：unavailable · 仲裁：not_started",
                    "暂无执行事件。", "尚无引文。", "输入新闻后可开始调查。", {})
        if not re.fullmatch(r"[0-9a-f]{32}", research_id):
            raise ValueError("invalid_archived_run_id")
        case = (self.root / research_id).resolve()
        if case.parent != self.root:
            raise ValueError("archived_run_outside_root")
        snapshot = ResearchSnapshot.model_validate_json((case / "snapshot.json").read_bytes())
        rendered = render_case(case, 0, finished=True)
        rows = _rows(case / "stages.jsonl")
        span = "阶段耗时未归档"
        if snapshot.elapsed_ms is not None:
            span = f"归档执行耗时 {snapshot.elapsed_ms / 1000:.1f}s"
        elif len(rows) >= 2:
            seconds = (datetime.fromisoformat(rows[-1]["time"]) - datetime.fromisoformat(rows[0]["time"])).total_seconds()
            span = f"归档阶段跨度 {seconds:.1f}s（不含启动时间）"
        mode = {"live": "真实调查归档", "mock": "Mock归档", "historical_replay": "历史回放"}[snapshot.execution_mode]
        heading = f"**{mode} · {snapshot.status}** · 只读查看，不发起请求\n\n{_cost_label(case)}\n\n{span}"
        details = {"view_mode": "archived_result", "research_id": research_id, "snapshot": rendered[5]}
        return (heading, rendered[1], rendered[2], rendered[3], friendly_report(rendered[4]), details)


def friendly_report(report: str) -> str:
    """Render program-generated dimension dictionaries as readable report lines."""
    names = {"fact": "事实", "emotion": "情绪", "spread": "传播", "compliance": "合规"}
    reasons = {"agent_unavailable": "数据或Agent不可用", "rules_empty": "规则未配置"}
    result = []
    for line in report.splitlines():
        if line.startswith(("- 缺席维度：{", "- 计分权重：{")):
            prefix, raw = line.split("：", 1)
            try:
                mapping = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                result.append(line)
                continue
            if isinstance(mapping, dict):
                values = [f"{names.get(str(key), str(key))}：{reasons.get(str(value), str(value))}" for key, value in mapping.items()]
                line = prefix + "：" + "；".join(values)
        result.append(line)
    return "\n".join(result)
