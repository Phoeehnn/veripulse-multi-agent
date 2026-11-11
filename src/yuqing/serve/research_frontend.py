"""Thin UI process controller; Agent dependencies remain in a separate Python environment."""
from collections.abc import Iterator
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import subprocess
from time import monotonic, sleep
from uuid import uuid4
from urllib.parse import urlsplit

from src.yuqing.serve.research_contracts import ResearchRequest, ResearchSnapshot
from src.yuqing.serve.research_public_view import public_report_view, public_business_timeline, public_agent_label


REQUIRED_ENV = ("TAVILY_API_KEY", "DEEPSEEK_API_KEY", "MODEL_CACHE_DIR", "LIVE_RESEARCH_CNY_PER_CREDIT")


def _rows(path: Path) -> list[dict[str, object]]:
    """Read complete JSONL records only; never invent progress for an unfinished line."""
    if not path.exists():
        return []
    data = path.read_text(encoding="utf-8")
    lines = data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith("\n") and line.strip()]


def render_case(case: Path, elapsed: float, *, finished: bool = False) -> tuple:
    """Render actual worker stages and trace records without exposing raw model payloads."""
    rows = _rows(case / "stages.jsonl")
    status = rows[-1]["stage"] + " · " + rows[-1]["status"] if rows else "工作进程已启动，等待执行回执"
    agents = {"fact": "not_started", "propagation": "unavailable", "arbiter": "not_started"}
    timeline = [f"- {row['time']} · {row['stage']} · {row['status']}" for row in rows]
    roles = {"fact_investigator": "fact", "propagation_analyst": "propagation", "arbiter": "arbiter"}
    observed = sorted((row for path in (case / "traces").rglob("*.jsonl") for row in _rows(path)), key=lambda r: r["ts"])
    for row in observed:
        role = roles.get(row.get("agent"))
        if role:
            agents[role] = "running"
            if row["type"] == "run_outcome":
                result = row["data"]
                accepted = str(result.get("outcome_class", "")).startswith("submitted_") or result.get("submitted") is True
                agents[role] = "completed" if accepted else "failed"
        if row["type"] in {"tool_call", "fact_v2_submission_validation", "run_outcome"}:
            tool = str(row.get("data", {}).get("tool", ""))
            timeline.append(f"- {row['ts']} · {row.get('agent')} · {row['type']} {tool}")
    report, sources, detail = "等待调查报告。", "尚无已核验引用。", {}
    propagation_label = "传播 Agent"
    if (case / "snapshot.json").exists():
        snapshot = ResearchSnapshot.model_validate_json((case / "snapshot.json").read_bytes())
        agents.update({agent.role: agent.status for agent in snapshot.agents})
        status = snapshot.status
        report = public_report_view(snapshot)
        propagation_label = public_agent_label(snapshot)
        timeline.extend(public_business_timeline(case, snapshot))
        if snapshot.unresolved:
            report += "\n\n### 未解决问题\n" + "\n".join("- " + escape(item) for item in snapshot.unresolved)
        sources = _citation_cards(snapshot)
        detail = snapshot.model_dump(mode="json")
    if finished and not (case / "snapshot.json").exists():
        status = "调查未完成，请检查配置与执行日志；未生成结论。"
        agents = {role: "interrupted" if state == "running" else state for role, state in agents.items()}
    cost = _cost_label(case)
    cards = " · ".join(f"**{label}** `{agents[role]}`" for role, label in (("fact", "事实 Agent"), ("propagation", propagation_label), ("arbiter", "仲裁 Agent")))
    cards += "\n\n总体传播规模：`unavailable` · 总体情绪：`unavailable` · 样本分析不计入全网风险"
    return (f"**实时调查** · {escape(str(status))} · {elapsed:.1f}s · {escape(cost)}",
            cards, "\n".join(timeline[-30:]) or "暂无执行事件。", sources, report, detail)


def run_investigation(text: str, *, worktree: Path, core_python: Path, run_root: Path, resume_source: Path | None = None, synthetic_acceptance: bool = False) -> Iterator[tuple]:
    """Launch actual work only when credentials exist; keep keys out of files and arguments."""
    existing = _existing_statement(run_root, text) if resume_source is None and text.strip() else None
    if existing is not None:
        yield existing
        return
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name, "").strip()]
    if missing or not text.strip():
        message = "缺少配置：" + ", ".join(missing) if missing else "请输入待核查的新闻或声明。"
        yield (message, "事实：未启动 · 传播：unavailable · 仲裁：未启动", "未执行请求。", "", "", {})
        return
    if text.strip().lower().startswith(("http://", "https://")):
        yield ("请先输入链接中待核查的具体声明；URL自动形成声明尚未接线。", "未启动", "未执行请求。", "", "", {})
        return
    if resume_source is not None:
        old = ResearchRequest.model_validate_json((resume_source / "request.json").read_bytes())
        if old.input_value != text:
            yield ("续查声明与原案不一致；未执行请求。", "未启动", "", "", "", {})
            return
    run_root.mkdir(parents=True, exist_ok=True)
    case = run_root / uuid4().hex
    case.mkdir()
    (case / "request.json").write_text(ResearchRequest(input_kind="text", input_value=text).model_dump_json(), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(worktree), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", ANONYMIZED_TELEMETRY="False")
    env["TEMP"] = env["TMP"] = str(run_root / "tmp")
    Path(env["TEMP"]).mkdir(exist_ok=True)
    started = monotonic()
    with (case / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([str(core_python), "-B", str(worktree / "scripts/run_research_worker.py"),
            "--case", str(case), "--root", str(run_root)] + (["--synthetic-acceptance"] if synthetic_acceptance else []) + (["--resume-source", str(resume_source)] if resume_source is not None else []), cwd=worktree, env=env,
            stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        while process.poll() is None:
            yield render_case(case, monotonic() - started)
            sleep(0.5)
        yield render_case(case, monotonic() - started, finished=True)


def _cost_label(case: Path) -> str:
    """Keep prior unknown reservations visible when rendering a continuation attempt."""
    if not (case / "cost.json").exists():
        return "费用未知（不记零）"
    amounts = json.loads((case / "cost.json").read_text(encoding="utf-8"))
    if amounts.get("known_cost_cny") is None:
        return "本次费用未知（不记零）；累计费用以原钱包为准，原未知预占继续保留。"
    label = "本次账本估算" if amounts["cost_status"] == "known" else "本次已知小计（总额未知）"
    text = f"{label} ¥{amounts['known_cost_cny']} · 累计已知 ¥{amounts['cumulative_known_cny']}/{amounts['cumulative_limit_cny']}"
    if amounts.get("historical_unknown_reserved_cny"):
        text += f" · 历史未知预占 ¥{amounts['historical_unknown_reserved_cny']}（总额仍未知）"
    return text


def _citation_cards(snapshot: ResearchSnapshot) -> str:
    """Escape extracted quotes and allow only ordinary web URLs in archive source links."""
    cards = []
    public_heading = False
    for citation in snapshot.citations:
        if citation.citation_id.startswith("public:") and not public_heading:
            cards.append("<h3>公开材料样本 W 引用 · 不计入全网风险</h3>")
            public_heading = True
        url = citation.source_url or ""
        parsed = urlsplit(url)
        label = escape(citation.source_label)
        link = (f"<a href='{escape(url, quote=True)}' target='_blank' rel='noopener noreferrer'>{label}</a>"
                if parsed.scheme in {"https", "http"} and parsed.netloc else label)
        cards.append(f"<article class='source-card'><b>{escape(citation.evidence_id)} · {escape(citation.validation)}</b>"
                     f"<p>{link}</p><pre>{escape(citation.quote)}</pre></article>")
    return "".join(cards) or "尚无已核验引用。"


def _existing_statement(root: Path, text: str) -> tuple | None:
    """Read an existing result before credentials or worker startup can incur costs."""
    from src.yuqing.serve.research_history import ResearchHistory
    history = ResearchHistory(root)
    identity = history.find_statement(text)
    if identity is None:
        return None
    result = history.load(identity)
    return ("**已打开相同声明的历史结果，未重复调用API。**\n\n" + result[0], *result[1:])
