"""Core-environment worker for a UI-owned investigation directory."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from decimal import Decimal
from pathlib import Path
from src.yuqing.serve.research_runner import run_live_research, run_research_resume, publish_stage


def main() -> None:
    """Run one case and expose a controlled failure category without credential messages."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume-source", type=Path)
    mode.add_argument("--synthetic-acceptance", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.synthetic_acceptance:
            from scripts.research_synthetic_acceptance import run_fixture
            run_fixture(arguments.case, arguments.root)
        elif arguments.resume_source is not None:
            run_research_resume(arguments.case, arguments.root, arguments.resume_source, limit_cny=Decimal("5"))
        else:
            run_live_research(arguments.case, arguments.root, limit_cny=Decimal("5"))
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else ""
        labels = {"synthetic_permission_mismatch": "离线验收声明指纹不匹配",
                  "research_cumulative_budget_exhausted": "累计额度不足，需要审核后才能继续",
                  "web_retrieval_failed_or_usage_unknown": "网页取证失败或费用未知",
                  "invalid_credit_conversion": "搜索credit换算配置无效",
                  "model_cache_must_be_on_d_drive": "模型缓存必须配置在D盘"}
        publish_stage(arguments.case, "调查停止：" + labels.get(reason, type(exc).__name__), "failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
