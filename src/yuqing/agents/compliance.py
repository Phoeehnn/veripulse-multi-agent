"""按配置文件中的规则检查声明和本案已返回评论。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.yuqing.schemas import ComplianceResult, RuleHit
from src.yuqing.settings import PROJECT_ROOT, load_yaml_no_duplicates


class _Rule(BaseModel):
    """单条规则的严格内部结构。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    severity: Literal["高", "中"]
    keywords: list[str] = Field(default_factory=list)
    pattern: str | None = None
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_rule(self) -> _Rule:
        """只允许已确认的严重度和至少一种有效匹配条件。"""
        if not self.id.strip() or not self.category.strip() or not self.note.strip():
            raise ValueError("empty_rule_text")
        if any(not item.strip() for item in self.keywords):
            raise ValueError("empty_keyword")
        if self.pattern is not None and not self.pattern.strip():
            raise ValueError("empty_pattern")
        if not self.keywords and not (self.pattern and self.pattern.strip()):
            raise ValueError("missing_matcher")
        if self.pattern:
            re.compile(self.pattern)
        return self


def check_compliance(claim: str, comments: tuple[tuple[str, str], ...] = (), *,
                     rules_path: Path | None = None) -> ComplianceResult:
    """将声明与评论独立检查；空规则、坏规则和无评论保持不同状态。"""
    path = rules_path or PROJECT_ROOT / "config/rules_compliance.yaml"
    if any(not re.fullmatch(r"C\d+", source) for source, _ in comments):
        return ComplianceResult(claim_status="check_failed", comment_status="check_failed")
    rules, status = _load_rules(path)
    if status == "check_failed":
        return ComplianceResult(claim_status=status, comment_status="no_comments" if not comments else status)
    if status == "rules_empty":
        return ComplianceResult(claim_status=status, comment_status="no_comments" if not comments else status)
    claim_hits, claim_status = _check_text(claim, rules)
    comment_hits: list[RuleHit] = []
    comment_status = "no_comments" if not comments else "checked_no_hit"
    for source, text in comments:
        hits, status = _check_text(text, rules, source=source)
        comment_hits.extend(hits)
        if status == "check_failed":
            comment_status = status
        elif hits and comment_status != "check_failed":
            comment_status = "checked_with_hits"
    return ComplianceResult(
        claim_status=claim_status, claim_hits=claim_hits,
        comment_status=comment_status, comment_hits=comment_hits,
    )


def _check_text(text: str, rules: list[_Rule], *, source: str | None = None
                ) -> tuple[list[RuleHit], str]:
    """分别记录单份声明或评论的命中与匹配失败状态。"""
    try:
        hits = _match_text(text, rules, source=source)
        return hits, "checked_with_hits" if hits else "checked_no_hit"
    except (ValidationError, ValueError):
        return [], "check_failed"


def _load_rules(path: Path) -> tuple[list[_Rule], str]:
    """读取并整体校验规则；有一条错误就拒绝整份文件。"""
    try:
        raw = load_yaml_no_duplicates(path.read_text(encoding="utf-8"))
        if raw is None:
            return [], "rules_empty"
        if not isinstance(raw, dict) or set(raw) != {"rules"} or not isinstance(raw["rules"], list):
            return [], "check_failed"
        if not raw["rules"]:
            return [], "rules_empty"
        rules = [_Rule.model_validate(item) for item in raw["rules"]]
        if len({rule.id for rule in rules}) != len(rules):
            return [], "check_failed"
        return rules, "checked_no_hit"
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, re.error, TypeError, ValueError):
        return [], "check_failed"


def _match_text(text: str, rules: list[_Rule], *, source: str | None = None) -> list[RuleHit]:
    """每条规则在单个来源文本中最多产生命中一次。"""
    hits = []
    for rule in rules:
        spans = _matching_spans(text, rule)
        if spans is None:
            continue
        start = min(span[0] for span in spans)
        end = max(span[1] for span in spans)
        hits.append(RuleHit(rule_id=rule.id, category=rule.category, severity=rule.severity,
                            matched_text=text[start:end], source=source))
    return hits


def _matching_spans(text: str, rule: _Rule) -> list[tuple[int, int]] | None:
    """关键词组内部任一命中；关键词与正则同时配置时两组都须命中。"""
    keyword_span = next(((text.find(word), text.find(word) + len(word))
                         for word in rule.keywords if word in text), None)
    pattern_match = re.search(rule.pattern, text) if rule.pattern else None
    if pattern_match is not None and pattern_match.start() == pattern_match.end():
        raise ValueError("zero_length_regex_match")
    if rule.keywords and keyword_span is None or rule.pattern and pattern_match is None:
        return None
    spans = [span for span in (keyword_span,
        (pattern_match.start(), pattern_match.end()) if pattern_match else None) if span]
    return spans or None
