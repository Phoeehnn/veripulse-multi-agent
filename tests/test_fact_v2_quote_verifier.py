"""V2 规范化引文位置与交付范围的离线测试。"""

from src.yuqing.harness.quote_verifier import locate_quote, normalize_quote_text


def test_nfkc_whitespace_match_maps_to_raw_half_open_coordinates() -> None:
    """兼容字符及换行规范化后仍保留原文位置和有限上下文切片依据。"""
    original = "公告：该机构发布了全\u3000新\n正式通报。"
    quote = "该机构发布了全 新 正式通报"
    result = locate_quote(original, quote, read_end=len(original))
    assert result.status == "matched"
    assert original[result.start:result.end] == "该机构发布了全\u3000新\n正式通报"
    assert normalize_quote_text(original[result.start:result.end]) == result.normalized_text


def test_quote_cannot_use_truncated_tail_or_join_different_evidence() -> None:
    """只检查传入的单个已交付正文前缀，尾部及另一个编号不参与匹配。"""
    text = "前段内容足够长，但未到尾部证据。"
    visible_end = text.index("尾部")
    hidden = locate_quote(text, "尾部证据", read_end=visible_end)
    first_text = "第一份材料提供线索"
    first = locate_quote(first_text, "材料提供线索第二份", read_end=len(first_text))
    assert hidden.status == "not_found"
    assert first.status == "not_found"


def test_nfkc_match_without_exact_raw_span_is_reported_unresolved() -> None:
    """兼容字符展开后的局部命中无法伪造精确原文区间。"""
    result = locate_quote("ﬃx", "f", read_end=len("ﬃx"))
    assert result.status == "offset_unresolved"
    assert result.start is None and result.end is None


def test_repeated_quote_chooses_earliest_raw_position_and_marks_ambiguity() -> None:
    """相同引文出现多次时选择最早原文范围并如实标记歧义。"""
    text = "该机构发布了公告。随后该机构发布了公告。"
    quote = "该机构发布了公告"
    result = locate_quote(text, quote, read_end=len(text))
    assert result.status == "matched" and result.start == 0
    assert result.end == len(quote) and result.ambiguous
