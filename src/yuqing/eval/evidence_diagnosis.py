"""只诊断历史证据的位置，不调用模型、不检索、不判证据语义是否充分。"""

from src.yuqing.eval.dataset import normalize_evidence_text
from src.yuqing.retrieval.chunk import chunk
from src.yuqing.schemas import EvidenceChunk

Span = tuple[int, int]


def locate_gold_spans(gold_text: str, document: str) -> list[Span]:
    """按既有归一化定位全部重叠出现，返回原文字符半开区间。"""
    needle = normalize_evidence_text(gold_text)
    if not needle:
        return []
    offsets = [index for index, char in enumerate(document) if normalize_evidence_text(char)]
    text = normalize_evidence_text(document)
    spans, start = [], 0
    while (found := text.find(needle, start)) != -1:
        spans.append((offsets[found], offsets[found + len(needle) - 1] + 1))
        start = found + 1
    return spans


def _covers(span: Span, intervals: list[Span]) -> bool:
    """只合并原文中连续或重叠的区间，禁止跨越未返回的缺口。"""
    reached = span[0]
    for start, end in sorted(intervals):
        if start > reached:
            break
        reached = max(reached, end)
        if reached >= span[1]:
            return True
    return False


def _validate_returned(documents: dict[str, str], returned: list[EvidenceChunk],
                       chunk_size: int, chunk_overlap: int) -> None:
    """逐段核对历史正文、来源、编号和位置，不用当前切片冒充历史输入。"""
    chunk("", size=chunk_size, overlap=chunk_overlap)
    seen: set[str] = set()
    for item in returned:
        if item.source_doc not in documents:
            raise ValueError("historical_source_missing")
        pieces = chunk(documents[item.source_doc], size=chunk_size, overlap=chunk_overlap)
        if (item.position < 0 or item.position >= len(pieces)
                or pieces[item.position] != item.text
                or item.chunk_id != f"{item.source_doc}:chunk:{item.position}"
                or item.chunk_id in seen):
            raise ValueError("historical_chunk_mismatch")
        seen.add(item.chunk_id)


def _classify_source(gold_text: str, source_doc: str, documents: dict[str, str],
                     returned: list[EvidenceChunk], size: int, overlap: int) -> dict[str, object]:
    """按单一来源的原文区间分A/B/C/D/E，并保留不能定位的原因。"""
    spans = locate_gold_spans(gold_text, documents.get(source_doc, ""))
    row: dict[str, object] = {"source_doc": source_doc, "category": "E",
                              "gold_spans": spans, "cross_boundary": False,
                              "reason": "gold_not_locatable"}
    if not spans:
        return row
    pieces = chunk(documents[source_doc], size=size, overlap=overlap)
    stride = size - overlap
    positions = [item.position for item in returned if item.source_doc == source_doc]
    intervals = [(position * stride, position * stride + len(pieces[position])) for position in positions]
    all_intervals = [(i * stride, i * stride + len(piece)) for i, piece in enumerate(pieces)]
    windows = [[interval for i, interval in enumerate(all_intervals) if abs(i - position) <= 1]
               for position in positions]
    for category, groups in (("A", [intervals]), ("B", windows)):
        for span in spans:
            for group in groups:
                if _covers(span, group):
                    row.update(category=category, reason="exact_text_coverage",
                               cross_boundary=not any(_covers(span, [part]) for part in group))
                    return row
    row.update(category="C" if positions else "D", reason="outside_neighbor_window" if positions else "source_not_returned",
               cross_boundary=all(not any(_covers(span, [part]) for part in all_intervals) for span in spans))
    return row


def classify_gold_evidence(gold_text: str, source_docs: list[str], documents: dict[str, str],
                           returned: list[EvidenceChunk], *, chunk_size: int,
                           chunk_overlap: int) -> dict[str, object]:
    """逐来源分类，采用最好可验证覆盖并保留其他来源差异和定位缺口。"""
    _validate_returned(documents, returned, chunk_size, chunk_overlap)
    sources = list(dict.fromkeys(source_docs))
    rows = [_classify_source(gold_text, source, documents, returned, chunk_size, chunk_overlap)
            for source in sources]
    # 其他题的资料也可能包含相同gold正文；精确覆盖不能因来源题号不同而忽略。
    extra_sources = sorted({item.source_doc for item in returned} - set(sources))
    extras = [_classify_source(gold_text, source, documents, returned, chunk_size, chunk_overlap)
              for source in extra_sources]
    direct = [row for row in extras if row["category"] == "A"]
    rank = {category: index for index, category in enumerate("ABCDE")}
    best = min(rows + direct, key=lambda row: rank[str(row["category"])]) if rows or direct else None
    return {"category": best["category"] if best else "E", "sources": rows,
            "cross_boundary": best["cross_boundary"] if best else False,
            "multiple_sources": len(sources) > 1,
            "unresolved_sources": [row["source_doc"] for row in rows if row["category"] == "E"],
            "historical_other_source_matches": direct,
            "reason": best["reason"] if best else "no_reliable_source_mapping"}


def diagnose_question(gold_evidence: dict[str, str], source_docs: dict[str, list[str]],
                      documents: dict[str, str], returned: list[EvidenceChunk], *,
                      chunk_size: int, chunk_overlap: int) -> list[dict[str, object]]:
    """一条gold一行，不把同题不同位置类别压成一个结论。"""
    if set(source_docs) - set(gold_evidence):
        raise ValueError("mapping_contains_unknown_gold")
    return [{"gold_key": key, "gold_text": text,
             **classify_gold_evidence(text, source_docs.get(key, []), documents, returned,
                                      chunk_size=chunk_size, chunk_overlap=chunk_overlap)}
            for key, text in gold_evidence.items()]
