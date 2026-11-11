"""每次运行独立保存证据与真实阅读记录，不接受模型自报已读。"""

from hashlib import sha256

from src.yuqing.schemas import EvidenceChunk, LedgerEntry


class EvidenceLedger:
    """正文相同共用编号；读取状态只能由可信工具执行器更新。"""

    def __init__(self) -> None:
        """为一次运行建立空台账，禁止跨题共享实例。"""
        self._entries: dict[str, LedgerEntry] = {}
        self._signatures: dict[str, str] = {}

    def add(self, chunks: list[EvidenceChunk]) -> list[str]:
        """登记检索片段并返回同顺序编号，仅登记不会标记已读。"""
        if any(not chunk.text.strip() for chunk in chunks):
            raise ValueError("不能登记空白正文")
        evidence_ids = []
        for chunk in chunks:
            signature = sha256(chunk.text.encode("utf-8")).hexdigest()
            evidence_id = self._signatures.get(signature)
            if evidence_id is None:
                evidence_id = f"E{len(self._entries) + 1}"
                self._entries[evidence_id] = LedgerEntry(
                    evidence_id=evidence_id, text=chunk.text,
                    sources=[chunk.model_copy(deep=True)],
                )
                self._signatures[signature] = evidence_id
            else:
                entry = self._entries[evidence_id]
                if entry.text != chunk.text:
                    raise ValueError("正文签名冲突")
                source = (chunk.chunk_id, chunk.source_doc, chunk.position)
                known = {(item.chunk_id, item.source_doc, item.position)
                         for item in entry.sources}
                if source not in known:
                    self._entries[evidence_id] = entry.model_copy(update={
                        "sources": entry.sources + [chunk.model_copy(deep=True)],
                    })
            evidence_ids.append(evidence_id)
        return evidence_ids

    def evidence_count(self) -> int:
        """只读返回去重正文数量；同正文新增来源不增加此数量。"""
        return len(self._entries)

    def get(self, evidence_id: str) -> LedgerEntry | None:
        """返回独立快照，外部修改快照不会改变内部来源或读取状态。"""
        entry = self._entries.get(evidence_id)
        return entry.model_copy(deep=True) if entry is not None else None

    def fork(self) -> "EvidenceLedger":
        """建立独立临时台账，准备结果期间不修改正式证据与已读状态。"""
        staged = EvidenceLedger()
        staged._entries = {key: entry.model_copy(deep=True) for key, entry in self._entries.items()}
        staged._signatures = dict(self._signatures)
        return staged

    def commit_from(self, staged: "EvidenceLedger") -> None:
        """审计成功后一次替换正式状态，后续修改临时台账不会影响正式台账。"""
        snapshot = staged.fork()
        self._entries, self._signatures = snapshot._entries, snapshot._signatures

    def record_read(self, evidence_id: str, returned_text: str) -> None:
        """成功返回正文后登记实际前缀范围，空白、摘要或伪造正文拒绝。"""
        entry = self._entries.get(evidence_id)
        if entry is None:
            raise ValueError(f"证据不存在：{evidence_id}")
        if not returned_text.strip() or not entry.text.startswith(returned_text):
            raise ValueError("已读内容必须是该片段实际返回的非空正文前缀")
        self._entries[evidence_id] = entry.model_copy(update={
            "read_end": max(entry.read_end, len(returned_text)),
        })
