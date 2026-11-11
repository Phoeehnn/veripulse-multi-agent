"""显式评测候选：同次请求追加结论对齐草稿，不改变默认图或提交合同。"""

from hashlib import sha256
from pathlib import Path

from src.yuqing.eval.evidence_scope import EvidenceScopeFactAdapter
from src.yuqing.settings import PROJECT_ROOT


ALIGNMENT_PATH = "experiments/archive/prompts/fact_check/conclusion_alignment_v1.draft.md"
ALIGNMENT_SHA256 = "5906a2a1a86e94adc19c15028b6a87d08f2da73c907051fe4ab03ec145ebd78b"


class ConclusionAlignmentFactAdapter(EvidenceScopeFactAdapter):
    """仅在评测显式选择时构造；不注册到默认图，不赋予真实运行许可。"""

    def __init__(self, *, root: Path = PROJECT_ROOT) -> None:
        """核对原冻结合同及候选字节，只修改本实例的系统提示词。"""
        super().__init__(root=root)
        raw = (self.root / ALIGNMENT_PATH).read_bytes()
        if sha256(raw).hexdigest() != ALIGNMENT_SHA256:
            raise ValueError("conclusion_alignment_draft_drift")
        self.control_system_sha256 = sha256(self.spec.system_prompt.encode("utf-8")).hexdigest()
        self.spec = self.spec.model_copy(update={
            "system_prompt": self.spec.system_prompt + "\n\n" + raw.decode("utf-8")})

    def fingerprints(self) -> dict[str, str]:
        """分别记录控制Prompt、组合Prompt及候选指纹，便于后续单因素配对。"""
        return {**super().fingerprints(), "variant": "conclusion-alignment-v1",
            "control_system_sha256": self.control_system_sha256,
            "alignment_sha256": ALIGNMENT_SHA256}
