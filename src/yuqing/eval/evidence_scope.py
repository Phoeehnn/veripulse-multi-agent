"""显式评测变体：在冻结事实合同之上追加配置规定的证据范围约束。"""

from hashlib import sha256
from pathlib import Path

from src.yuqing.graph.fact_adapter import FactNodeAdapter
from src.yuqing.settings import PROJECT_ROOT


PLAYBOOK_PATH = "experiments/archive/prompts/fact_check/evidence_scope_playbook.draft.md"
PLAYBOOK_SHA256 = "4a45464ef4d37badb9f5c4d8224473f4516541d19aef58622223fbf8a2c81209"


class EvidenceScopeFactAdapter(FactNodeAdapter):
    """仅供显式实验装配；默认图仍使用原来的FactNodeAdapter。"""

    def __init__(self, *, root: Path = PROJECT_ROOT) -> None:
        """先核验冻结基线，再核验批准字节，拒绝悄悄换规则。"""
        super().__init__("v2", root=root)
        raw = (self.root / PLAYBOOK_PATH).read_bytes()
        if sha256(raw).hexdigest() != PLAYBOOK_SHA256:
            raise ValueError("evidence_scope_playbook_not_approved")
        self.base_system_sha256 = sha256(self.spec.system_prompt.encode("utf-8")).hexdigest()
        composed = self.spec.system_prompt + "\n\n" + raw.decode("utf-8")
        self.spec = self.spec.model_copy(update={"system_prompt": composed})

    def fingerprints(self) -> dict[str, str]:
        """为单Agent循环与图节点提供相同的实验条件指纹。"""
        return {"variant": "evidence-scope-v1", "base_system_sha256": self.base_system_sha256,
            "playbook_sha256": PLAYBOOK_SHA256,
            "system_sha256": sha256(self.spec.system_prompt.encode("utf-8")).hexdigest()}
