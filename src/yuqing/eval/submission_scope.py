"""独立离线候选：在原T后显式追加命题范围或格式提醒，不注册默认入口。"""

from hashlib import sha256
from pathlib import Path
from typing import Literal

from src.yuqing.eval.conclusion_alignment import ConclusionAlignmentFactAdapter
from src.yuqing.settings import PROJECT_ROOT


_CANDIDATES = {
    "scope": ("experiments/archive/prompts/fact_check/claim_scope_v1.draft.md",
              "ce1ae9b507a69fe920a769512543e595c646200d9f198b0d5f635163de29ed8c"),
    "format": ("experiments/archive/prompts/fact_check/submission_format_v1.draft.md",
               "704d052bae64742855ec3029769fba1e07b1fb501fa5615b8d959617d4e869ca"),
}


class SubmissionScopeFactAdapter(ConclusionAlignmentFactAdapter):
    """一次仅选一个片段；未选时spec和指纹均保持原T，不产生模型调用。"""

    def __init__(self, candidate: Literal["scope", "format"] | None = None,
                 *, root: Path = PROJECT_ROOT) -> None:
        """核对所选草稿字节后只修改本实例，不授予额外工具、调用或付费权限。"""
        if candidate is not None and candidate not in _CANDIDATES:
            raise ValueError("submission_scope_candidate_invalid")
        super().__init__(root=root)
        self.candidate = candidate
        self.aligned_system_sha256 = sha256(self.spec.system_prompt.encode("utf-8")).hexdigest()
        if candidate is not None:
            path, expected = _CANDIDATES[candidate]
            raw = (self.root / path).read_bytes()
            if sha256(raw).hexdigest() != expected:
                raise ValueError("submission_scope_draft_drift")
            self.spec = self.spec.model_copy(update={
                "system_prompt": self.spec.system_prompt + "\n\n" + raw.decode("utf-8")})

    def fingerprints(self) -> dict[str, str]:
        """选中片段独立留指纹；控制仍返回原T指纹，不组合两个新因素。"""
        baseline = super().fingerprints()
        if self.candidate is None:
            return baseline
        path, expected = _CANDIDATES[self.candidate]
        return {**baseline, "variant": f"submission-{self.candidate}-v1",
                "aligned_system_sha256": self.aligned_system_sha256,
                "candidate_path": path, "candidate_sha256": expected}
