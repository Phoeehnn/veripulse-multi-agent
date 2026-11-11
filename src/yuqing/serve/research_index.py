"""Create isolated per-investigation retrieval stores with the unchanged retrieval stack."""
from hashlib import sha256
import json
from pathlib import Path

from src.yuqing.retrieval.vector_store import index_chunks, validate_vector_corpus
from src.yuqing.serve.research_service import load_research_chunks


def build_research_index(materials: Path, store: Path, *, sample_id: str, claim: str) -> Path:
    """Create a fresh vector store and publish readiness only after corpus validation."""
    chunks = load_research_chunks(materials, sample_id, claim)
    store.mkdir(parents=True, exist_ok=False)
    index_chunks(chunks, store)
    validate_vector_corpus(chunks, store)
    digest = sha256((materials / "chunks.json").read_bytes()).hexdigest()
    (store / "research-index.json").write_text(json.dumps({"chunks_sha256": digest,
        "sample_id": sample_id, "count": len(chunks)}, sort_keys=True), encoding="utf-8")
    return store


def verify_research_index(materials: Path, store: Path, *, sample_id: str, claim: str) -> None:
    """Reject missing, stale or foreign stores before a graph can invoke search tools."""
    chunks = load_research_chunks(materials, sample_id, claim)
    record = json.loads((store / "research-index.json").read_text(encoding="utf-8"))
    expected = {"chunks_sha256": sha256((materials / "chunks.json").read_bytes()).hexdigest(),
                "sample_id": sample_id, "count": len(chunks)}
    if record != expected:
        raise ValueError("research_index_identity_mismatch")
    validate_vector_corpus(chunks, store)


def link_cached_research_models(source: Path, target: Path) -> int:
    """Reuse existing weight files by hard link; copy only small cache refs into an isolated cache."""
    import os
    from src.yuqing.settings import get_retrieval_config
    if source.resolve() == target.resolve() or target.resolve().drive.lower() != "d:":
        raise ValueError("research_cache_requires_isolated_d_drive_target")
    config = get_retrieval_config()
    count = 0
    for model in (config.embedding_model, config.rerank_model):
        name = "models--" + model.replace("/", "--")
        original = source / "huggingface/hub" / name
        if not (original / "refs/main").is_file():
            raise ValueError("required_model_not_cached:" + model)
        for branch in ("refs", "snapshots"):
            for path in (original / branch).rglob("*"):
                if not path.is_file():
                    continue
                destination = target / "huggingface/hub" / name / path.relative_to(original)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    continue
                if branch == "refs":
                    destination.write_bytes(path.read_bytes())
                else:
                    os.link(path.resolve(), destination)
                count += 1
    return count
