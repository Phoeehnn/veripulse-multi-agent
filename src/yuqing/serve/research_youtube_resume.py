"""Explicitly authorized three-role continuation using an unchanged source corpus and wallet."""
from pathlib import Path
from hashlib import sha256
from decimal import Decimal
import json
from src.yuqing.serve.research_youtube_activation import YoutubeActivation, request_identity, activation_fingerprint, check_grant
from src.yuqing.serve import research_runner as runner
from src.yuqing.serve.research_service import ResearchSample, assemble_research_case, research_snapshot, save_research_snapshot
from src.yuqing.retrieval.public_materials import load_public_material
from src.yuqing.retrieval.public_signal_contracts import PublicPropagationTask
from src.yuqing.harness.tools import EvidenceWorkerScope


def resume_scope(case: Path, activation: YoutubeActivation) -> dict:
    """Bind the new output, unchanged source receipts, archive bytes and current wallet."""
    source = activation.collection.case.resolve()
    if case.resolve().parent != source.parent or case.resolve() == source:
        raise ValueError('resume_output_outside_root')
    if request_identity(case)['claim'] != request_identity(source)['claim']:
        raise ValueError('resume_claim_mismatch')
    files = [source/'request.json', source/'snapshot.json', source/'model-grant.json',
        source/'youtube-collection.json', source/'index/research-index.json', activation.wallet_path]
    for directory in ('retrieval/evidence', 'public-material'):
        files.extend(sorted(p for p in (source/directory).rglob('*') if p.is_file()))
    return dict(stage='youtube_resume', identity=request_identity(case), source=request_identity(source),
        original_grant=activation.pin, runtime=activation_fingerprint(),
        files={str(p.resolve()):sha256(p.read_bytes()).hexdigest() for p in files})


def run_resume(case: Path, activation: YoutubeActivation, grant: Path, pin: str) -> None:
    """Claim one authorized continuation and preserve all source results and cumulative charges."""
    source = activation.collection.case.resolve()
    check_grant(grant, pin, resume_scope(case, activation))
    activation.validate(source, live=True)
    request = runner.ResearchRequest.model_validate_json((case/'request.json').read_bytes())
    runner.verify_research_index(source/'retrieval/evidence', source/'index',
        sample_id=source.name, claim=request.input_value)
    with (case/'youtube-resume.claim').open('x', encoding='utf8') as stream:
        stream.write(pin)
    wallet = activation.bind_wallet(runner.open_research_wallet(source.parent,
        limit_cny=activation.limit_cny, credit_cny=runner._live_research_rate()))
    fact, prop = EvidenceWorkerScope(), EvidenceWorkerScope()
    try:
        if wallet.snapshot().halted or wallet.snapshot().unknown_requests or wallet.snapshot().pending_requests:
            raise ValueError('resume_wallet_not_healthy')
        batch = load_public_material(source/'public-material',case_id=source.name,claim=request.input_value)
        task = PublicPropagationTask(sample_id=source.name,claim=request.input_value,batch=batch) if activation.use and batch.observations else None
        inputs = runner._case_inputs(case,ResearchSample(sample_id=source.name,claim=request.input_value),
            wallet,source/'index',fact,prop)
        (case/'reuse.json').write_text(json.dumps(dict(source_case=source.name,permission_sha=pin)),encoding='utf8')
        runner.publish_stage(case,'复用已核验网页、评论及索引','completed')
        runner.publish_stage(case,'Agent调查与仲裁','running')
        assembly, comp = assemble_research_case(request,inputs,source/'retrieval/evidence',
            external_calls_authorized=True,sample_bind=lambda c:activation.bind(c,task),fact_guard=activation.guard_fact)
        output = assembly.invoke()
        cost = runner.save_case_cost(case,source.name,wallet)
        snapshot = activation.project(research_snapshot(output,comp,execution_mode='live'),task,cost,wallet.snapshot().halted)
        save_research_snapshot(case,snapshot)
        runner.publish_stage(case,'Agent调查与仲裁',snapshot.status)
    finally:
        runner.save_case_cost(case,source.name,wallet)
        fact.close()
        prop.close()
        wallet.close()
