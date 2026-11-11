"""Explicit M6b graph binding; keep verified propagation rounds and publish atomically."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any
from uuid import uuid4

from src.yuqing.collab.board import PropagationReferenceSnapshot, _issue, _next_issue_number, _report_sha256, build_board
from src.yuqing.collab.propagation_delivery import read_propagation_delivery, read_qualified_propagation
from src.yuqing.collab.propagation_followup import PropagationFollowupHarnessConfig, prepare_propagation_followup, run_propagation_m6b
from src.yuqing.collab.progress import DeliveredEvidence, EvidenceDeliverySnapshot, evaluate_progress
from src.yuqing.graph.runtime import PendingFollowupIntent, _bound_apply, _bound_merge, _propagation_patch
from src.yuqing.harness.trace import record_event, use_run_context
from src.yuqing.schemas import Issue, IssueEvent, Progress, PropagationReviewRun, TaskResult
from src.yuqing.features.propagation import CommentItem, SpreadStats
from src.yuqing.schemas import PropagationReviewTask, PropagationReviewTaskItem, ReviewResult
from src.yuqing.settings import (get_m5_case_budget_config, get_m6c_case_budget_config,
    get_m6c_review_harness_config, get_propagation_harness_config)


class PropagationCollaboration:
    """Invocation-owned source, claimed task and unpublished candidate; never stored in State."""

    def __init__(self, composition: Any, *, m6c: bool = False) -> None:
        """Bind one case and derive its effective cap from existing approved configurations."""
        if composition.inputs.propagation_contract_version != 'v1.1':
            raise ValueError('m6b_requires_propagation_v11')
        self.case, self.source, self.pending, self.candidate = composition, None, None, None
        self.m6c = m6c
        budget = get_m6c_case_budget_config() if m6c else get_m5_case_budget_config()
        key = 'propagation_investigate'
        self.cap = min(Decimal(str(get_propagation_harness_config().max_cost_cny)),
            Decimal(str(budget.action_caps[key])))

    def bind(self, bindings: Any) -> Any:
        """Replace only the explicitly enabled M6b edges of the existing composition."""
        self.bindings = bindings
        return replace(bindings, propagation_round1=self.first, propagation_followup=self.followup,
            propagation_review=self.review if self.m6c else None,
            claim_propagation=self.claim, apply_result=self.apply,
            merge_round=lambda state: _bound_merge(state, self.case.context, self.bindings, self.source))

    def _read(self, result: Any, call: Any, plan: Any = None) -> Any:
        """Use one external archive boundary for first-round sources and qualified candidates."""
        args = dict(case_id=self.case.inputs.case_id, sample=call.sample, ctx=call.context,
            as_of=call.as_of, wallet=self.case.inputs.wallet, identity=call.experiment_identity)
        if plan is None:
            args["allowed_pending"] = self._registered_initial_fact_runs()
        if result is None:
            return read_propagation_delivery(**args)
        return read_qualified_propagation(result, **args, followup=plan)

    def _registered_initial_fact_runs(self) -> dict[str, dict[str, str]]:
        """Allow only pending requests belonging to this composition's registered first fact run."""
        return {run_id: {"run_id": run_id, "sample_id": self.case.inputs.sample.sample_id,
            "group_id": self.case.inputs.group_id, "variant": self.case.inputs.variant}
            for run_id, binding in self.case.runs.items()
            if binding.role == "initial" and binding.target == "fact"
            and binding.action_id is None and binding.action_kind is None}

    def first(self, state: dict) -> dict:
        """Run the explicit provenance-aware first round without changing default graph calls."""
        call = self.case.propagation_call(state)
        self.case.context.register_usage_context(call.context)
        result = run_propagation_m6b(call.spec, call.sample, call.context, node_config=call.node_config,
            worker_scope=call.worker_scope, as_of=call.as_of, coordinator_factory=call.coordinator_factory,
            experiment_identity=call.experiment_identity, case_id=self.case.inputs.case_id)
        self.source = self._read(result, call)
        if result is not None:
            self.case.context.store_propagation(result)
            self.case.context._propagation_rounds[1] = result
        patch = _propagation_patch(result, state)
        for row in patch.get("prop_rounds", []):
            row["round"] = 1
        return patch

    def unreturned_count(self) -> int:
        """Count unseen original comment identities without shrinking the first-round baseline."""
        returned = set(c.comment_id for c in self.source.comments) if self.source else set()
        prop = self.case.context._propagation_result
        if prop is not None:
            returned.update(c.comment_id for c in prop.comments)
        return len(self.source.visible_ids - returned) if self.source else 0

    def snapshot(self, snapshot: Any, state: dict) -> Any:
        """Use all visible IDs for availability, and only delivered IDs for citation history."""
        visible = self.source.visible_ids if self.source is not None else frozenset()
        returned = snapshot.target.returned_comment_ids
        target = replace(snapshot.target, propagation_has_data=self.source is not None and bool(visible),
            visible_comment_ids=visible, returned_comment_ids=returned)
        config = replace(snapshot.config, targets=snapshot.config.targets | {'propagation'})
        action = (state.get('decisions') or [{}])[-1].get('action') or {}
        cap_key = 'propagation_review' if action.get('kind') == 'review' else 'propagation_investigate'
        budget = get_m6c_case_budget_config() if self.m6c else get_m5_case_budget_config()
        action_cap = Decimal(str(budget.action_caps[cap_key]))
        if action.get('kind') == 'investigate':
            action_cap = min(action_cap, Decimal(str(get_propagation_harness_config().max_cost_cny)))
        target = replace(target, propagation_report_exists=(
            self.case.context._propagation_result is not None))
        snapshot = replace(snapshot, target=target)
        budget_view = replace(snapshot.budget, action_cap=action_cap) if action.get('target') == 'propagation' else snapshot.budget
        return replace(snapshot, target=target, config=config, budget=budget_view)

    def claim(self, action: Any, snapshot: Any, references: Any, checked: Any) -> None:
        """Consume the one gate receipt before any assignment or node request mutates its view."""
        if self.pending is not None or self.source is None or not checked.allowed:
            raise ValueError('m6b_claim_unavailable')
        if action.kind == 'review' and not self.m6c:
            raise ValueError('m6c_review_not_enabled')
        budget = get_m6c_case_budget_config() if self.m6c else get_m5_case_budget_config()
        cap_key = 'propagation_review' if action.kind == 'review' else 'propagation_investigate'
        self.cap = min(Decimal(str(get_propagation_harness_config().max_cost_cny)),
                       Decimal(str(budget.action_caps[cap_key])))
        intent = PendingFollowupIntent(action.model_copy(deep=True), uuid4().hex,
            len(snapshot.executed_actions) + 1, snapshot.decision_round)
        self.pending = (intent, snapshot, references, checked, self.case.context.current_issues())
        self.candidate = None
        self.case.context._pending_intent = intent

    def followup(self, state: dict) -> dict:
        """Run one claimed propagation task; leave the old report published until apply_result."""
        if self.pending is not None and self.pending[0].action.kind == 'review':
            raise ValueError('m6c_review_requires_review_node')
        if self.pending is None:
            raise ValueError('m6b_claim_missing')
        intent, snapshot, refs, checked, issues = self.pending
        if (state.get('pending_task') or {}).get('action_id') != intent.action_id:
            raise ValueError('m6b_pending_task_mismatch')
        ctx = self.case._run('action', 'propagation', {**state, '_program_action_id': intent.action_id})
        self.case.context.register_usage_context(ctx)
        config = get_propagation_harness_config().model_copy(update={
            'max_steps': min(get_propagation_harness_config().max_steps,
                PropagationFollowupHarnessConfig.model_json_schema()['properties']['max_steps']['maximum']),
            'max_cost_cny': float(self.cap)})
        plan = prepare_propagation_followup(source=self.source, sample=self.case.inputs.sample,
            target=ctx, action=intent.action, claimed=self.case.runs[ctx.run_id], snapshot=snapshot,
            references=refs, issues=issues, node_config=config, as_of=self.case.inputs.as_of, checked=checked)
        result = run_propagation_m6b(self.case.inputs.propagation_spec, self.case.inputs.sample, ctx,
            node_config=config, worker_scope=self.case.inputs.propagation_worker_scope,
            as_of=self.case.inputs.as_of, coordinator_factory=self.case.coordinator,
            experiment_identity=self.case.identity(ctx), followup=plan, case_id=self.case.inputs.case_id)
        if result is not None:
            call = replace(self.case.propagation_call_view(ctx), node_config=config)
            delivery = self._read(result, call, plan)
            self.candidate = (result, delivery)
        row = {'round': self.source.round_no + 1, 'status': result.stop_reason if result else 'failed',
            'run_id': ctx.run_id, 'trace_path': str(ctx.trace_path)}
        return {'prop_rounds': [row]}

    def review(self, state: dict) -> dict:
        """Run only the claimed propagation review with its own bounded budget profile."""
        if not self.m6c or self.pending is None or self.pending[0].action.kind != 'review':
            raise ValueError('m6c_review_claim_missing')
        intent, snapshot, _, _, issues = self.pending
        task = self._review_task(intent, issues)
        ctx = self.case._run('action', 'propagation', {**state, '_program_action_id': intent.action_id})
        self.case.context.register_usage_context(ctx)
        from src.yuqing.agents.propagation import run_propagation_review
        current = self.case.context._propagation_result
        if current is None:
            raise ValueError('m6c_review_report_missing')
        result = run_propagation_review(self.case.inputs.propagation_review_spec,
            self.case.inputs.sample, ctx, node_config=get_m6c_review_harness_config(),
            worker_scope=self.case.inputs.propagation_worker_scope, review_task=task,
            current_report=current.report,
            allowed_peer_refs=frozenset(ref for item in task.items for ref in item.peer_refs),
            as_of=self.case.inputs.as_of, coordinator_factory=self.case.coordinator,
            experiment_identity=self.case.identity(ctx),
            contract_version=self.case.inputs.propagation_contract_version)
        self.candidate = ('review', result)
        return {'pending_task': state.get('pending_task')}

    def _review_task(self, intent: PendingFollowupIntent, issues: tuple) -> PropagationReviewTask:
        """Build review fields and counterpart findings from current trusted case state."""
        report = self.case.context._propagation_result.report
        selected = [item for item in issues if item.issue_id in intent.action.issue_ids]
        from src.yuqing.schemas import FactVerdict
        facts = self.case.context.fact_snapshots()
        counterpart = FactVerdict.model_validate_json(facts[-1].legacy_projection_json) if facts else None
        findings = [item.text for item in counterpart.key_facts] if counterpart else []
        fact_refs = {f"fact:F{facts[-1].scope.fact_round_no}:{ref}"
            for item in counterpart.key_facts for ref in item.evidence_ids} if counterpart else set()
        items = [PropagationReviewTaskItem(field=field, old_value=getattr(report, field),
            current_basis_refs=self._current_basis_refs(report, field),
            peer_findings=[*[item.text for item in selected], *findings],
            peer_refs=sorted({f'issue:{item.issue_id}' for item in selected}
                | fact_refs | {ref for issue in selected for ref in issue.origin_refs + issue.evidence_refs
                   if ref.startswith(('fact:', 'issue:'))}))
            for field in intent.action.review_fields]
        if not items or not selected:
            raise ValueError('m6c_review_task_empty')
        return PropagationReviewTask(action_id=intent.action_id, items=items)

    def _current_basis_refs(self, report: Any, field: str) -> list[str]:
        """Show the prior report's own citations without treating peer references as evidence."""
        if field == 'narrative':
            return list(report.narrative_basis)
        return sorted({item.source for item in report.emotion_signals if item.source != 'claim'})

    def apply(self, state: dict) -> dict:
        """Prepare every result first; publish report, references, board and task as one graph patch."""
        if self.pending is None:
            return _bound_apply(state, self.case.context)
        intent = self.pending[0]
        if intent.action.kind == 'review':
            return self._apply_review(state, intent)
        before = self.case.context.current_issues()
        old = self.case.context._propagation_result
        result, delivery = self.candidate if self.candidate is not None else (None, None)
        after, added_leads, merged = self._candidate_board(before, old, result, delivery, intent)
        progress, new_refs = self._progress(before, after, delivery, intent)
        task = TaskResult(action_id=intent.action_id, kind='investigate',
            target='propagation', status='completed' if result else 'failed',
            stop_reason=result.stop_reason if result else 'no_valid_result', new_refs=list(new_refs),
            changed_fields=[key for key in ('narrative', 'emotion_level', 'spread_level')
                if result and (old is None or getattr(old.report, key) != getattr(result.report, key))],
            issue_updates=[event for issue in after for event in issue.history
                if event.action_id == intent.action_id],
            validation_flags={'report_accepted': result is not None}, progress=progress,
            usage={'cost_cny': result.cost_cny if result else None}, trace_path=result.trace_path if result else None)
        patch = {'issues': [i.model_dump(mode='json') for i in after],
            'task_results': [*state.get('task_results', []), task.model_dump(mode='json')], 'pending_task': None,
            'actions': [*state.get('actions', []), {'action_id': intent.action_id, 'kind': 'investigate',
                'target': 'propagation', 'issue_ids': list(intent.action.issue_ids),
                'question': intent.action.question, 'status': task.status}]}
        if result:
            patch.update(propagation=result.report.model_dump(mode='json'),
                leads=[i.model_dump(mode='json') for i in merged.leads_by_id.values()])
        context = self.case.context
        if context._board is not None:
            context._board = replace(context._board, issues=after)
        else:
            context._no_fact_issues = after
        if result:
            context._propagation_rounds[self.source.round_no + 1] = result
            context._propagation_result = merged
        context._pending_intent = None
        self.pending = None
        return patch

    def _apply_review(self, state: dict, intent: PendingFollowupIntent) -> dict:
        """Validate and atomically publish the review, issue transitions, progress, and trace."""
        candidate = self.candidate
        run = candidate[1] if isinstance(candidate, tuple) and candidate[0] == 'review' else None
        issues_before = self.case.context.current_issues()
        report_result = self.case.context._propagation_result
        accepted = isinstance(run, PropagationReviewRun) and report_result is not None
        if not accepted:
            return self._failed_review(state, intent, issues_before)
        review = run.review
        after, changed, new_refs, progress = _apply_review_to_issues(
            issues_before, intent, review, run, report_result, self.case.inputs.case_id)
        task = TaskResult(action_id=intent.action_id, kind='review', target='propagation',
            status='completed', stop_reason=run.stop_reason, new_refs=sorted(new_refs),
            changed_fields=changed, issue_updates=[event for issue in after for event in issue.history
                if event.action_id == intent.action_id],
            validation_flags={'report_accepted': True, 'refs_valid': True, 'permissions_ok': True},
            progress=progress, usage={'requests': run.steps, 'input_output_tokens': run.input_output_tokens,
                'cost_cny': run.cost_cny}, trace_path=run.trace_path)
        patch = {'issues': [item.model_dump(mode='json') for item in after],
            'task_results': [*state.get('task_results', []), task.model_dump(mode='json')], 'pending_task': None,
            'reviews': [*state.get('reviews', []), review.model_dump(mode='json')],
            'actions': [*state.get('actions', []), {'action_id': intent.action_id, 'kind': 'review',
                'target': 'propagation', 'issue_ids': list(intent.action.issue_ids),
                'question': intent.action.question, 'status': 'completed'}]}
        report = _reviewed_report(report_result.report, review)
        comments = {item.comment_id: item for item in report_result.comments}
        comments.update({item['comment_id']: CommentItem(**item) for item in run.comments})
        merged = report_result.model_copy(update={
            'report': report, 'comments': list(comments.values()),
            'stats': SpreadStats(**run.stats) if run.stats is not None else report_result.stats})
        if report != report_result.report:
            patch['propagation'] = report.model_dump(mode='json')
        if any(item.decision == 'disagree' for item in review.items):
            patch['m6c_force_review'] = True
        trace_context = self.case.context._usage_contexts[run.run_id]
        with use_run_context(trace_context):
            record_event('m6c_review_applied', {'action_id': intent.action_id,
                'decisions': [item.decision for item in review.items], 'changed_fields': changed,
                'progress': progress.model_dump(mode='json')})
        self.case.context._propagation_result = merged
        if self.case.context._board is not None:
            self.case.context._board = replace(self.case.context._board, issues=after)
        else:
            self.case.context._no_fact_issues = after
        self.case.context._pending_intent = None
        self.pending, self.candidate = None, None
        return patch

    def _failed_review(self, state: dict, intent: PendingFollowupIntent,
                       issues_before: tuple[Issue, ...]) -> dict:
        """Record a failed review attempt while retaining the last accepted propagation report."""
        after = tuple(_review_workflow_failed(item, intent) if item.issue_id in intent.action.issue_ids
                      else item for item in issues_before)
        task = TaskResult(action_id=intent.action_id, kind='review', target='propagation',
            status='failed', stop_reason='review_submission_missing',
            issue_updates=[event for issue in after for event in issue.history
                if event.action_id == intent.action_id],
            validation_flags={'report_accepted': False},
            progress=Progress(substantive=False, reasons=['task_failed']), usage={})
        if self.case.context._board is not None:
            self.case.context._board = replace(self.case.context._board, issues=after)
        else:
            self.case.context._no_fact_issues = after
        patch = {'issues': [item.model_dump(mode='json') for item in after],
            'task_results': [*state.get('task_results', []), task.model_dump(mode='json')], 'pending_task': None,
            'actions': [*state.get('actions', []), {'action_id': intent.action_id, 'kind': 'review',
                'target': 'propagation', 'issue_ids': list(intent.action.issue_ids),
                'question': intent.action.question, 'status': 'failed'}]}
        self.case.context._pending_intent = None
        self.pending, self.candidate = None, None
        return patch

    def _candidate_board(self, before: tuple, old: Any, result: Any, delivery: Any,
                         intent: Any) -> tuple:
        """Reuse the existing narrative mapping and append stable case-level lead identities."""
        leads = dict(old.leads_by_id) if old else {}
        additions = []
        state, refs = None, []
        if result:
            snapshot = PropagationReferenceSnapshot(self.case.inputs.case_id, delivery.round_no, True, True,
                frozenset(c.comment_id for c in delivery.comments), _report_sha256(result.report))
            built = build_board(case_id=self.case.inputs.case_id, round_no=delivery.round_no, fact_verdict=None,
                fact_status='not_dispatched', fact_refs=None, propagation_report=result.report, propagation_refs=snapshot)
            mapped = next(i for i in built.issues if i.kind == '传播性质')
            state, refs = mapped.epistemic, mapped.evidence_refs
            known = {(i.source, ''.join(i.claim_to_verify.split())) for i in leads.values()}
            for lead in result.report.leads:
                key = (lead.source, ''.join(lead.claim_to_verify.split()))
                if key not in known:
                    lid = 'L' + str(max((int(key[1:]) for key in leads), default=0) + 1)
                    leads[lid], known = lead, known | {key}
                    additions.append(_issue('I' + str(_next_issue_number([*before, *additions])), lead.claim_to_verify,
                        '线索', 'lead:' + lid, 'fact', False, 'unknown', delivery.round_no, ['prop:' + lead.source]))
        after = tuple(self._updated(i, intent, state, refs) if i.issue_id in intent.action.issue_ids else i
            for i in before) + tuple(additions)
        comments = {c.comment_id: c for c in old.comments} if old else {}
        if result:
            comments.update({c.comment_id: c for c in result.comments})
        merged = result.model_copy(update={'leads_by_id': leads, 'comments': list(comments.values())}) if result else old
        return after, additions, merged

    def _updated(self, issue: Issue, intent: Any, state: str | None, refs: list) -> Issue:
        """Update only assigned propagation issues; failures retain the evidence conclusion."""
        if issue.owner != 'propagation' or issue.kind != '传播性质':
            raise ValueError('m6b_target_issue_invalid')
        history = list(issue.history)
        workflow = 'done' if state is not None else 'failed'
        history.append(IssueEvent(round=intent.action_round, action_id=intent.action_id,
            field='workflow', old=issue.workflow, new=workflow, by='program', refs=refs))
        if state is not None and state != issue.epistemic:
            history.append(IssueEvent(round=intent.action_round, action_id=intent.action_id,
                field='epistemic', old=issue.epistemic, new=state, by='program', refs=refs))
        return Issue.model_validate({**issue.model_dump(), 'workflow': workflow,
            'epistemic': state if state is not None else issue.epistemic,
            'evidence_refs': refs if state is not None else issue.evidence_refs, 'history': history})

    def _progress(self, before: tuple, after: tuple, delivery: Any, intent: Any) -> tuple:
        """Measure body-backed progress using the original shared progress evaluator."""
        def snapshot(comments: tuple, action: str) -> EvidenceDeliverySnapshot:
            """Describe actual preview bodies without promoting renamed IDs into new evidence."""
            return EvidenceDeliverySnapshot(self.case.inputs.case_id, intent.action_round, action,
                {'prop:' + c.comment_id: DeliveredEvidence(sha256(c.text.encode()).hexdigest(), len(c.text))
                    for c in comments})
        old = snapshot(self.source.comments, intent.action_id)
        new = snapshot(delivery.comments if delivery else (), intent.action_id)
        refs = {i.issue_id: frozenset(i.evidence_refs) & new.deliveries.keys()
            for i in after if i.issue_id in intent.action.issue_ids}
        progress = evaluate_progress(before=before, after=after, target_issue_ids=intent.action.issue_ids,
            new_valid_refs=refs, case_id=self.case.inputs.case_id, round_no=intent.action_round,
            action_id=intent.action_id, before_delivery=old, action_delivery=new)
        known = {i.body_sha256 for i in old.deliveries.values()}
        return progress, tuple(k for k, value in new.deliveries.items() if value.body_sha256 not in known)


def _stats_fields(stats: dict[str, Any] | None) -> frozenset[str]:
    """Expose only returned, nonempty business statistics as reviewable source fields."""
    if not isinstance(stats, dict) or not stats.get('has_data'):
        return frozenset()
    excluded = {'has_data', 'visible_nodes', 'unavailable_fields'}
    unavailable = set(stats.get('unavailable_fields') or ())
    return frozenset(key for key, value in stats.items()
        if key not in excluded and key not in unavailable and value is not None)


def _review_deliveries(comments: list[dict[str, Any]], stats: dict[str, Any] | None
                       ) -> dict[str, DeliveredEvidence]:
    """Represent only review-run comments and available statistic values in progress checks."""
    deliveries = {'prop:' + item['comment_id']: DeliveredEvidence(
        sha256(item['text'].encode()).hexdigest(), len(item['text'])) for item in comments}
    if stats is not None:
        for key in _stats_fields(stats):
            body = json.dumps([key, stats[key]], ensure_ascii=False, sort_keys=True)
            deliveries['prop:' + key] = DeliveredEvidence(sha256(body.encode()).hexdigest(), len(body))
    return deliveries


def _review_refs(refs: list[str]) -> frozenset[str]:
    """Map the review contract's bare C/stat names into issue-ledger propagation refs."""
    return frozenset('prop:' + item for item in refs)


def _reviewed_report(report: Any, review: ReviewResult) -> Any:
    """Apply only approved interpretation fields while keeping source citations in the review record."""
    updates = {}
    for item in review.items:
        if item.decision == 'revised':
            updates[item.field] = item.new_value
            if item.field == 'narrative':
                updates['narrative_basis'] = [ref for ref in item.basis_refs
                    if ref.startswith('C') and ref[1:].isdigit()]
    return report.model_copy(update=updates) if updates else report


def _review_workflow_failed(issue: Issue, intent: PendingFollowupIntent) -> Issue:
    """Mark work failure without changing any accepted evidence conclusion."""
    event = IssueEvent(round=intent.action_round, action_id=intent.action_id,
        field='workflow', old=issue.workflow, new='failed', by='program')
    return issue.model_copy(update={'workflow': 'failed', 'history': issue.history + [event]})


def _apply_review_to_issues(before: tuple[Issue, ...], intent: PendingFollowupIntent,
                            review: ReviewResult, run: Any, current: Any, case_id: str
                            ) -> tuple[tuple[Issue, ...], list[str], frozenset[str], Progress]:
    """Record review work and evidence-backed issue state changes without editing source facts."""
    selected = set(intent.action.issue_ids)
    decisions = [item.decision for item in review.items]
    epistemic = ('conflicting' if 'disagree' in decisions else
        'supported' if 'revised' in decisions else None)
    cited = _review_refs([ref for item in review.items if item.decision in {'revised', 'disagree'}
        for ref in item.basis_refs])
    after = []
    for issue in before:
        if issue.issue_id not in selected:
            after.append(issue)
            continue
        if issue.owner != 'propagation' or issue.kind not in {'传播性质', '冲突'}:
            raise ValueError('m6c_review_issue_invalid')
        history = list(issue.history)
        history.append(IssueEvent(round=intent.action_round, action_id=intent.action_id,
            field='workflow', old=issue.workflow, new='done', by='program', refs=sorted(cited)))
        if epistemic is not None and epistemic != issue.epistemic:
            history.append(IssueEvent(round=intent.action_round, action_id=intent.action_id,
                field='epistemic', old=issue.epistemic, new=epistemic, by='program', refs=sorted(cited)))
        after.append(issue.model_copy(update={'workflow': 'done',
            'epistemic': epistemic or issue.epistemic,
            'evidence_refs': sorted(set(issue.evidence_refs) | set(cited)), 'history': history}))
    action_delivery = _review_deliveries(run.comments, run.stats)
    progress_inputs = {issue_id: frozenset(cited & action_delivery.keys()) for issue_id in selected}
    progress = _review_progress(before, tuple(after), selected, progress_inputs, intent,
        run.comments, run.stats, current, case_id)
    if any(item.decision == 'disagree' for item in review.items):
        progress = Progress(substantive=False, reasons=['review_disagree'])
    known = {value.body_sha256 for value in _review_deliveries(
        [item.model_dump(mode="json") for item in current.comments],
        current.stats.model_dump(mode="json") if current.stats else None).values()}
    new_refs = frozenset(ref for ref in cited if action_delivery[ref].body_sha256 not in known)
    return tuple(after), (['prop.' + item.field for item in review.items if item.decision == 'revised']), \
        new_refs, progress


def _review_progress(before: tuple[Issue, ...], after: tuple[Issue, ...], selected: set[str],
                     candidates: dict[str, frozenset[str]], intent: PendingFollowupIntent,
                     comments: list[dict[str, Any]], stats: dict[str, Any] | None,
                     current: Any, case_id: str) -> Progress:
    """Reuse the standard progress gate with body hashes from this review's delivered materials."""
    old_deliveries = _review_deliveries(
        [item.model_dump(mode='json') for item in current.comments],
        current.stats.model_dump(mode='json') if current.stats else None)
    delivered = _review_deliveries(comments, stats)
    before_snapshot = EvidenceDeliverySnapshot(case_id, intent.action_round,
        intent.action_id, old_deliveries)
    action_snapshot = EvidenceDeliverySnapshot(case_id, intent.action_round,
        intent.action_id, delivered)
    return evaluate_progress(before=before, after=after, target_issue_ids=tuple(selected),
        new_valid_refs=candidates, case_id=case_id,
        round_no=intent.action_round, action_id=intent.action_id,
        before_delivery=before_snapshot, action_delivery=action_snapshot)
