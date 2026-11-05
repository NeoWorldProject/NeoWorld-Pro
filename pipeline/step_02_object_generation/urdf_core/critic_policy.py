"""Preserve semantic judgments while separating observations from asset defects."""
from __future__ import annotations
import math

POLICY = 'kinematic_evidence_v3'
EVIDENCE_ROUNDS = 1
ISSUE_TYPES = {'none', 'asset_error', 'needs_evidence', 'out_of_scope', 'observation'}


def component(value):
    kind = value.get('issue_type')
    if kind not in ISSUE_TYPES:
        raise ValueError('Unknown component issue_type')
    if kind == 'observation':
        if value.get('verdict') not in {'pass', 'fail', 'uncertain'}:
            raise ValueError('Observation must preserve a valid semantic verdict')
    else:
        expected = {'none': 'pass', 'asset_error': 'fail', 'needs_evidence': 'uncertain', 'out_of_scope': 'pass'}[kind]
        if value.get('verdict') != expected:
            raise ValueError('Component verdict contradicts its issue_type')
    return kind


def accepted(value):
    if value.get('review_policy') == POLICY:
        return value.get('accepted_for_pipeline') is True
    return value.get('verdict') == 'pass' or value.get('passed') is True


def blender_decision(raw, allowed_refs):
    if not isinstance(raw, dict):
        raise ValueError('Blender review must be an object')
    score = raw.get('score')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 10:
        raise ValueError('Invalid Blender review score')
    findings = raw.get('findings')
    if not isinstance(findings, list):
        raise ValueError('Blender review requires findings')
    for finding in findings:
        if not isinstance(finding, dict) or finding.get('issue_type') not in ISSUE_TYPES - {'none'}:
            raise ValueError('Invalid Blender finding issue_type')
        if not isinstance(finding.get('reason'), str) or not finding['reason'].strip():
            raise ValueError('Finding requires a reason')
        refs = finding.get('evidence')
        if not isinstance(refs, list) or not refs or any(ref not in allowed_refs for ref in refs):
            raise ValueError('Finding must cite supplied frame_ref values')
    defects = [f['reason'] for f in findings if f['issue_type'] == 'asset_error']
    gaps = [f for f in findings if f['issue_type'] == 'needs_evidence']
    observations = [f for f in findings if f['issue_type'] != 'asset_error']
    verdict = raw.get('verdict')
    if verdict not in {'pass', 'fail', 'uncertain'} or defects and verdict != 'fail':
        raise ValueError('Blender verdict contradicts its defects')
    if not defects and not any(f['issue_type'] == 'observation' for f in findings):
        if verdict != ('uncertain' if gaps else 'pass'):
            raise ValueError('Blender verdict contradicts its findings')
    return {**raw, 'review_policy': POLICY, 'critical_errors': defects,
            'semantic_verdict': verdict, 'passed': verdict == 'pass',
            'accepted_for_pipeline': not defects, 'observations': observations}


def finalize_attempts(attempts):
    if len(attempts) != 1 or attempts[0].get('review_policy') != POLICY:
        raise ValueError('Current policy uses one semantic review of complete evidence')
    final = dict(attempts[0])
    final.update(semantic_verdict=final['verdict'], acceptance_basis='classified_evidence',
                 review_attempts=attempts)
    return final
