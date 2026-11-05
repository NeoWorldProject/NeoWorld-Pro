"""Scene-only recovery: no object repair actions are exposed here."""
import json
from pathlib import Path

from urdf_core.error_recovery import Recovery, error_evidence, report_error


def for_scene(root, config, api):
    def ask(stage, prompt, payload):
        return api.call(stage, prompt, json.loads(payload), include_original=False)
    return Recovery(root, 'scene', ask, limit=config.get('max_error_diagnoses', 5))


def incomplete(error):
    return {'status': 'invalid_candidate', 'passed': False, 'complete': False,
            'acceptance_complete': False, 'violation': None, 'physical_residual': None,
            'reason': 'operation_incomplete', 'error_context': error_evidence(error)}


def checked(report):
    if report.get('status') in ('unclassified_error', 'environment_error', 'infrastructure_error', 'worker_error', 'timeout'):
        report_error(report)
    return report
