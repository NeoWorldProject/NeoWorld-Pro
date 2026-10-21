"""API reviews grounded in simulator measurements and hash-bound real images."""
from __future__ import annotations

import json
import copy
import re
from pathlib import Path

from urdf_core.run_state import file_hash, write_json

from urdf_core.paths import RESOURCE_ROOT
PROMPTS = RESOURCE_ROOT / "prompts/isaac"


class ResponseContractError(ValueError):
    """The API responded, but exhausted its syntax/schema debug allowance."""


def read_json(path):
    data = json.loads(Path(path).read_text())
    json.dumps(data, allow_nan=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def bound_file(path, digest, root, *, allow_empty=False):
    """Verify an exact file; semantic evidence requires nonempty bytes.

    Generic checkpoint inventories may also contain valid empty logs, locks or
    package markers. Their path and hash checks remain identical.
    """
    path = Path(path).resolve()
    if (not path.is_relative_to(Path(root).resolve()) or not path.is_file()
            or (not allow_empty and not path.stat().st_size)):
        raise ValueError(f"Missing/outside physics evidence: {path}")
    if not isinstance(digest, str) or file_hash(path) != digest:
        raise ValueError(f"Physics evidence changed: {path}")
    return path


def ask_json(ask, api, stage, content, validate=lambda x: x, repairs=2):
    """Repair syntax/contract only; never ask for a better semantic decision."""
    original = list(content)
    for attempt in range(repairs + 1):
        raw = ask(f"{stage}_try_{attempt + 1}", content)
        try:
            result = api.extract_json(raw)
            json.dumps(result, allow_nan=False)
            return validate(result)
        except (ValueError, KeyError, TypeError) as error:
            if attempt == repairs:
                raise ResponseContractError(f"{stage}: response contract repair budget exhausted: {error}") from error
            content = [*original, api.make_text_input("Repair only this response's JSON/required fields; preserve your substantive judgment.\n" + raw),
                       api.make_text_input(str(error))]


def ask_initial_with_budget(ask, api, stage, content, validate, budget, *, contract_patches=False):
    """Initial content is free; every corrected initial proposal uses physics budget."""
    from urdf_core.delivery_policy import BudgetExhausted
    feedback, previous = [], None
    for index in range(budget.limit + 1):
        identity = f'{stage}_repair_{index}'
        if index:
            budget.reserve(identity, 'physics_initial_content', {'stage': stage, 'index': index})
        name = stage if index == 0 else identity
        raw = ask(f'{name}_try_1', [*content, *feedback])
        try:
            proposal = api.extract_json(raw)
            json.dumps(proposal, allow_nan=False)
            if contract_patches and previous is not None and isinstance(proposal, dict) and 'patches' in proposal:
                proposal = apply_contract_patch(previous, proposal)
            if isinstance(proposal, dict):
                previous = copy.deepcopy(proposal)
            value = validate(proposal)
        except (ValueError, KeyError, TypeError) as error:
            if index:
                budget.complete(identity, status='rejected', error=str(error))
            feedback = [api.make_text_input(
                'Repair only this response\'s JSON/required fields; preserve your substantive judgment.\n' + str(error))]
            if contract_patches and previous is not None:
                feedback.extend([
                    api.make_text_input('Previous complete proposal:\n' + json.dumps(previous)),
                    api.make_text_input(
                        'Return only {"patches":[{"op":"add|replace|remove","path":'
                        '"/profile/...","value":...}]} using JSON Pointer paths. '
                        'Patch only profile, effort_budgets or orientation_policy. '
                        'Include value for add/replace. Preserve unaffected fields. '
                        'The complete patched proposal will undergo the same validation.')])
            else:
                feedback.append(api.make_text_input('Previous response:\n' + raw))
        else:
            if index:
                budget.complete(identity, status='validated')
            return value
    raise BudgetExhausted('Physics initial content exhausted the shared refinement budget')


def apply_contract_patch(previous, proposal):
    """Apply a small field correction before running the complete validator."""
    if set(proposal) != {'patches'} or not isinstance(proposal['patches'], list) or not proposal['patches']:
        raise ValueError('Expected nonempty contract patches')
    result = copy.deepcopy(previous)
    def array_index(token, values, *, add=False):
        if add and token == '-':
            return len(values)
        if not re.fullmatch(r'0|[1-9][0-9]*', token):
            raise ValueError('Invalid contract patch array index')
        index = int(token)
        if index >= len(values) + int(add):
            raise ValueError('Contract patch array index is out of range')
        return index

    for patch in proposal['patches']:
        if not isinstance(patch, dict):
            raise ValueError('Contract patch must be an object')
        op, path = patch.get('op'), patch.get('path')
        if (op not in {'add', 'replace', 'remove'} or not isinstance(path, str)
                or not path.startswith('/') or set(patch) != (
                    {'op', 'path'} if op == 'remove' else {'op', 'path', 'value'})):
            raise ValueError('Malformed contract patch')
        parts = path[1:].split('/')
        for part in parts:
            if re.search(r'~(?![01])', part):
                raise ValueError('Invalid JSON Pointer escape')
        parts = [p.replace('~1', '/').replace('~0', '~') for p in parts]
        if parts[0] not in {'profile', 'effort_budgets', 'orientation_policy'}:
            raise ValueError('Contract patch is outside parameter fields')
        parent = result
        for key in parts[:-1]:
            if isinstance(parent, list):
                parent = parent[array_index(key, parent)]
            elif isinstance(parent, dict) and key in parent:
                parent = parent[key]
            else:
                raise ValueError('Contract patch parent does not exist')
        key = parts[-1]
        if isinstance(parent, list):
            index = array_index(key, parent, add=op == 'add')
            if op == 'add':
                parent.insert(index, copy.deepcopy(patch['value']))
            elif op == 'remove':
                del parent[index]
            else:
                parent[index] = copy.deepcopy(patch['value'])
            continue
        if not isinstance(parent, dict) or (op in {'replace', 'remove'} and key not in parent):
            raise ValueError('Contract patch target does not exist')
        if op == 'remove':
            del parent[key]
        else:
            parent[key] = copy.deepcopy(patch['value'])
    return result


def evidence_frames(report_path, maximum_per_trial=7):
    """Select real initial/extreme/end frames, without silently dropping a trial."""
    report_path = Path(report_path).resolve()
    report = read_json(report_path)
    result = []
    seen = set()
    trials = list(report.get('trials', []))
    reload_trial = report.get('reload_check', {}).get('trial')
    if reload_trial and reload_trial.get('passed') is False:
        trials.append(reload_trial)
    for trial in trials:
        identity = trial["id"]
        if identity in seen:
            raise ValueError("Duplicate physics trial")
        seen.add(identity)
        frames = trial.get("frames", [])
        from .deferred_capture import POLICY as DEFERRED_POLICY, NUMERIC_DELIVERY_POLICY
        if (report.get('media_policy') in {DEFERRED_POLICY, NUMERIC_DELIVERY_POLICY} and not report.get('passed')
                and trial.get('recording_mode') == 'numeric_only' and trial.get('passed') is True
                and not frames):
            continue  # Measurements remain in evidence; only failed trials need images.
        if len(frames) < 2:
            raise ValueError(f"Physics trial has no actual motion captures: {identity}")
        times = [float(f["time_seconds"]) for f in frames]
        if times != sorted(times) or times[0] == times[-1]:
            raise ValueError("Physics frame times must advance")
        if trial.get("capture_retention") is not None:
            from .capture_retention import verify_retained_captures
            verify_retained_captures(trial, report_path.parent)
            for frame in frames:
                path = bound_file(frame["path"], frame["sha256"], report_path.parent)
                result.append({**frame, "path": str(path), "trial_id": identity,
                               "frame_ref": f"{identity}::{frame['frame_index']}"})
            continue
        indices = {0, len(frames) - 1}
        name = trial.get("joint_name")
        if name:
            qs = [float(f.get("positions", {}).get(name, 0)) for f in frames]
            indices.update((min(range(len(qs)), key=qs.__getitem__), max(range(len(qs)), key=qs.__getitem__)))
        indices.update(round(i * (len(frames) - 1) / (maximum_per_trial - 1)) for i in range(maximum_per_trial))
        failures = [i for i, frame in enumerate(frames) if frame.get("geometry_passed") is False]
        if failures:
            indices.update((max(0, failures[0] - 1), failures[0], min(len(frames) - 1, failures[0] + 1), failures[-1]))
        for i in sorted(indices):
            frame = frames[i]
            path = bound_file(frame["path"], frame["sha256"], report_path.parent)
            result.append({**frame, "path": str(path), "trial_id": identity,
                           "frame_ref": f"{identity}::{frame['frame_index']}"})
    if not result:
        raise ValueError("Physics report has no captured evidence")
    return result


def frame_content(api, frames):
    content = []
    for frame in frames:
        label = {k: v for k, v in frame.items() if k not in {"path", "sha256"}}
        content.extend([api.make_text_input(json.dumps(label)), api.make_image_input(Path(frame["path"]))])
    return content
