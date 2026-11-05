"""Keep operator review prose out of model inputs, including cached requests.

Review receipts remain local audit evidence. Models receive measured evidence
and their own previous decisions through explicit projections, never an operator
strategy. This check is also used by the offline request auditor.
"""
from __future__ import annotations

import json
import re

POLICY = "measured_evidence_and_api_decisions.v1"
_OPERATOR_KEYS = {"next_strategy", "latest_codex_review", "implementation_change",
                  "codex_diagnosis", "codex_strategy", "operator_notes"}
_FIELD = re.compile(r'"(?:' + '|'.join(sorted(_OPERATOR_KEYS)) + r')"\s*:')
_REVIEWER = re.compile(r'"reviewer"\s*:\s*"codex"')


def operator_context_findings(content):
    """Inspect JSON blocks/embedded JSON without reading referenced local files."""
    findings = []

    def visit(value, path, depth=0):
        if depth > 32:
            raise ValueError("API context nesting exceeds its inspectable bound")
        if isinstance(value, dict):
            if value.get("encoding") == "object_prompt_payload.v1":
                from .prompt_payload import decode
                visit(decode(value), path + "/expanded", depth + 1)
                return
            if value.get("reviewer") == "codex":
                findings.append({"path": path, "kind": "operator_review"})
            for key, child in value.items():
                if key in _OPERATOR_KEYS:
                    findings.append({"path": path + "/" + key, "kind": "operator_field"})
                # Field names alone do not prove that their values are image
                # bytes. A nested history can also use `data`, and image
                # metadata can contain arbitrary extra fields. Inspect these
                # values as ordinary JSON/text; never decode image bytes or
                # open a URL here.
                visit(child, path + "/" + key, depth + 1)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, path + "/" + str(i), depth + 1)
        elif isinstance(value, str):
            # Generation contexts can be preceded by an ordinary text heading.
            # Also inspect JSON encoded inside another JSON string.
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                if _REVIEWER.search(value):
                    findings.append({"path": path, "kind": "embedded_operator_review"})
                for match in _FIELD.finditer(value):
                    findings.append({"path": path, "kind": "embedded_operator_field",
                                     "field": match.group(0)})
                # A heading/fence can wrap JSON whose fields themselves contain
                # serialized JSON. Regex alone misses those escaped field names.
                decoder = json.JSONDecoder()
                consumed = 0
                for match in re.finditer(r'[\[{]', value):
                    if match.start() < consumed:
                        continue
                    try:
                        block, consumed = decoder.raw_decode(value, match.start())
                    except (ValueError, RecursionError):
                        continue
                    visit(block, path + '/embedded', depth + 1)
                return
            if isinstance(decoded, (dict, list, str)):
                visit(decoded, path + "/decoded", depth + 1)

    visit(content, "content")
    return findings


def assert_autonomous_context(content):
    findings = operator_context_findings(content)
    if findings:
        raise ValueError("Operator review text cannot enter an API request: " +
                         json.dumps(findings[:8], ensure_ascii=False))
    return POLICY
