"""Exact native-source equivalence across immutable snapshot directories."""
from pathlib import Path

from urdf_core.run_state import file_hash


def same_contract_values(previous, regenerated):
    """Only revision/hash-chain stamps may differ; all actual values stay exact."""
    if not isinstance(previous, dict) or not isinstance(regenerated, dict):
        return False
    ignored = {'revision', 'previous_contract_sha256'}
    return ({k: v for k, v in previous.items() if k not in ignored}
            == {k: v for k, v in regenerated.items() if k not in ignored})


def runtime_sources_match(recorded, expected):
    """Verify original bytes AND exact role/hash equality; never trust names.

    Moving identical code into another frozen supervisor snapshot does not
    change its native implementation. Missing roles, changed bytes, duplicates
    or unbound paths are incompatible and require a fresh measurement.
    """
    def roles(values):
        if not isinstance(values, dict) or not values:
            return None
        result = {}
        for name, digest in values.items():
            if not isinstance(name, str) or not isinstance(digest, str):
                return None
            path = Path(name).resolve()
            parts = path.parts
            try:
                index = next(i for i in range(len(parts)-1)
                    if parts[i:i+2] == ('pipeline', 'step_02_object_generation'))
            except StopIteration:
                return None
            role = '/'.join(parts[index+2:])
            if not role or role in result or not path.is_file() or file_hash(path) != digest:
                return None
            result[role] = digest
        return result
    old, current = roles(recorded), roles(expected)
    return old is not None and current is not None and old == current
