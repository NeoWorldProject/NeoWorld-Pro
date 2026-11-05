"""Lossless JSON tables for diagnostic requests that exceed the readable budget."""
import copy
import json
from collections import Counter

SCHEMA = 'physics_diagnostic_transport.v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def encode(payload):
    def reserved(value):
        if isinstance(value, dict):
            if any(k in {'$records', '$ref'} or k.startswith('@') for k in value):
                raise ValueError('Diagnostic input contains reserved transport keys')
            for child in value.values():
                reserved(child)
        elif isinstance(value, list):
            for child in value:
                reserved(child)
    reserved(payload)

    def records(value):
        if not isinstance(value, (dict, list)):
            return value
        rows = list(value.values()) if isinstance(value, dict) else value
        if len(rows) >= 3 and all(isinstance(row, dict) for row in rows):
            common = set(rows[0]).intersection(*(set(row) for row in rows[1:]))
            defaults = {}
            for key in sorted(common):
                serialized, count = Counter(canonical(row[key]) for row in rows).most_common(1)[0]
                if count >= max(2, len(rows) // 2):
                    defaults[key] = json.loads(serialized)
            if defaults:
                changes = [{k: records(v) for k, v in row.items()
                            if k not in defaults or canonical(v) != canonical(defaults[k])} for row in rows]
                result = {'keys': list(value) if isinstance(value, dict) else None,
                          'defaults': records(defaults), 'rows': changes}
                if all(set(row) == set(changes[0]) for row in changes):
                    result['columns'] = list(changes[0])
                    result['rows'] = [[row[k] for k in result['columns']] for row in changes]
                return {'$records': result}
        return ({k: records(v) for k, v in value.items()} if isinstance(value, dict)
                else [records(v) for v in value])

    value = records(payload)
    counts = Counter()
    def count_values(node):
        serialized = canonical(node)
        if len(serialized) >= 16:
            counts[serialized] += 1
        if isinstance(node, (dict, list)):
            for child in node.values() if isinstance(node, dict) else node:
                count_values(child)
    count_values(value)
    shared, identities = [], {}
    def share(node, reference=True):
        serialized = canonical(node)
        repetitions = counts[serialized]
        if reference and repetitions > 1 and (repetitions-1)*len(serialized) > repetitions*12+15:
            if serialized not in identities:
                index = len(shared)
                identities[serialized] = index
                shared.append(None)
                shared[index] = share(node, False)
            return {'$ref': identities[serialized]}
        if isinstance(node, dict):
            return {k: share(v) for k, v in node.items()}
        return [share(v) for v in node] if isinstance(node, list) else node
    result = {'payload': share(value), 'shared_values': shared}
    key_counts = Counter()
    def count_keys(node):
        if isinstance(node, dict):
            for key, child in node.items():
                key_counts[key] += 1
                count_keys(child)
        elif isinstance(node, list):
            for child in node:
                count_keys(child)
    count_keys(result)
    keys = [k for k, n in key_counts.items() if len(k) >= 12 and n >= 3 and not k.startswith('$')]
    codes = {k: '@'+str(i) for i, k in enumerate(keys)}
    def rename(node):
        if isinstance(node, dict):
            return {codes.get(k, k): rename(v) for k, v in node.items()}
        return [rename(v) for v in node] if isinstance(node, list) else node
    return {**rename(result), 'key_dictionary': keys, 'api_evidence_schema': SCHEMA,
        'decoding': [
            'Decode before diagnosis. Object keys @N mean key_dictionary[N]; string values are unchanged.',
            'Replace each {"$ref":N} recursively with shared_values[N].',
            'For each $records object, decode defaults and each row. If columns exists, zip columns with '
            'each row array. Merge defaults with row values (row values override defaults). '
            'If keys is null the result is a list; otherwise zip keys with the reconstructed rows to form a dictionary.',
            'The reconstructed payload is the diagnostic request. Parameters, permissions, failed metrics '
            'and history are exact; compression grants no additional edit permission.']}


def decode(encoded):
    if encoded.get('api_evidence_schema') != SCHEMA:
        raise ValueError('Unknown diagnostic transport')
    keys = encoded['key_dictionary']
    def rename(node):
        if isinstance(node, dict):
            result = {}
            for key, value in node.items():
                if key.startswith('@'):
                    key = keys[int(key[1:])]
                if key in result:
                    raise ValueError('Duplicate decoded key')
                result[key] = rename(value)
            return result
        return [rename(v) for v in node] if isinstance(node, list) else node
    shared = rename(encoded['shared_values'])
    def expand(node, stack=()):
        if isinstance(node, dict):
            if set(node) == {'$ref'}:
                index = node['$ref']
                if type(index) is not int or not 0 <= index < len(shared) or index in stack:
                    raise ValueError('Invalid or cyclic diagnostic reference')
                return expand(shared[index], (*stack, index))
            return {k: expand(v, stack) for k, v in node.items()}
        return [expand(v, stack) for v in node] if isinstance(node, list) else node
    def restore(node):
        if isinstance(node, dict):
            if set(node) == {'$records'}:
                table = node['$records']
                defaults = restore(table['defaults'])
                rows = []
                for row in table['rows']:
                    if 'columns' in table:
                        if len(row) != len(table['columns']):
                            raise ValueError('Invalid diagnostic table width')
                        row = dict(zip(table['columns'], row))
                    rows.append({**copy.deepcopy(defaults), **restore(row)})
                if table['keys'] is None:
                    return rows
                if len(table['keys']) != len(rows) or len(set(table['keys'])) != len(rows):
                    raise ValueError('Invalid diagnostic table keys')
                return dict(zip(table['keys'], rows))
            return {k: restore(v) for k, v in node.items()}
        return [restore(v) for v in node] if isinstance(node, list) else node
    return restore(expand(rename(encoded['payload'])))
