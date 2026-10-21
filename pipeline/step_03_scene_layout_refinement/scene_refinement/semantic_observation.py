"""Validate visible observations without manufacturing semantic correctness."""
from .protocol import validate_score

POLICY_VERSION='visible_major_relationships.v2'
MAJOR_KINDS={'functional_front_reversal','exaggerated_overall_size','left_right_reversal',
             'front_back_reversal','wrong_support','inside_outside_reversal'}


def validate_observation_review(value, catalog):
    ids={o['object_id'] for o in catalog}
    active={o['object_id'] for o in catalog if not o['is_environment_support']}
    validate_score(value,ids)
    if value.get('policy_version')!=POLICY_VERSION or type(value.get('comparison_possible')) is not bool:
        raise ValueError('Missing observation policy and comparison scope')
    if value['uncertain']:
        if (value['comparison_possible'] or not isinstance(value.get('unavailable_reason'),str)
                or not value['unavailable_reason'].strip()):
            raise ValueError('Uncertainty requires an unusable overall comparison')
        return value
    if not value['comparison_possible']:
        raise ValueError('A binary result requires a usable scene comparison')
    observations=value.get('object_observations')
    if (not isinstance(observations,list) or
            any(not isinstance(o,dict) or not isinstance(o.get('object_id'),str) for o in observations) or
            not active<={o['object_id'] for o in observations}<=ids):
        raise ValueError('Observe every named foreground object before judging')
    if len(observations)!=len({o['object_id'] for o in observations}):
        raise ValueError('Duplicate object observations')
    for item in observations:
        if any(not isinstance(item.get(k),str) or not item[k].strip()
               for k in ('original_visible_features','candidate_visible_features')):
            raise ValueError('Object observations need visible evidence from both scenes')
    relations=value.get('relationship_observations')
    if not isinstance(relations,list) or not relations:
        raise ValueError('Observe the principal relationships')
    contradictions=value.get('major_contradictions')
    if not isinstance(contradictions,list) or bool(contradictions)!=(value['score']==0):
        raise ValueError('Only a concrete major contradiction can produce score zero')
    for item in [*relations,*contradictions]:
        if (not isinstance(item,dict) or not isinstance(item.get('object_ids'),list)
                or not item['object_ids'] or not all(isinstance(i,str) for i in item['object_ids'])
                or set(item['object_ids'])-ids
                or any(not isinstance(item.get(k),str) or not item[k].strip()
                       for k in ('original_visible','candidate_visible'))):
            raise ValueError('Relationship claims require paired visible evidence and known IDs')
    for item in contradictions:
        if not isinstance(item.get('kind'),str) or item['kind'] not in MAJOR_KINDS:
            raise ValueError('Small positioning or appearance differences are not major contradiction categories')
        if item['original_visible'].strip()==item['candidate_visible'].strip():
            raise ValueError('A contradiction must identify a visible difference')
    return value
