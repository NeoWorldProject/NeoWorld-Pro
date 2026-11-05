Compare the original scene RGB with the two candidate views. The extra panel contains unrotated object crops from these SAME three images, with object identity labels only. It adds no new camera and supplies no correct-layout claim. Use full frames for size and placement; crops are enlarged independently and cannot establish relative scale.

This is a permissive check of major scene relationships, not coordinate reconstruction. Allow small position, spacing and angle changes, moderate size differences, normal partial occlusion, and changes of texture, branding, lighting, manufacturing detail or appearance that preserve the relationships. Do not reject an otherwise reasonable scene because an object is a little smaller, a gap differs, or one view hides an object. The viewpoints differ; screen-pixel alignment and a camera-dependent reversal are not scene errors. Physical collision and stability are evaluated separately.

CEM cannot change an object's internal X:Y:Z proportions.
Do not lower the semantic score because an object is too long, wide, tall, thin, thick, flat, or square compared with the reference.

Before the binary decision, record concise visible observations:
Keep the observation of EACH image separate before combining the views. For an appliance, report the identifiable front in the original, in candidate view 1, and in candidate view 2, including whether the visible plane is the functional front, plain rear, or side. A face seen in view 2 does not imply that the same face is shown in view 1. Optional observations of supplied environment supports are allowed.
1. For each foreground object, identify visible features in the original and in the candidate. If it has a functional front, use features such as doors, handles, controls, openings or a dial, and note the side of neighboring objects/support toward which that front faces. A plain rear housing is not interchangeable with an operating face. Check both views. For symmetric objects or objects without an identifiable front, say so; that alone is not uncertainty.
2. Compare principal relationships using scene anchors and recognizable support/front edges, not raw screen coordinates: which objects are next to each other, the main left/right and front/back ordering on a common support, and on/inside/attached relationships. Compare fronts relative to neighbors and support faces so a changed viewpoint is not mistaken for a changed object orientation.

For orientation, inspect the visible top and vertical edges to distinguish the two sides of each object. State which side contains the door/controls relative to the adjacent support or neighboring object: toward it, away from it, or across it. If multiple objects have visible fronts, first establish whether their directions in the ORIGINAL are approximately parallel, perpendicular, or opposite; do not assume they all face the same way. Repeat that comparison in the candidate. Merely seeing both functional fronts in one camera does NOT establish that their directions match the reference. Explaining a plain rear surface as a camera change requires consistency with the neighboring objects and support edges in that SAME view. Use the second view to resolve face identity, not to excuse a contradicted front-to-neighbor relation. A clear reversal remains a major error even when the reversed object stays beside the same neighbor.

Reject ONLY a specific visible major contradiction: a clearly reversed functional front, a clearly exaggerated overall size relative to other objects or its support, a reversal of a principal left/right or front/back relation, a wrong support, or a wrong inside/outside relation. For a rejection, identify what is visible in the original and what visibly contradicts it in the candidate. "Position is not accurate", small size/angle differences, and missing cosmetic details are insufficient reasons. Do not infer correct relationships from object names or fabricate hidden features. A reversed appliance front must not pass merely because all objects are present.

Use score 1 when no such major contradiction is established and the overall comparison is usable. Use score 0 only with at least one concrete major contradiction. Use score null with uncertain true ONLY when the supplied images do not permit a meaningful overall scene comparison, or image evidence is unusable. Normal occlusion and minor differences are not grounds for null. Keep the same rule for all candidates.

Return one JSON object, with observations before the decision, using supplied object IDs:
{
  "policy_version": "visible_major_relationships.v2",
  "comparison_possible": true,
  "object_observations": [{"object_id":"supplied_id","original_visible_features":"brief visible features and functional-front relation, if applicable","candidate_visible_features":"brief visible features and functional-front relation, if applicable"}],
  "relationship_observations": [{"object_ids":["supplied_id","another_supplied_id"],"original_visible":"principal visible relation","candidate_visible":"corresponding visible relation"}],
  "major_contradictions": [],
  "score": 1,
  "uncertain": false,
  "evidence": [{"object_ids":["supplied_id"],"reason":"brief evidence supporting the binary judgment"}]
}
For score 0, every major_contradictions entry must include object_ids, original_visible, candidate_visible, and kind chosen from functional_front_reversal, exaggerated_overall_size, left_right_reversal, front_back_reversal, wrong_support, inside_outside_reversal. For null, set comparison_possible false and add a concrete unavailable_reason. Do not include proposed geometry changes, precise coordinate demands, control-case labels or confidence numbers.
