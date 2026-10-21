# Articulation review — kinematic_evidence_v3

Compare the original image, generated articulation frames, joint coordinates and
kinematics JSON. Judge the intended mechanics of the actual moving parts. The
proposed protocol and a successful numerical execution are evidence, not semantic
ground truth. Use only supplied `blender::N` frame references.

Each joint is previewed independently while the other movable branches remain at
zero. Frames include rest and both limits; repeated displayed frames may hold a
rendered sample. Occlusion or overlap with another movable branch alone does not
justify rejecting the joint or shortening its useful travel.

Classify each finding:

- `asset_error`: directly evidenced wrong pivot/axis, wrong sliding direction,
  broken connection, wrong parent or rigid ownership, or a rigid follower that
  fails to move with its assembly. Grossly misplaced wheel axles and motion
  inconsistent with visible guides belong here. Known construction pre-tilt
  must be represented in the static joint origin; the authored initial appearance
  remains q=0. A numerically consistent wrong attachment remains an asset error.
- `observation`: an expected active degree of freedom is absent. Record the
  missing function and your actual semantic judgment; this alone does not reject
  the current asset. Do not confuse a missing optional degree of freedom with
  an existing joint moving incorrectly or a missing rigid follower.
- `needs_evidence`: valid supplied images do not support a defensible mechanical
  decision because of occlusion, symmetry, subpixel motion or ambiguous identity.
  Uncertainty alone is not an asset defect and does not request a parameter edit.
- `out_of_scope`: minor clipping/jitter, appearance, mesh topology, operation
  sequencing, overlap with a separate movable branch, or separate surrounding
  objects visible in the crop. These observations do not reject the asset.

Identical pixels alone prove neither missing motion nor correct motion. Do not
invent missing mechanisms from occlusion or treat every surrounding prop as part
of this asset. Preserve existing authored joints and their useful travel; never
suggest deleting a failed joint, changing it to fixed, or shrinking its range.
Technical completeness is checked separately: missing/corrupt files or native
samples are not waived by an observation classification.

Return one strict JSON object:

```json
{"strengths": [], "minor_imperfections": [], "findings": [
  {"issue_type": "asset_error | observation | needs_evidence | out_of_scope",
   "reason": "Concrete finding grounded in supplied evidence", "evidence": ["blender::0"]}
], "critical_errors": [], "score": 9, "verdict": "pass | fail | uncertain"}
```

Use empty lists when appropriate. `critical_errors` contains only `asset_error`
reasons. Verdict is fail if an asset error exists. For an observation of a missing
expected degree of freedom, retain your actual semantic verdict, including fail
or uncertain. Otherwise verdict is uncertain if any needs_evidence finding
exists, else pass. Score is descriptive and cannot independently trigger a repair.
The pipeline records observations separately and never converts uncertainty to
semantic pass. Perform one semantic review; format correction is not reassessment.
