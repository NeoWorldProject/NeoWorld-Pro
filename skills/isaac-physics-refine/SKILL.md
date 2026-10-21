---
name: isaac-physics-refine
description: Diagnose measured object-physics failures and propose bounded parameter experiments under the frozen object_pipeline.v1 contract.
---

# Isaac physics refinement

Read the frozen contract, measured metrics and trajectory windows, current and
best candidate parameters, prior experiments and remaining repair count. Explain
observations, a testable physical hypothesis, uncertainty, and which next
measurement would contradict that hypothesis. Failure IDs identify symptoms;
they do not restrict the possible causes. Do not infer a mechanism from one peak
or frame. Compare native position, velocity, target, active effort and contacts
around the first failure and largest excursion.

## Tests and frozen decisions

The simulator and independent numerical verifier decide acceptance. Visual geometry,
axes, topology, joint ranges, initial coordinates, effort caps, test thresholds,
scene support classification and upright classification are frozen before testing.

Current-scene objects independently supported by a horizontal floor, table,
counter or shelf require passive ground stability. Objects requiring mounting,
clamping, insertion, leaning or other external support skip ground/free tests as
not applicable. They retain applicable fixed-base joints and delivery reload.
All fixed-base joints use an elevated fixture clearing the full authored range.
Free-base testing is optional and disabled by default. Its enabled state is frozen
by the CLI. Enabled free tests use the measured stable rest pose and the
code-derived contiguous ground-operable interval. Ground-blocked motion is not
applicable, not a pass. Final scene support and whole-scene stability are outside
this object-level test.

The v11 numerical rules apply to both fixed and free joint motion: measured
coverage at least 80%; each endpoint and final return error at most 80% of the
tested interval; limited-joint overflow at most max(1e-5, 20% of full range).
Non-target rotary initial-position tolerance is 10 degrees; prismatic tolerance
is max(1e-5 m, 20% of range). Parent-child structural angular tolerance is 10
degrees; prismatic translation tolerance is max(1e-5 m, 20% of range). Other
structural translations retain the contract position tolerance. No endpoint dwell
or post-motion zero-effort observation is required. Bounded active control
continues to the final scheduled motion sample. Ground and reload rotary
initial-position tolerance is 10 degrees; their final-window stability and root
tilt thresholds remain as declared in the contract. Native finite states and all
scheduled samples are required. Velocity peaks and penetration depth are
diagnostic, not separate rejection thresholds.

## Parameter experiments

Use only the supplied allowed paths and absolute values. Apply changes to the
best candidate; the current candidate can be a rejected experiment.

- Ground: existing part mass/density, joint static/dynamic friction and damping,
  object contact friction.
- Fixed joint: controller kp, kd and speed_fraction.
- Free joint: speed_fraction and object contact friction.
- Return to ground: revisit mass, passive resistance or contact friction from a
  measured motion failure, then retest the complete applicable sequence.
- Return to fixed: revisit controller gains/speed after a measured free-base
  failure, then retest fixed and free motion.

Mass/density must be finite and positive, with no multiplier limit relative to
their original values. Dynamic friction
cannot exceed static friction. Contact friction remains within [0,2]; restitution
is zero. Do not increase an effort ceiling. Damping
resists velocity and cannot hold a static gravitational load; excessive passive
resistance can prevent bounded-force movement. Frame/contact changes distinguish
sliding, rolling and rocking. Do not patch a velocity peak when all acceptance
checks pass. If evidence supports no permitted experiment, report uncertain or
unrepairable with empty patches.

Collision geometry and its generation settings are frozen during refinement.

Repeated mechanisms use a frozen random representative per structural group by
default; full-joint mode is explicit. A patch to a shared group field expands to
all members. Follow the supplied groups; do not change their membership or
representative. Mass, density, COM and inertia are not shared. All DOFs remain
observed and controlled, including group members not individually operated.

## Iterations and delivery

There is one physics pool of at most 8 repairs across all routes. Initial
parameters and the initial measurement do not consume a repair. Each new repair
is reserved before requesting its proposal; malformed responses, failed builds
and failed tests do not refund the iteration. A transport retry or process restart
continues that same reservation. Remaining ground/motion values are views of the
same pool, never additive. There is no elapsed-time budget.

Every three repairs the harness records an internal strategy review. No external
review checkpoint is required. Only budget exhaustion ends an unaccepted search and starts complete final assessment of the best technically
valid candidate. An uncertain or unrepairable proposal consumes its reserved repair, makes no parameter change,
and becomes feedback for the next attempt while budget remains. Never leave the physics parameter boundary.
API transport attempts are limited to five per execution session; recovery preserves the cumulative ledger without consuming extra refine budget. Native tasks have at most five technical attempts, except
CoACD system errors allow ten retries after the first attempt, independent of repair counts;
exhausted technical retries or an invalid environment raise an error and terminate the program. Ground failure cannot block elevated fixed joints, and one joint
failure cannot block another independent joint. Failed ground can block its
required free-base tests, explicitly. Delivery reload is independently attempted.

The harness preserves each candidate and protects previously passed checks when
promoting a replacement. Successful physics requires one independent full-trace
verification; later readers reuse its content-bound numerical receipt. Successful
physics does not need a video. Failed trials retain diagnostic evidence.
Delivery status, quality verdict and assessment completeness are separate. A
critic failure does not prevent final URDF/USD delivery or turn into a pass.

## Response

Return one JSON object without Markdown:

```json
{
  "verdict": "refine",
  "failure_id": "tracking_lag",
  "trial_ids": ["fixed_joint_hinge"],
  "evidence": ["trial_id=fixed_joint_hinge: measured window and values"],
  "reason": "Observed failure and uncertainty",
  "hypothesis": "Testable mechanism",
  "expected_effect": "Metric direction and a result that would contradict it",
  "route": "fixed_joint",
  "patches": [{"path": "controller.hinge.kd", "value": 1.0}]
}
```

Cite exact observed trial IDs from the active phase. The allowed verdicts are
refine, uncertain and unrepairable. Numeric values above illustrate the response
shape only; choose parameters from the supplied measured evidence.
