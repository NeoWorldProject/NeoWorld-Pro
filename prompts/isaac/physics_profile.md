Infer a SIMPLE simulator physical profile from the source images, measured metre-scale
parts and validated kinematics. This is a functional approximation, not measured GT.
No new geometry, joint axes/ranges, actuation motors, springs or fidelity tests.

Return JSON with exactly profile, effort_budgets and orientation_policy.
orientation_policy is the separately supplied frozen upright-use classification.
profile = {
 "schema_version": 3,
 "stable_required": true,
 "stability_reason": "actual support and placement in the current scene, with visible evidence",
 "links": {"exact_link": {
   "material": {"static_friction":0.5,"dynamic_friction":0.5,"restitution":0},
   "objects": {"EVERY exact authored part": {
      "role":"contact", "reason":"function/material reasoning", "mass_kg":0.1
   }}
 }},
 "joints": {"EVERY active joint": {
   "static_friction_effort":0.1, "dynamic_friction_effort":0.05, "damping":0.01
 }},
 "collision":{"approximation":"coacd","enabled":true,"self_collision":false}
}
Numbers above are FORMAT EXAMPLES, not defaults for this object. Infer part-specific
positive mass_kg OR density_kg_m3, never both. Part masses are summed and COM/inertia
are computed from original geometry. Pure visual markings may be decoration: keep
its positive mass but omit independent collision. Do not relabel functional surfaces
as decoration to evade a failure. Open contact surfaces require thickness_m and an
approximation_reason; mass_model may be surface_shell. Open decoration with explicit
mass uses a triangle-sheet mass approximation and needs no collision thickness.

Use the original scene RGB, target crop, parent object and support relations together.
stable_required=true when THIS placement is independently supported by a horizontal
floor, tabletop, counter or shelf. stable_required=false when THIS placement is not
on a horizontal plane or requires external support/installation: wall mounting,
hanging, inclined stands, hole/slot insertion, clamping or leaning. A phone lying
flat is true; that same phone on an inclined stand is false. A rack on a table is
true; tubes inserted in it are false. Do not substitute category-level potential
uses for the observed scene placement. When a frozen scene classification is supplied,
copy its stable_required exactly. Never change this choice because a test failed.
With no scene context, use visible evidence only and do not invent support relations.
False skips standalone ground/free tests as NOT APPLICABLE, never as stability passed.
Applicable fixed-base joint tests and delivery reload checks are still required.
Final scene support/stability remains untested by this classification.

Ground testing starts exactly on the floor with zero ACTIVE effort. The root may
settle naturally; articulated joints must retain frozen q_initial within tolerance,
tilt remains bounded, and all links/joints must be stable over the final tail.
No free-fall/wind, dropped-ball probe or internal collision criterion.
Native static friction holds joints at rest. Damping only resists motion and cannot
hold a gravity-loaded joint at zero speed. Use the smallest credible static friction
that permits holding; excessive resistance makes later bounded-force motion fail.
Static/dynamic efforts are N for prismatic, N*m for rotational joints. Damping is
N*s/m or N*m*s/rad. Set 0 <= dynamic <= static <= that joint's max_effort.
Contact friction lies in [0,2], dynamic <= static; restitution is zero.

For every active joint, effort_budgets contains {"max_effort":positive_number,
"reason":"why this direct force/torque is sufficient and reasonable for the measured size/use"}.
These ceilings are frozen for all refinements. Include sufficient margin for gravity
and passive friction without assuming unlimited actuation. No active joints means {}.
Do not output link-level mass, custom COM/inertia, controller parameters, collision
recipes or material-specific fine detail. Keep the existing part and joint names.

All fixed-base joint tests are elevated and clear the floor across the full
authored range. Infer sufficient finite effort for the suspended moving load.
Free-base testing defaults to disabled and is selected only by the command line.
Ground/reload rotary initial-position tolerance is 10 degrees; their final-window
stability and root tilt requirements retain the frozen contract values. Motion
trials use the supplied v11 rules without endpoint dwell or post-motion passive
holding. Native velocity peaks are diagnostic, while finite complete native
measurements remain mandatory. Parameter replies cannot change test rules.
