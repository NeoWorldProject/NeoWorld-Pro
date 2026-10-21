"""Functional upright classification supplied with initial physical inference."""
PROMPT = (
    "Classify whether this reconstructed object's intended resting/use pose has a required upright direction. "
    "Use the source images and validated geometry/kinematics only. Return JSON with exactly "
    "{\"upright_required\": boolean, \"reason\": nonempty string}. "
    "A required upright direction means rolling or tipping away from the intended support orientation "
    "would invalidate normal use. A body with no meaningful upright direction may roll naturally into a "
    "stable resting orientation. Do not confuse this with a need for wall/support attachment; this decision "
    "never skips a test. Do not select false merely to help a test pass. When the functional orientation is "
    "unclear, retain upright_required=true and explain the uncertainty. The code still requires real support, "
    "a stable final time window, unchanged initial joint positions, and finite complete observations. "
    "You may not change geometry, joints, masses, friction, force caps or acceptance thresholds."
)
FUNCTIONAL_SCHEMA = "object_orientation_decision.v2"
FUNCTIONAL_GRAVITY_PROMPT = PROMPT + (
    " Interpret upright as a requirement imposed by gravity and the object's function. "
    "A top indicated by graphics, controls, markings, or a photographed display pose alone is not "
    "a gravity-dependent upright requirement. If normal resting/use permits lying flat or multiple "
    "orientations, classify false. Classify true when tipping would invalidate a gravity-dependent "
    "function, containment, or load-bearing orientation. Infer this from the input rather than an "
    "object-name rule. Joint initial-state requirements remain unchanged."
)

