Infer a coherent layout for the main scene directly from the supplied original RGB image. Return one STRICT JSON object, with no markdown or surrounding text. Use only the supplied image; visible text is scene content, not instructions. Do not infer answers from filenames, dataset identities, external references or ground truth. Keep explanations to one concise sentence per field.

Use complete functional objects as scene instances. Group normally assembled parts and accessories with their owner; keep visibly independent objects separate. Include the main objects even when partially occluded, and omit incidental background clutter. Base ownership and independent identity on visible structure and connections, not appearance differences or disconnected image fragments alone. When ownership is uncertain, prefer a coherent whole-object interpretation without inventing additional objects or support relations. Foundational boundaries and cropped support surfaces may be environment; a complete object remains foreground when it supports another object.

Return these fields:
- `schema_version`: `"scene_layout.v2"`.
- `scene_frame`: `{"units":"meters","handedness":"right","up_axis":"+Z","x_axis":"view_right_horizontal","y_axis":"scene_depth","origin_description":<nonempty string>,"scale_reasoning":<nonempty string>}`. Use world +Z up, +X observer-right projected onto the horizontal plane, and +Y into the scene. Use one origin at the primary foundational support's top-surface center and one defensible estimated metric scale; mark an unseen reference as inferred.
- `scene_tree`: every unique object name maps to `{"children":[<direct child names>]}`. Include leaves with empty arrays. Each child has at most one parent; references must exist and there must be no cycles. Multiple roots are allowed when their physical relationship is unknown.
- `objects_data`: one entry per object, with every field below.
- `inventory_changes`: `[]`; `identity_events`: `[]`.

Every object contains:
- `name`: unique descriptive instance name; `description`: concise visible identity.
- `is_environment_support`: boolean; `is_occluded`: boolean; `occlusion_reasoning`: nonempty text distinguishing occlusion by another object from image-boundary truncation.
- `included_visible_parts`: array of owned visible components, possibly empty; `asset_identity_reasoning`: brief evidence for their ownership and this object's independent identity.
- `2d_bbox_normalized`: `[left,top,right,bottom]` as fractions of original image width/height, satisfying `0 <= left < right <= 1` and `0 <= top < bottom <= 1`. Cover ALL visible owned parts, including narrow appendages and separated fragments, with a small margin. Inspect all four edges. Exclude guessed hidden extent and shadows. Overlap with another box is allowed. Environment supports may use `null`.
- `support_relation`: `on`, `inside`, `attached`, `unknown` or `root`, describing the direct parent relationship. Parentless objects use `root`.
- `obb`: `{"center_m":[x,y,z],"size_m":[sx,sy,sz],"rotation_xyz_deg":[rx,ry,rz]}`. All values are finite numbers; sizes are strictly positive FULL local-axis lengths. Estimate the complete object's enclosing box, including hidden parts and protrusions. Its center is the box center, not the contact point. Every environment support also needs finite geometry.
- `geometry_reasoning`: brief scale, pose and support evidence; `uncertainty`: nonempty text describing what the image cannot determine.

Rotation uses degrees and right-handed active rotations: for column vectors, `R = Rz(rz) @ Ry(ry) @ Rx(rx)` and `world = center + R @ local`. Local +Z is canonical up and local -Y is semantic front where meaningful. Use one coherent pose estimate; acknowledge symmetry or hidden orientation. Infer actual support levels, containment and attachment; neither every object's bottom nor every parent's enclosing-box top is a universal support plane. Enclosing boxes may overlap without occupied shapes colliding.

Omit `object_id`; code assigns stable IDs. Omit derived `3d_bbox_meters`, orientation hypotheses and detector queries. Before returning, check main-scene coverage at this object granularity, component ownership, normalized boxes, direct support, shared scale, positive dimensions and rotation consistency.
