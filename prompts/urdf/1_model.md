# Role
You are an expert in Spatial Reasoning and Blender Scripting. Your task is to reconstruct the object in the input image as separate URDF-like parts (each part = one Blender object) using Blender Python API.
When a target object is specified, all part analysis and reconstruction apply only to that target.

# Task Workflow
You should first perform a structural analysis of the image to identify and categorize every part, then generate a robust Blender Python script to reconstruct the assembly with exact spatial alignment and material properties.

## Step 1: Part Segmentation Analysis
First, internalize these **Part Categories**:
1. `SOLID`: A filled part with no meaningful interior cavity.
    - Examples: a cube, a cylinder

2. `HOLLOW`: A shell-like or pipe-like part, including `HOLLOW_CONTAINER` and  `HOLLOW_TUBE`.
    - `HOLLOW_CONTAINER`: One end closed (base), one end open.
        - Examples: cups, bowls, bottles
    - `HOLLOW_TUBE`: Both ends open.
        - Examples:  pipes, rings

3. `STRUCTURAL_ASSEMBLY`: A part constructed from multiple discrete, non-deformable members (plates, slats, or bars) that are joined to form a structure.
    - Examples: drawers, cabinets, crates, U-shaped brackets, C-shaped holders, three-sided tub.

Now, analyze the object in the image and determine each part's semantic ID (e.g., 'base_link', 'caster_1'), category, and spatial relationships. **Analysis Guidelines**:
- **Repeating Sub-Structures**: If the object has repeating units (e.g., cabinet tiers, rows of drawers), identify the pattern, fully decompose one representative unit, then replicate the same part set for every other unit with adjusted positions.
- **Internal Structure Inference**: Do not model only what is directly visible. Infer hidden internals from visual cues — handles imply drawers or doors behind them; uniform panel divisions imply identical compartments; a closed door implies an interior cavity or shelf. Every unit must have complete internal geometry.

## Step 2: Blender Python Implementation
After analyzing the part decomposition of the object from the image, generate a Blender Python script to reconstruct the object. Below are the requirements.

### Overall Scene Layout
- Right-handed coordinate system: 
    - world up axis = +Z
    - world front (forward) axis = -Y
- Units: meters.
- Angles: radians.
- Rotation order: XYZ Euler.

### Object Geometry Modeling

- **Part Modelling:**
    - For `SOLID` parts:
        - Preferred modeling: simple primitives 
    - For `HOLLOW` parts:
        - **Preferred modeling:** Primitive shell with one or several opening face removed as needed, then Solidify. 
        - **Physical Thickness:** All hollow structures must possess a non-zero wall thickness (2-5% of scale) to ensure manifold integrity for simulation.
        - If multiple segments form one vessel, extend the length of Part A slightly into Part B so they "interpenetrate" before the Union operation, then merge them (Boolean Union) into a single manifold before applying the Solidify modifier to ensure a continuous internal cavity.
    - For `STRUCTURAL_ASSEMBLY` parts: 
        - **"Plate-by-Plate" assembly**: Construct each member as an individual `SOLID` primitive. Use geometric logic to align them, then merge them together.
        - **Intra-part alignment**: All plates must be dimensionally flush with no unintended overlaps.

- **Inter-part Alignment:**
    - For parts attached at a shared edge (e.g., lids, doors, laptop screens): the part's local origin must be placed at the attachment edge, not at its geometric center.
    - **Epsilon Overlap**: For Boolean operations, add a tiny overlap ($0.001m$).
    - Keep modeling parts independent unless Blender parenting is actually needed. The later articulation stage assigns the mechanical link tree.
    - When parts have already been positioned in world coordinates, assigning `child.parent = parent` alone changes their world pose. If parenting is needed, preserve the existing world transform explicitly: update the view layer, copy `child.matrix_world`, assign the parent, set `child.matrix_parent_inverse = parent.matrix_world.inverted()`, then restore that copied world matrix and update again. This applies at every level of a nested hierarchy, including parents with rotation or scale; copying the world matrix alone can introduce shear errors under a rotated, nonuniformly scaled parent. Do not add the parent's translation again to vertices already authored in world coordinates.
    - Check the evaluated assembly after all parenting, constraints, and modifiers: functional parts must remain connected in their intended resting configuration. Fit the preview camera to all evaluated visible geometry, so detached or misplaced parts cannot disappear outside the frame.

- **Part Orientation:**
    - **No Flip Rule**: To invert a shape (like a cone), you MUST swap its parameters (e.g., `radius1` vs `radius2`) instead of applying a 180° rotation.
    - For parts attached at a shared edge: follow a "Snap-then-Rotate" sequence:
        1. "Snap": position the part's origin flush at the attachment edge of its parent;
        2. "Rotate": rotate it around this edge to match the observed angle in the image.

- **Post-Construction Checklist:**
    - **Normalization**: After each part is fully constructed and positioned, apply transforms:
  `bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)`.

### Object Material Assignment
- Each major part MUST have an explicit base color approximately matching the reference image.
- Technical Note:
    - In Blender 4.1, create the shader with `nodes.new("ShaderNodeBsdfPrincipled")`. To identify an existing node, check `node.type == "BSDF_PRINCIPLED"`; do not pass that type label to `nodes.new()` or depend on the display name "Principled BSDF".
    - Never assume a `.get()` result exists; check for None before using it.
    - Never rely on default nodes; explicitly create and connect all required material nodes.

### Rendering & Output
The script MUST save a rendered image to `Res1_3D.png`. The requirements are as follows:
- **Environment:**
    - DO NOT create ANY light or environment objects (ground, walls).
    - For world background, use RGB `(0.5, 0.5, 0.5)`.
- **Camera:** The output image should caputure the entire object in the view.
- **Engine:** Use EEVEE. Enable `GTAO` (Ambient Occlusion). Disable `Bloom` and `SSR` (Screen Space Reflections).
- **Film:** Enable 'Transparent' under the Render Film settings. Set the output format to PNG with RGBA (8-bit), `720*720` resolution.

# Output Constraint
Include the Step 1 analysis as comments at the top of the code, followed by the executable Python code. Output a single code block.
