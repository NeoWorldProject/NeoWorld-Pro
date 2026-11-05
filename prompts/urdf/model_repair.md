# You are given:
1. The reference images and target object description, when supplied.
2. The current blender Python script.
3. The execution and geometry review feedback.

# Your task:
Fix this blender Python script. 

Requirements:
- Fix the reported geometry, component ownership, material, preview, and execution defects.
- Preserve existing lighting, world, exposure, and color-management settings. Do not add lights.
- Keep the target's own parts; remove neighboring objects mistakenly included in the model.
- Do not change the output path.

# Output
Output only the executable Python code.
