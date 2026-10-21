# NeoWorld-Pro: Programming Interactive Scenes from Monocular Images for Embodied Simulation

**NeurIPS 2026**

[ArXiv Paper](https://arxiv.org/abs/2608.24212) | [Project Page](https://neoworldproject.github.io/neoworld-pro-website/)


## 1. Installation

Tested on Linux with Python 3.13, Blender 4.1.1, and Isaac Sim 5.1.0. Install Blender and Isaac Sim separately; an Isaac-compatible NVIDIA GPU and FFmpeg/FFprobe on `PATH` are required.

Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
/path/to/isaac-sim/python.sh -m pip install "trimesh>=4.0" "scipy>=1.10" "Pillow>=9.1"
```

## 2. Experiments

### 2.1 Configuration

Set the API key and executable paths:

```bash
export OPENAI_API_KEY='your-api-key'
export NEOWORLD_BLENDER=/path/to/blender
export NEOWORLD_ISAAC_PYTHON=/path/to/isaac-sim/python.sh
```

For scene reconstruction, edit [`config/pipeline/runtime.json`](config/pipeline/runtime.json) and set `blender` and `isaac_python` in both the `objects` and `scene` sections.

Configure model versions in [`config/pipeline/algorithm.json`](config/pipeline/algorithm.json).

Isaac workers share a host-wide resource queue. Admission checks CPU load and pressure, available RAM, swap activity, GPU memory, and disk space; it does not impose a fixed concurrency cap. New workers start one at a time, with 8 tasking/TBB threads and a 10-second interval after the first physics context is initialized. Queue diagnostics are written to `resource_wait.json` in each worker directory; host-wide samples and peak worker counts are recorded in `/tmp/neoworld_isaac_admission/`.

### 2.2 Scene Reconstruction

Run scene parsing, object generation and physics refinement, scene composition, and CEM refinement:

```bash
python run_scene.py run \
  --image /path/to/scene.png \
  --scene-id my_scene \
  --output output/my_scene \
  --config config/pipeline/algorithm.json \
  --runtime config/pipeline/runtime.json
```

Resume an interrupted run using its saved configuration:

```bash
python run_scene.py resume --output output/my_scene
```

<details>
<summary>Command-line arguments</summary>

```text
run                 Start the full pipeline in a new, empty output directory.
resume              Resume a run with the same implementation and saved configuration.
--image PATH        Original scene image (required for run).
--scene-id ID       Scene identifier using letters, digits, underscores, or hyphens (required for run).
--output PATH       Pipeline output directory (required for run and resume).
--config PATH       Algorithm JSON with parsing, objects, and scene sections (required for run).
--runtime PATH      Runtime JSON with objects and scene sections (required for run).
-h, --help          Show help for run_scene.py or the selected subcommand.
```

</details>

Results are saved under the specified output directory:

```text
01_parsing/           Scene layouts, object crops, and parsing records.
02_objects/           Generated objects and their refinement records.
03_scene/delivery/    Exported scene, dependencies, and delivery receipt.
pipeline_report.json  Overall pipeline status and stage results.
```

Delivery and acceptance are recorded separately. Full scene acceptance requires physics checks, two independent positive semantic judgments, an eight-second reload validation, and complete evidence. Budget-exhausted runs may deliver best-effort assets with their validation status recorded in the receipt.

### 2.3 Single-object Reconstruction

Generate an object from an image, including articulation generation and Isaac physics refinement:

```bash
python run_object.py \
  --image /path/to/object.png \
  --output output/my_object
```

Repeat the same command with the same inputs, configuration, and output directory to resume an interrupted run. Published assets and their delivery manifest are saved in `output/my_object/final/`.

<details>
<summary>Command-line arguments</summary>

```text
--image PATH                                  Input image of the target object; mutually exclusive with --image-dir.
--image-dir, --image_dir PATH                  Directory of reference images of the same object (PNG, JPEG, or WebP).
--output, --out-dir, --out_dir PATH            Output directory; required except for --help and --check-environment.
--fresh                                       Start a new run; requires a nonexistent or empty output directory.
--scene-context, --scene_context PATH          Scene-context JSON binding the target identity, crop, and scene evidence.
--physics-from-run, --physics_from_run PATH    Reuse a verified current-version object run for a new physics refinement run.
--physics-api-cache-from-run PATH             Reuse top-level physics API replies from a prior run only for identical requests.
--articulation-from-run PATH                  Reuse a generation run's geometry and articulation; rerun articulation critics and subsequent physics in a new output directory.
--physics-source-manifest, --physics_source_manifest PATH  Import a hash-bound asset manifest instead of --physics-from-run.
--prompts-dir, --prompts_dir PATH              Object prompt directory (default: prompts/urdf).
--extra-model-instruction, --extra_model_instruction TEXT  Additional instructions for object modeling.
--stop-after, --stop_after STAGE               Stop after geometry, kinematics, or all stages (default: all).
--check-environment                           Report executable discovery and API credential availability, then exit.
--runtime-config PATH                         Object runtime JSON for executable paths, timeouts, API concurrency, and reasoning effort.
--blender PATH                                Blender executable; overrides the runtime configuration and environment.
--isaac-python, --isaac_python PATH            Isaac Sim Python launcher; overrides the runtime configuration and environment.
--ffmpeg PATH                                 FFmpeg executable; overrides the runtime configuration and environment.
--startup-timeout SECONDS                     Technical startup timeout (default: 300); not a total run time limit.
--no-progress-timeout SECONDS                 Technical inactivity timeout (default: 600); not a total run time limit.
--api-concurrency N                           Shared API concurrency limit (default: 64).
--model NAME                                  Object generation and refinement model.
--reasoning-effort LEVEL                       Reasoning effort: low, medium, high, or xhigh (default: high).
--num-model-candidates, --num_model_candidates N  Number of initial modeling candidates (fixed at 3).
--max-generation-repairs N                    Modeling repair budget, 0-3 (default: 3).
--max-articulation-repairs N                  Articulation repair budget, 0-3 (default: 3).
--max-physics-repairs N                       Physics repair budget, 0-8 (default: 8).
--joint-test-scope SCOPE                       Joint test scope: representative or all (default: representative; retained on resume).
--physics-free-base, --no-physics-free-base     Enable or disable free-base joint-motion trials (default: disabled).
--physics-trial-workers N                     Trial workers per object, 1-4 (default: 2); host CPU/memory and GPU availability control native process admission.
-h, --help                                    Show the command-line reference and exit.
```

</details>

## Acknowledgements

We thank the authors of [CoACD](https://github.com/SarahWeiii/CoACD) for their open-source contribution.

## Citation

```bibtex
@inproceedings{he2026neoworld,
  title={{NeoWorld-Pro}: Programming Interactive Scenes from Monocular Images for Embodied Simulation},
  author={He, Yumeng and Song, Yichen and Yang, Xiaotian and Zhang, Weijia and Zhou, Zanwei and Gong, Junru and Yang, Xiaokang and Wang, Yunbo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
