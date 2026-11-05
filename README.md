# NeoWorld-Pro: Programming Interactive Scenes from Monocular Images for Embodied Simulation

**NeurIPS 2026**

[ArXiv Paper](https://arxiv.org/abs/2608.24212) | [Project Page](https://neoworldproject.github.io/neoworld-pro-website/)


## 1. Installation

Tested on Linux with Python 3.13, Blender 4.1.1, and Isaac Sim 5.1.0. Install Blender and Isaac Sim separately; an Isaac-compatible NVIDIA GPU and FFmpeg/FFprobe on `PATH` are required.

Install dependencies from the repository root (a C++17 compiler is required):

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

For scenes, configure parsing and object models in [`config/pipeline/algorithm.json`](config/pipeline/algorithm.json), and the scene semantic model in [`config/pipeline/runtime.json`](config/pipeline/runtime.json). For single objects, use [`config/object_generation/algorithm.json`](config/object_generation/algorithm.json).

### 2.2 Scene Reconstruction

Run scene parsing, object generation and physics refinement, scene composition, and CEM refinement:

```bash
python pipeline/run_scene.py run \
  --image /path/to/scene.png \
  --scene-id my_scene \
  --output output/my_scene \
  --config config/pipeline/algorithm.json \
  --runtime config/pipeline/runtime.json
```

Resume an interrupted run with the same source version:

```bash
python pipeline/run_scene.py resume --output output/my_scene
```

Results are saved under the specified output directory:

```text
01_parsing/           Scene layouts, object crops, and parsing records.
02_objects/           Generated objects and their refinement records.
03_scene/delivery/    Exported scene, dependencies, and delivery receipt.
pipeline_report.json  Overall pipeline status and stage results.
```

The generated scene and its associated records are saved in `03_scene/delivery/`.
Generated object assets are saved in `02_objects/objects/<object_id_hash>/final/`,
with the object inventory in `02_objects/asset_manifest.json`.

### 2.3 Single-object Reconstruction

Generate an object from an image, including articulation generation and Isaac physics refinement:

```bash
python pipeline/run_object.py \
  --image /path/to/object.png \
  --output output/my_object \
  --config config/object_generation/algorithm.json \
  --runtime config/object_generation/runtime.json
```

Published assets and their delivery manifest are saved in `output/my_object/final/`.

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
