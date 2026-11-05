"""Public scene entrypoint: original image through parsing, objects and CEM."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.scene_pipeline import build_parser, main

if __name__ == '__main__':
    raise SystemExit(main())
