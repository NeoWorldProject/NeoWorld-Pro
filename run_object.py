"""Public object entrypoint; all modes use the same object state machine."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'pipeline/step_02_object_generation'))

from urdf_core.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
