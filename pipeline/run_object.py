"""Public object entrypoint; all modes use the same object state machine."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'step_02_object_generation'))

from urdf_core.cli import main

if __name__ == '__main__':
    import os
    import uuid
    from generate_object_urdf import build_parser, parse_configured_args
    args = parse_configured_args(build_parser(), None)
    if args.out_dir and (args.out_dir/'run_manifest.json').is_file() and not args.fresh:
        os.environ['NEOWORLD_EXPLICIT_RESUME_ID'] = uuid.uuid4().hex
    else:
        os.environ.pop('NEOWORLD_EXPLICIT_RESUME_ID', None)
    raise SystemExit(main())
