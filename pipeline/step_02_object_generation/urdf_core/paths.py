"""Code and installed runtime-resource locations."""
from pathlib import Path
import os
import sys

URDF_CORE = Path(__file__).resolve().parent
OBJECT_GENERATION = URDF_CORE.parent
PIPELINE = OBJECT_GENERATION.parent
PROJECT = PIPELINE.parent
KINEMATICS_VALIDATION = OBJECT_GENERATION / 'kinematics_validation'
RESOURCE_ROOT = Path(os.environ.get('NEOWORLD_RESOURCE_ROOT', str(PROJECT))).resolve()
