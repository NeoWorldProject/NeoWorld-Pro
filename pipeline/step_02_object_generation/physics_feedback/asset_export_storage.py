"""Publish native object exports through the immutable run-local asset pool."""
from pathlib import Path
import uuid

from urdf_core.storage import copy_asset


def export_asset(layer, destination):
    destination = Path(destination)
    temporary = destination.with_name('.object_export_' + uuid.uuid4().hex + '.usd')
    try:
        if not layer.Export(str(temporary)):
            raise ValueError('Native object USD export failed')
        copy_asset(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
