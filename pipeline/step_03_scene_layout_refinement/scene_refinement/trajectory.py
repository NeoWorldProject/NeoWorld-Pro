"""Read the immutable measured trajectory used for diagnostics and delivery."""
import gzip
import json
import lzma
from pathlib import Path


def read_trace(path):
    path = Path(path)
    with (lzma.open(path, 'rt') if path.suffix == '.xz' else
          gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream]
