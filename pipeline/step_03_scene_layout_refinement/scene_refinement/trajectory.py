"""Read the immutable measured trajectory used for diagnostics and delivery."""
import gzip
import json
import lzma
from pathlib import Path


def read_trace(path):
    path = Path(path)
    if str(path).endswith(".compact.json"):
        raise ValueError("Full scene trace was processed; use its exact terminal state or replay samples")
    with (lzma.open(path, 'rt') if path.suffix == '.xz' else
          gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return [json.loads(line) for line in stream]


def terminal_sample(path):
    if str(path).endswith(".compact.json"):
        from physics_feedback.compact_trajectory import CompactTrajectory
        return CompactTrajectory(path).terminal_sample
    return read_trace(path)[-1]
