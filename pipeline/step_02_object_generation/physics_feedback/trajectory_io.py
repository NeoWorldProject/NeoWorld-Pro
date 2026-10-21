"""Lossless trajectory storage; compression never changes sample serialization.

New native runs use gzip level 1. Historical plain JSONL remains readable.
Hashes in reports bind the stored file (including the gzip footer), not an
unverified decoded surrogate. Readers must consume the iterator completely to
validate the complete stream; acceptance and diagnostic windows do so.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import io
import json
from pathlib import Path
import zlib


@contextmanager
def open_trajectory_writer(path):
    """Write UTF-8 text unchanged; callers retain their exact JSON formatting."""
    path = Path(path)
    if path.suffix != ".gz":
        with path.open("w", encoding="utf-8", newline="") as stream:
            yield stream
        return
    with path.open("wb") as raw:
        # Stable header, low-cost compression, no filename or current time.
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                           compresslevel=1, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                yield stream


@contextmanager
def open_trajectory_reader(path):
    """Read plain or gzip JSONL; corrupt/truncated gzip is an explicit error."""
    path = Path(path)
    try:
        opener = gzip.open if path.suffix == ".gz" else Path.open
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            yield stream
    except (gzip.BadGzipFile, EOFError, zlib.error, UnicodeError) as error:
        raise ValueError(f"Invalid or truncated physics trajectory {path}: {error}") from error


def iter_trajectory_samples(path):
    """Strict streaming JSONL decoding, including the gzip CRC/length footer."""
    def reject_constant(value):
        raise ValueError(f"Nonfinite constant in physics trajectory: {value}")

    with open_trajectory_reader(path) as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                sample = json.loads(line, parse_constant=reject_constant)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid physics trajectory JSONL at line {line_number}: {error}") from error
            if not isinstance(sample, dict):
                raise ValueError(f"Physics trajectory line {line_number} is not a sample object")
            yield sample


def first_trajectory_sample(path):
    """Get initialization while validating the entire compressed stream."""
    first = None
    for sample in iter_trajectory_samples(path):
        if first is None:
            first = sample
    if first is None:
        raise ValueError("Physics trajectory contains no samples")
    return first
