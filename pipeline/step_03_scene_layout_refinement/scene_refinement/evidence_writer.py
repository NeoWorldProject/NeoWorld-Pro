"""Bounded background evidence encoding; close is a mandatory acceptance barrier."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import lzma
import json
from pathlib import Path
import threading


class EvidenceWriter:
    def __init__(self, trace_path, *, compressed=True, asynchronous=False, capacity=8):
        self.path = Path(trace_path)
        self.compressed, self.asynchronous = compressed, asynchronous
        self.capacity = threading.BoundedSemaphore(capacity)
        # One writer preserves trace/frame order and bounds uncompressed memory.
        self.executor = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.futures = []
        self.closed = False
        self.handle = None

    def __enter__(self):
        self.handle = (lzma.open(self.path, 'wt', encoding='utf-8', preset=1) if self.path.suffix == '.xz' else
                       gzip.open(self.path, 'wt', encoding='utf-8', compresslevel=1) if self.compressed
                       else self.path.open('w', encoding='utf-8'))
        return self

    def _submit(self, operation, *args):
        if self.closed:
            raise RuntimeError('Evidence writer is already closed')
        if self.executor is None:
            operation(*args)
            return
        self.capacity.acquire()
        def run():
            try:
                operation(*args)
            finally:
                self.capacity.release()
        self.futures.append(self.executor.submit(run))
        # Surface errors promptly, without retaining a future for every frame.
        while self.futures and self.futures[0].done():
            self.futures.pop(0).result()

    def write(self, serialized_sample):
        if not isinstance(serialized_sample, str):
            raise TypeError('Trace evidence must be immutable serialized text')
        self._submit(self.handle.write, serialized_sample)

    def frame(self, pixels, path):
        from PIL import Image
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Render buffers are reused by the SDK. Own the bytes before dispatch.
        owned = pixels.copy()
        owned.flags.writeable = False
        self._submit(lambda: Image.fromarray(owned).save(path))

    def flush(self):
        self._submit(self.handle.flush)

    def __exit__(self, *args):
        self.closed = True
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            for future in self.futures:
                future.result()
        finally:
            if self.handle:
                self.handle.close()
        # Caller hashes/encodes/accepts only after this barrier succeeds.
        return False
