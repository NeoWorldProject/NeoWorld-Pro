"""Bounded, lossless trajectory writer; completion joins and checks the child."""
from pathlib import Path
import json
import pickle
import queue
import struct
import subprocess
import sys
import threading


class TrajectoryWriter:
    def __init__(self, path, *, queue_size=4):
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.closed = False
        self.proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).with_name('trajectory_writer_worker.py')), str(path)],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                data = self.queue.get()
                if data is None:
                    break
                self.proc.stdin.write(struct.pack('!Q', len(data)))
                self.proc.stdin.write(data)
            self.proc.stdin.close()
        except BaseException as error:
            self.error = error
            try:
                self.proc.stdin.close()
            except OSError:
                pass

    def _put(self, value):
        while True:
            if self.error is not None or self.proc.poll() is not None:
                raise self._failure() from self.error
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                continue

    def _failure(self):
        from urdf_core.error_recovery import VerifiedExternalError, external_failure
        code = self.proc.poll()
        if code is None and self.error is not None:
            try:
                code = self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
        if code is None:
            return RuntimeError('Trajectory writer failed')
        text = self.proc.stderr.read().decode('utf-8', errors='replace')
        if code < 0:
            return VerifiedExternalError('Trajectory writer terminated by signal', kind='worker_exit',
                evidence={'returncode': code, 'stderr': text[-2000:]})
        try:
            detail = json.loads(text)
        except (ValueError, TypeError):
            detail = {}
        if isinstance(detail, dict) and type(detail.get('errno')) is int:
            error = OSError(detail['errno'], detail.get('message', 'Trajectory write failed'))
            if external_failure(error):
                return error
        return RuntimeError('Trajectory writer failed: ' + text[-2000:])

    def write_sample(self, sample):
        if self.closed:
            raise ValueError('Trajectory writer is closed')
        # Pickling before enqueueing makes later producer mutations irrelevant.
        self._put(pickle.dumps(sample, protocol=5))

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.closed = True
        if kind is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait()
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            self.thread.join(timeout=5)
            self.proc.stderr.close()
            return False
        try:
            self._put(None)
            self.thread.join()
            code = self.proc.wait()
            if self.error is not None or code:
                raise self._failure() from self.error
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait()
            self.proc.stderr.close()
        return False
