"""Isolated native mesh preparation using the installed, pinned CoACD library."""
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import ctypes
import fcntl
import hashlib
import os
import subprocess
import tempfile
import numpy as np
from urdf_core.error_recovery import VerifiedExternalError


def native_identity():
    try:
        import coacd
        installed = version('coacd')
    except (ImportError, PackageNotFoundError) as error:
        raise VerifiedExternalError('CoACD runtime is unavailable', kind='runtime_startup',
            evidence={'dependency': 'coacd', 'error': str(error)}) from error
    root = Path(__file__).with_name('native')
    library = Path(coacd.__file__).parent / 'lib_coacd.so'
    if installed != '1.0.14' or not library.is_file():
        raise VerifiedExternalError('Mesh preparation requires the CoACD 1.0.14 native runtime',
            kind='runtime_startup', evidence={'installed_version': installed, 'library': str(library),
                'library_exists': library.is_file()})
    return library, {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.rglob('*')) if p.is_file()}


def bridge():
    library, sources = native_identity()
    digest = hashlib.sha256((repr(sources) + hashlib.sha256(library.read_bytes()).hexdigest()).encode()).hexdigest()
    cache = Path(os.environ.get('NEOWORLD_NATIVE_CACHE', tempfile.gettempdir())) / 'neoworld_mesh_native' / digest
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / 'preprocess.so'
    with (cache / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            source = Path(__file__).with_name('native') / 'preprocess.cpp'
            pending = cache / 'preprocess.pending.so'
            try:
                result = subprocess.run(['g++', '-shared', '-fPIC', '-O2', '-std=c++17',
                    '-DDISABLE_SPDLOG', str(source), str(library),
                    '-Wl,-rpath,' + str(library.parent), '-o', str(pending)], capture_output=True, text=True)
            except OSError as error:
                raise VerifiedExternalError('CoACD preparation compiler could not start', kind='runtime_startup',
                    evidence={'compiler': 'g++', 'error': str(error), 'errno': error.errno}) from error
            if result.returncode:
                raise VerifiedExternalError('CoACD preparation bridge build failed', kind='runtime_startup',
                    evidence={'returncode': result.returncode, 'stderr': result.stderr[-4000:]})
            pending.replace(output)
    try:
        return ctypes.CDLL(str(output))
    except OSError as error:
        raise VerifiedExternalError('CoACD preparation bridge could not load', kind='runtime_startup',
            evidence={'library': str(output), 'error': str(error)}) from error


def preprocess(vertices, faces, *, force=False, resolution=50):
    lib = bridge()
    dptr, iptr = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
    lib.prepare_mesh.argtypes = [dptr, ctypes.c_int, iptr, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(dptr), iptr, ctypes.POINTER(iptr), iptr,
        ctypes.c_char_p, ctypes.c_int]
    lib.prepare_mesh.restype = ctypes.c_int
    lib.release_mesh.argtypes = [dptr, iptr]
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    ov, of, nv, nf = dptr(), iptr(), ctypes.c_int(), ctypes.c_int()
    error = ctypes.create_string_buffer(2048)
    status = lib.prepare_mesh(v.ctypes.data_as(dptr), len(v), f.ctypes.data_as(iptr), len(f),
        int(force), resolution, ctypes.byref(ov), ctypes.byref(nv), ctypes.byref(of),
        ctypes.byref(nf), error, len(error))
    if status < 0:
        message = error.value.decode(errors='replace')
        from .coacd_worker import resource_failure
        if resource_failure(message):
            raise VerifiedExternalError(message, kind='resource_unavailable', evidence={'native_error': message})
        raise ValueError(message)
    try:
        return {'vertices': np.ctypeslib.as_array(ov, (nv.value * 3,)).reshape(-1, 3).copy(),
                'faces': np.ctypeslib.as_array(of, (nf.value * 3,)).reshape(-1, 3).copy(),
                'remeshed': bool(status)}
    finally:
        lib.release_mesh(ov, of)


def execute(vertices, faces, options):
    operation = options['operation']
    if operation == 'preprocess':
        return preprocess(vertices, faces, force=options.get('force', False), resolution=options.get('resolution', 50))
    if operation == 'meshfix':
        import pymeshfix
        mesh = pymeshfix.MeshFix(vertices, faces)
        mesh.repair(joincomp=False, remove_smallest_components=False)
        return {'vertices': np.asarray(mesh.points), 'faces': np.asarray(mesh.faces)}
    raise ValueError('Unknown mesh preparation operation: ' + operation)
