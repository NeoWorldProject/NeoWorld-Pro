"""Serialize trusted in-process measurements without occupying Isaac's thread."""
import gzip
import io
import json
import pickle
import struct
import sys


def main():
    with open(sys.argv[1], 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, compresslevel=1, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
                while True:
                    header = sys.stdin.buffer.read(8)
                    if not header:
                        break
                    if len(header) != 8:
                        raise ValueError('Truncated internal trajectory record')
                    size = struct.unpack('!Q', header)[0]
                    data = sys.stdin.buffer.read(size)
                    if len(data) != size:
                        raise ValueError('Truncated internal trajectory payload')
                    sample = pickle.loads(data)
                    stream.write(json.dumps(sample, allow_nan=False, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'error_type': type(error).__name__,
            'errno': error.errno if isinstance(error, OSError) else None,
            'message': str(error)[:2000]}), file=sys.stderr)
        raise SystemExit(1) from None
