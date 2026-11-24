#!/usr/bin/env python3
"""Generate F=64, H=128 inputs for existing large-inputs CSR matrices.
Python 3.6 compatible. Never overwrite existing datasets by default.
"""
import argparse
import hashlib
import json
from pathlib import Path
from generate_large_inputs import create_dense_x, create_dense_w


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(output, n, features, hidden):
    metadata_path = output / 'metadata.json'
    manifest = output / 'SHA256SUMS.txt'
    if not metadata_path.is_file() or not manifest.is_file():
        raise SystemExit('FAIL: reverse input metadata or SHA256 manifest missing')
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get('n'), metadata.get('features'), metadata.get('hidden')) != (n, features, hidden):
        raise SystemExit('FAIL: reverse matrix dimensions do not match metadata')
    expected = set(['X.dense', 'W.dense'])
    found = set()
    with manifest.open() as f:
        for line in f:
            hash_value, filename = line.strip().split('  ', 1)
            if filename not in expected or filename in found:
                raise SystemExit('FAIL: unexpected or duplicated manifest entry')
            found.add(filename)
            file_path = output / filename
            if not file_path.is_file() or checksum(file_path) != hash_value:
                raise SystemExit('FAIL: hash mismatch for {}'.format(file_path))
            print('OK: {}'.format(file_path))
    if found != expected:
        raise SystemExit('FAIL: manifest is incomplete')
    if (output / 'X.dense').open().readline().split() != [str(n), str(features)]:
        raise SystemExit('FAIL: X header mismatch')
    if (output / 'W.dense').open().readline().split() != [str(features), str(hidden)]:
        raise SystemExit('FAIL: W header mismatch')
    print('PASS: reverse inputs verified, N={} F={} H={}'.format(n, features, hidden))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=Path('reverse-inputs'))
    p.add_argument('--n', type=int, default=40000)
    p.add_argument('--features', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.n <= 0 or args.features <= 0 or args.hidden <= args.features:
        p.error('Need positive N, F and H > F to force (AX)W')
    output = args.output_dir
    if args.verify:
        verify(output, args.n, args.features, args.hidden)
        return
    if output.exists() and any(output.iterdir()):
        print('Existing reverse-input directory detected; verifying rather than overwriting.')
        verify(output, args.n, args.features, args.hidden)
        return
    output.mkdir(parents=True, exist_ok=True)
    create_dense_x(output / 'X.dense', args.n, args.features)
    create_dense_w(output / 'W.dense', args.features, args.hidden)
    (output / 'metadata.json').write_text(json.dumps({
        'n': args.n, 'features': args.features, 'hidden': args.hidden,
        'computation_order': '(AX)W',
        'dense_generation': 'generate_large_inputs.py create_dense_x/create_dense_w',
        'sparse_matrices': 'reused unchanged from large-inputs',
        'note': 'Different F/H dimensions from earlier A(XW) benchmark'
    }, sort_keys=True, indent=2) + '\n')
    with (output / 'SHA256SUMS.txt').open('w') as f:
        for name in ['X.dense', 'W.dense']:
            f.write('{}  {}\n'.format(checksum(output / name), name))
    verify(output, args.n, args.features, args.hidden)


if __name__ == '__main__':
    main()
