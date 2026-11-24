#!/usr/bin/env python3
"""Deterministic N=40000 COP5522 benchmark data; Python 3.6 compatible.

Five CSR datasets (regular, two shuffled, two clustered), one shared X, one W.
The irregular degree and column generation is delegated to the existing
`generate_irregular_inputs.py`, so its algorithm remains unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

from generate_irregular_inputs import write_csr


def create_regular(path, n, nnz_per_row):
    nnz = n * nnz_per_row
    with path.open('w') as out:
        out.write('{} {} {}\n'.format(n, n, nnz))
        out.write(' '.join(str(i * nnz_per_row) for i in range(n + 1)) + '\n')
        for row in range(n):
            out.write(' '.join(str((row + k * 37) % n) for k in range(nnz_per_row)))
            out.write('\n' if row == n - 1 else ' ')
        values = ['{:.9g}'.format(.10 + .01 * ((k % 9) + 1)) for k in range(nnz_per_row)]
        for row in range(n):
            out.write(' '.join(values))
            out.write('\n' if row == n - 1 else ' ')


def create_dense_x(path, n, features):
    with path.open('w') as out:
        out.write('{} {}\n'.format(n, features))
        for i in range(n):
            out.write(' '.join('{:.9g}'.format(((i + 3 * j) % 17 + 1) / 17.0)
                               for j in range(features)) + '\n')


def create_dense_w(path, features, hidden):
    with path.open('w') as out:
        out.write('{} {}\n'.format(features, hidden))
        for i in range(features):
            out.write(' '.join('{:.9g}'.format(((2 * i + 5 * j) % 19 + 1) / 19.0)
                               for j in range(hidden)) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, default=Path('large-inputs'))
    p.add_argument('--n', type=int, default=40000)
    p.add_argument('--nnz-per-row', type=int, default=16)
    p.add_argument('--features', type=int, default=128)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--seed', type=int, default=5522)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if args.n <= 0 or not (0 < args.nnz_per_row <= args.n) or args.features <= 0 or args.hidden <= 0:
        p.error('Invalid matrix dimensions')
    outdir = args.output_dir
    manifest = outdir / 'SHA256SUMS.txt'
    names = ['regular-A.csr', 'shuffled-moderate-A.csr', 'shuffled-strong-A.csr',
             'clustered-moderate-A.csr', 'clustered-strong-A.csr', 'X.dense', 'W.dense']
    if args.verify:
        if not manifest.exists():
            p.error('No SHA256 manifest in {}'.format(outdir))
        with manifest.open() as stream:
            for line in stream:
                expected, name = line.rstrip('\n').split('  ', 1)
                path = outdir / name
                if not path.is_file() or sha256(path) != expected:
                    raise SystemExit('FAILED hash check: {}'.format(path))
                print('OK: {}'.format(path))
        metadata = json.loads((outdir / 'metadata.json').read_text())
        for key in ('n', 'nnz_per_row', 'features', 'hidden', 'seed'):
            if metadata[key] != getattr(args, key):
                raise SystemExit('Metadata mismatch for {}: {} != {}'.format(
                    key, metadata[key], getattr(args, key)))
        print('PASS: all input hashes and generation parameters verified')
        return
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.force and (manifest.exists() or any((outdir / x).exists() for x in names)):
        p.error('Data already exists; run --verify or explicitly --force to regenerate')
    print('Generating 5 CSR inputs and X/W: N={} nnz={} F={} H={}'.format(
        args.n, args.n * args.nnz_per_row, args.features, args.hidden), flush=True)
    create_regular(outdir / names[0], args.n, args.nnz_per_row)
    for layout in ('shuffled', 'clustered'):
        for level, exponent in (('moderate', .60), ('strong', .75)):
            write_csr(outdir / '{}-{}-A.csr'.format(layout, level),
                      args.n, args.n * args.nnz_per_row, exponent, args.seed, layout)
    create_dense_x(outdir / 'X.dense', args.n, args.features)
    create_dense_w(outdir / 'W.dense', args.features, args.hidden)
    metadata = dict(n=args.n, nnz_per_row=args.nnz_per_row, nnz=args.n * args.nnz_per_row,
                    features=args.features, hidden=args.hidden, seed=args.seed,
                    original_irregular_generator='generate_irregular_inputs.py',
                    original_exponents={'moderate': .60, 'strong': .75},
                    regular_pattern='(row + 37*k) modulo N, 16 k values per row',
                    note='Independent synthetic large inputs, not Rolando frozen primary')
    (outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    with manifest.open('w') as f:
        for name in names:
            digest = sha256(outdir / name)
            f.write('{}  {}\n'.format(digest, name))
            print('{}  {}'.format(digest, name), flush=True)
    print('DONE: {}'.format(outdir), flush=True)


if __name__ == '__main__':
    main()
