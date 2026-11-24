#!/usr/bin/env python3
"""Generate deterministic power-law-style CSR matrices for MPI partition tests.

Produces two matrices with N=4000 and nnz=64000 by default.
These are newly generated inputs, not Rolando's frozen A060/A075 files.
"""
import argparse
import hashlib
import math
import random
import statistics
from pathlib import Path


def degrees_power_law(n, nnz, exponent):
    if not (0 <= nnz <= n * n):
        raise ValueError('Require 0 <= nnz <= N*N (unique columns per row)')
    weights = [(rank + 1) ** (-exponent) for rank in range(n)]
    remaining = nnz
    degrees = [0] * n
    eligible = list(range(n))
    # Capped largest-remainder apportionment; typically one pass at chosen settings.
    while remaining:
        weight_sum = sum(weights[i] for i in eligible)
        if not weight_sum:
            raise ValueError('Cannot allocate remaining nonzeros')
        ideals = {i: remaining * weights[i] / weight_sum for i in eligible}
        saturated = [i for i in eligible if ideals[i] >= n - degrees[i]]
        if saturated:
            for i in saturated:
                room = n - degrees[i]
                degrees[i] += room
                remaining -= room
            eligible = [i for i in eligible if degrees[i] < n]
            continue
        floors = {i: math.floor(ideals[i]) for i in eligible}
        for i, v in floors.items():
            degrees[i] += v
            remaining -= v
        leftovers = sorted(eligible, key=lambda i: (-(ideals[i] - floors[i]), i))
        for i in leftovers[:remaining]:
            degrees[i] += 1
        remaining = 0
    assert sum(degrees) == nnz
    assert all(0 <= degree <= n for degree in degrees)
    return degrees


def write_csr(path, n, nnz, exponent, seed, layout):
    rng = random.Random(seed)
    degrees = degrees_power_law(n, nnz, exponent)
    if layout == 'shuffled':
        rng.shuffle(degrees)
    rowptr = [0]
    indices = []
    values = []
    for degree in degrees:
        columns = sorted(rng.sample(range(n), degree))
        indices.extend(columns)
        values.extend(0.10 + 0.01 * ((j % 9) + 1) for j in range(degree))
        rowptr.append(len(indices))
    assert rowptr[-1] == nnz
    with path.open('w') as out:
        out.write(f'{n} {n} {nnz}\n')
        out.write(' '.join(map(str, rowptr)) + '\n')
        out.write(' '.join(map(str, indices)) + '\n')
        out.write(' '.join(f'{v:.9g}' for v in values) + '\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'{path}: nnz={nnz} median={statistics.median(degrees)} '
          f'min={min(degrees)} max={max(degrees)} '
          f'stdev={statistics.pstdev(degrees):.3f} SHA256={digest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=4000)
    parser.add_argument('--nnz', type=int, default=64000)
    parser.add_argument('--seed', type=int, default=5522)
    parser.add_argument('--layout', choices=('shuffled', 'clustered'), default='shuffled')
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    if args.n <= 0 or args.nnz < 0:
        parser.error('--n must be positive and --nnz nonnegative')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, exponent in [('moderate', 0.60), ('strong', 0.75)]:
        write_csr(args.output_dir / f'irregular-{name}-A.csr',
                  args.n, args.nnz, exponent, args.seed, args.layout)


if __name__ == '__main__':
    main()
