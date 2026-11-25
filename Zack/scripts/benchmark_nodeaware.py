#!/usr/bin/env python3
"""A/B test: selected original/replicated hybrid versus node-aware hybrid.
Python 3.6 compatible. Run from Final_Project within an Expanse allocation.
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

FIELDS = ('job_id', 'nodes', 'ranks', 'ranks_per_node', 'dataset', 'order',
          'variant', 'repetition', 'distribution_ms', 'mpi_elapsed_ms',
          'setup_inclusive_ms', 'local_compute_ms', 'external_wall_ms')
PATTERNS = {
    'distribution_ms': r'^CSR distribution\s*=\s*([0-9.eE+-]+) seconds',
    'mpi_elapsed_ms': r'^MPI elapsed time\s*=\s*([0-9.eE+-]+) seconds',
    'setup_inclusive_ms': r'^MPI setup-inclusive time\s*=\s*([0-9.eE+-]+) seconds',
    'local_compute_ms': r'^Local computation\s*=\s*([0-9.eE+-]+) seconds',
}
DATASETS = ('large_regular', 'clustered_strong')
ORDERS = ('A(XW)', '(AX)W')
VARIANTS = ('reference', 'nodeaware')

def run(cmd):
    start = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    wall = (time.perf_counter() - start) * 1000
    if result.returncode:
        raise RuntimeError('Command failed: {}\n{}\n{}'.format(
            ' '.join(map(str, cmd)), result.stdout, result.stderr))
    return result.stdout, wall

def input_paths(dataset, order):
    csr = ('large-inputs/regular-A.csr' if dataset == 'large_regular'
           else 'large-inputs/clustered-strong-A.csr')
    prefix = 'large-inputs' if order == 'A(XW)' else 'reverse-inputs'
    return csr, prefix + '/X.dense', prefix + '/W.dense'

def command(ranks, nodes, executable, csr, x, w, output):
    return ['mpirun', '-np', str(ranks), '--bind-to', 'core',
            '--map-by', 'ppr:{}:node'.format(ranks // nodes),
            str(executable), csr, x, w, str(output)]

def parse(output):
    result = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, output, re.M)
        if m is None:
            raise RuntimeError('Missing {} in output:\n{}'.format(key, output[-2000:]))
        result[key] = float(m.group(1)) * 1000
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--ranks', type=int, nargs='+', default=[16])
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) != args.nodes:
        parser.error('SLURM allocation node count does not match --nodes')
    if any(r % args.nodes or r // args.nodes > 16 or r < args.nodes for r in args.ranks):
        parser.error('Ranks must be divisible by nodes and at most 16 per node')
    if args.repetitions <= 0 or args.warmups < 0:
        parser.error('Invalid repetition counts')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logdir = args.output.parent / ('nodeaware-validation-' + os.environ.get('SLURM_JOB_ID', 'local'))
    logdir.mkdir(parents=True, exist_ok=True)
    binaries = {
        'reference': args.build_dir / ('mpi-hybrid-original-timed' if args.nodes == 1 else 'mpi-hybrid-replicated-timed'),
        'nodeaware': args.build_dir / 'mpi-hybrid-nodeaware-timed',
    }
    measurements = []
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for dataset in DATASETS:
            for order in ORDERS:
                csr, x, w = input_paths(dataset, order)
                # Validate each executable against independent sequential reference.
                reference = logdir / ('ref-{}-{}.dense'.format(dataset, order.replace('(', '').replace(')', '')))
                run(['./sequential-v21', csr, x, w, str(reference)])
                for ranks in args.ranks:
                    for variant in VARIANTS:
                        produced = logdir / ('{}-{}-{}-{}.dense'.format(dataset,
                                               order.replace('(', '').replace(')', ''), ranks, variant))
                        stdout, _ = run(command(ranks, args.nodes, binaries[variant], csr, x, w, produced))
                        if not re.search(r'^Order=' + re.escape(order) + r'\s', stdout, re.M):
                            raise RuntimeError('Wrong multiplication order for ' + variant)
                        if variant == 'nodeaware':
                            expected = 'scatter' if args.nodes == 1 else 'replicated'
                            if not re.search(r'^CSR strategy=' + expected + r' Physical_nodes=' + str(args.nodes) + r'$', stdout, re.M):
                                raise RuntimeError('Node-aware version selected the wrong CSR strategy:\n' + stdout[-2000:])
                        run(['python3', 'validate_large_outputs.py', str(reference), str(produced)])
                        produced.unlink()
                reference.unlink()
                for ranks in args.ranks:
                    for repetition in range(-args.warmups, args.repetitions):
                        # Alternate execution order across repetitions to reduce bias.
                        order_variants = VARIANTS if repetition % 2 == 0 else VARIANTS[::-1]
                        for variant in order_variants:
                            scratch = logdir / ('scratch-{}.dense'.format(variant))
                            stdout, wall = run(command(ranks, args.nodes, binaries[variant], csr, x, w, scratch))
                            scratch.unlink()
                            if repetition < 0:
                                continue
                            timing = parse(stdout)
                            row = dict(job_id=os.environ.get('SLURM_JOB_ID', ''), nodes=args.nodes,
                                       ranks=ranks, ranks_per_node=ranks // args.nodes,
                                       dataset=dataset, order=order, variant=variant,
                                       repetition=repetition + 1, external_wall_ms=wall)
                            row.update(timing)
                            writer.writerow(row)
                            f.flush()
                            measurements.append(row)
                            print('{} {} nodes={} ranks={} {} rep={} distribution={:.3f}ms inclusive={:.3f}ms'.format(
                                dataset, order, args.nodes, ranks, variant, repetition + 1,
                                timing['distribution_ms'], timing['setup_inclusive_ms']), flush=True)
    for dataset in DATASETS:
        for order in ORDERS:
            for ranks in args.ranks:
                groups = {v: [r for r in measurements if r['dataset'] == dataset and r['order'] == order and r['ranks'] == ranks and r['variant'] == v]
                          for v in VARIANTS}
                base = statistics.median(r['setup_inclusive_ms'] for r in groups['reference'])
                candidate = statistics.median(r['setup_inclusive_ms'] for r in groups['nodeaware'])
                print('RESULT {} {} ranks={} reference={:.3f}ms nodeaware={:.3f}ms change={:+.1f}%'.format(
                    dataset, order, ranks, base, candidate, 100 * (candidate/base - 1)), flush=True)
    print('PASS: {} measurements, validated reference and node-aware outputs'.format(len(measurements)), flush=True)

if __name__ == '__main__':
    main()
