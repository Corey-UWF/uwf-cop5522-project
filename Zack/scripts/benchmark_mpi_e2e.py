#!/usr/bin/env python3
"""MPI whole-process warm-up comparison with safe CSV resumption and ORTE retries.

Python 3.6-compatible. Intended for launch inside an Expanse Slurm allocation.
Resuming preserves prior job IDs and measurements, and revalidates all variants.
"""
import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIELDS = ('job_id', 'nodes', 'ranks', 'ranks_per_node', 'dataset', 'order',
          'variant', 'repetition', 'distribution_ms', 'mpi_elapsed_ms',
          'setup_inclusive_ms', 'local_compute_ms', 'rank0_process_lifetime_ms',
          'external_wall_ms')
PATTERNS = {
    'distribution_ms': r'^CSR distribution\s*=\s*([0-9.eE+-]+) seconds',
    'mpi_elapsed_ms': r'^MPI elapsed time\s*=\s*([0-9.eE+-]+) seconds',
    'setup_inclusive_ms': r'^MPI setup-inclusive time\s*=\s*([0-9.eE+-]+) seconds',
    'local_compute_ms': r'^Local computation\s*=\s*([0-9.eE+-]+) seconds',
    'rank0_process_lifetime_ms': r'^Rank0 process lifetime\s*=\s*([0-9.eE+-]+) seconds',
}
DATASETS = ('large_regular', 'clustered_strong')
ORDERS = ('A(XW)', '(AX)W')
VARIANTS = ('baseline', 'warmed', 'nodeaware')
# Retrying only launcher failures avoids concealing numerical or application bugs.
TRANSIENT_MPI_MARKERS = (
    'an ORTE daemon has unexpectedly failed',
    'orted: error',
    'unable to launch the specified application',
    'failed to start the requested process',
    'lost communication with a remote daemon',
)


def run(cmd, retries=0, retry_delay=5):
    """Return (stdout, wall_ms). Retry only recognized transient mpirun failures."""
    for attempt in range(retries + 1):
        start = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
        wall = (time.perf_counter() - start) * 1000
        if result.returncode == 0:
            return result.stdout, wall
        details = (result.stdout + '\n' + result.stderr)
        transient = (Path(str(cmd[0])).name == 'mpirun' and
                     any(marker in details.lower() for marker in TRANSIENT_MPI_MARKERS))
        if transient and attempt < retries:
            print('WARNING: transient MPI launch failure, retry {}/{} after {} s: {}'.format(
                attempt + 1, retries, retry_delay, ' '.join(map(str, cmd))),
                file=sys.stderr, flush=True)
            time.sleep(retry_delay)
            continue
        raise RuntimeError('Command failed (exit {}): {}\n{}\n{}'.format(
            result.returncode, ' '.join(map(str, cmd)), result.stdout, result.stderr))
    raise AssertionError('Unreachable')


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
        match = re.search(pattern, output, re.M)
        if match is None:
            raise RuntimeError('Missing {} in output:\n{}'.format(key, output[-2000:]))
        result[key] = float(match.group(1)) * 1000
    return result


def key_for(row):
    return (int(row['nodes']), int(row['ranks']), row['dataset'], row['order'],
            row['variant'], int(row['repetition']))


def load_previous(path, nodes, ranks, repetitions):
    """Require complete rows, matching schema, and unique valid measurement keys."""
    previous = {}
    if path is None:
        return previous
    if not path.is_file():
        raise RuntimeError('Resume CSV does not exist: {}'.format(path))
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise RuntimeError('Resume CSV header is incompatible with this benchmark')
        for lineno, row in enumerate(reader, 2):
            if any(v is None or v == '' for v in row.values()):
                raise RuntimeError('Incomplete row {} in {} (not safe to resume)'.format(lineno, path))
            try:
                key = key_for(row)
                if (key[0] != nodes or key[1] not in ranks or key[2] not in DATASETS or
                        key[3] not in ORDERS or key[4] not in VARIANTS or
                        not 1 <= key[5] <= repetitions):
                    raise ValueError('unexpected configuration')
                if int(row['ranks_per_node']) != key[1] // nodes:
                    raise ValueError('ranks_per_node mismatch')
                for metric in ('distribution_ms', 'mpi_elapsed_ms',
                               'setup_inclusive_ms', 'local_compute_ms',
                               'rank0_process_lifetime_ms', 'external_wall_ms'):
                    value = float(row[metric])
                    if not 0 <= value < float('inf'):
                        raise ValueError('invalid metric ' + metric)
            except ValueError as exc:
                raise RuntimeError('Invalid resume CSV row {}: {}'.format(lineno, exc))
            if key in previous:
                raise RuntimeError('Duplicate measurement key in resume CSV: {}'.format(key))
            previous[key] = row
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nodes', type=int, required=True, choices=(1, 2, 4))
    parser.add_argument('--ranks', type=int, nargs='+', default=[16])
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path,
                        help='Existing partial CSV; copied to --output, leaving original unchanged')
    parser.add_argument('--mpi-retries', type=int, default=3,
                        help='Retries per transient ORTE/mpirun launch failure (default 3)')
    parser.add_argument('--retry-delay', type=int, default=5,
                        help='Seconds between launch retries (default 5)')
    args = parser.parse_args()
    if int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) != args.nodes:
        parser.error('SLURM allocation node count does not match --nodes')
    if (any(r % args.nodes or r // args.nodes > 16 or r < args.nodes for r in args.ranks)
            or len(set(args.ranks)) != len(args.ranks)):
        parser.error('Ranks must be unique, divisible by nodes, and at most 16 per node')
    if args.repetitions <= 0 or args.warmups < 0 or args.mpi_retries < 0 or args.retry_delay < 0:
        parser.error('Invalid repetition/retry settings')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A new job should use a new output path and resume-from the old partial CSV.
    if args.resume_from is None and args.output.exists():
        parser.error('Output exists. Supply --resume-from FILE to resume without overwriting it.')
    if (args.resume_from is not None and args.resume_from.resolve() != args.output.resolve()
            and args.output.exists()):
        parser.error('New output already exists. Choose a fresh --output path or resume in place.')
    previous = load_previous(args.resume_from, args.nodes, set(args.ranks), args.repetitions)
    if args.resume_from is not None and args.resume_from.resolve() != args.output.resolve():
        # Copy original bytes so existing measurements retain the original job_id.
        shutil.copyfile(str(args.resume_from), str(args.output))
    new_file = not args.output.exists()
    logdir = args.output.parent / ('nodeaware-validation-' + os.environ.get('SLURM_JOB_ID', 'local'))
    logdir.mkdir(parents=True, exist_ok=True)
    binaries = {
        'baseline': args.build_dir / 'mpi-hybrid-replicated-e2e',
        'warmed': args.build_dir / 'mpi-hybrid-replicated-warmup-e2e',
        'nodeaware': args.build_dir / 'mpi-hybrid-nodeaware-e2e',
    }
    for variant, binary in binaries.items():
        if not binary.is_file():
            raise RuntimeError('Missing {} executable: {}'.format(variant, binary))

    print('RESUME: {} measurements loaded from {}; {} expected overall'.format(
        len(previous), args.resume_from or '(none)',
        len(DATASETS) * len(ORDERS) * len(args.ranks) * len(VARIANTS) * args.repetitions),
        flush=True)
    measurements = dict(previous)
    with args.output.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            f.flush()
        for dataset in DATASETS:
            for order in ORDERS:
                csr, x, w = input_paths(dataset, order)
                # Only validate/process configurations with missing measurements.
                pending = any((args.nodes, ranks, dataset, order, variant, rep) not in measurements
                              for ranks in args.ranks for variant in VARIANTS
                              for rep in range(1, args.repetitions + 1))
                if not pending:
                    print('SKIP: {} {} already complete'.format(dataset, order), flush=True)
                    continue
                reference = logdir / ('ref-{}-{}.dense'.format(
                    dataset, order.replace('(', '').replace(')', '')))
                run(['./sequential-v21', csr, x, w, str(reference)])
                try:
                    # Freshly validate all three binaries against the sequential result.
                    for ranks in args.ranks:
                        for variant in VARIANTS:
                            produced = logdir / ('{}-{}-{}-{}.dense'.format(
                                dataset, order.replace('(', '').replace(')', ''), ranks, variant))
                            cmd = command(ranks, args.nodes, binaries[variant], csr, x, w, produced)
                            stdout, _ = run(cmd, args.mpi_retries, args.retry_delay)
                            if not re.search(r'^Order=' + re.escape(order) + r'\s', stdout, re.M):
                                raise RuntimeError('Wrong multiplication order for ' + variant)
                            if variant == 'nodeaware':
                                expected = 'scatter' if args.nodes == 1 else 'replicated'
                                if not re.search(r'^CSR strategy=' + expected +
                                                 r' Physical_nodes=' + str(args.nodes) + r'$', stdout, re.M):
                                    raise RuntimeError('Node-aware selected wrong strategy:\n' + stdout[-2000:])
                            run(['python3', 'validate_large_outputs.py', str(reference), str(produced)])
                            produced.unlink()
                finally:
                    if reference.exists():
                        reference.unlink()
                for ranks in args.ranks:
                    for repetition in range(-args.warmups, args.repetitions):
                        order_variants = VARIANTS if repetition % 2 == 0 else VARIANTS[::-1]
                        for variant in order_variants:
                            if repetition >= 0:
                                key = (args.nodes, ranks, dataset, order, variant, repetition + 1)
                                if key in measurements:
                                    continue
                            else:
                                # Warm up only if this variant has remaining measurements.
                                if not any((args.nodes, ranks, dataset, order, variant, rep) not in measurements
                                           for rep in range(1, args.repetitions + 1)):
                                    continue
                            scratch = logdir / ('scratch-{}.dense'.format(variant))
                            stdout, wall = run(command(ranks, args.nodes, binaries[variant],
                                                       csr, x, w, scratch),
                                               args.mpi_retries, args.retry_delay)
                            if scratch.exists():
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
                            os.fsync(f.fileno())
                            measurements[key] = row
                            print('{} {} nodes={} ranks={} {} rep={} rank0_lifetime={:.3f}ms launch_wall={:.3f}ms'.format(
                                dataset, order, args.nodes, ranks, variant, repetition + 1,
                                timing['rank0_process_lifetime_ms'], wall), flush=True)

    expected_count = len(DATASETS) * len(ORDERS) * len(args.ranks) * len(VARIANTS) * args.repetitions
    if len(measurements) != expected_count:
        raise RuntimeError('Incomplete result: {} of {} measurements'.format(len(measurements), expected_count))
    for dataset in DATASETS:
        for order in ORDERS:
            for ranks in args.ranks:
                groups = {v: [r for k, r in measurements.items() if k[2] == dataset
                              and k[3] == order and k[1] == ranks and k[4] == v]
                          for v in VARIANTS}
                for metric in ('setup_inclusive_ms', 'mpi_elapsed_ms',
                               'rank0_process_lifetime_ms', 'external_wall_ms'):
                    med = {v: statistics.median(float(r[metric]) for r in groups[v])
                           for v in VARIANTS}
                    pct = (med['baseline'] - med['warmed']) / med['baseline'] * 100
                    print('RESULT {} {} ranks={} metric={} baseline={:.3f}ms '
                          'warmed={:.3f}ms nodeaware={:.3f}ms warmup_gain={:+.1f}%'.format(
                          dataset, order, ranks, metric, med['baseline'], med['warmed'],
                          med['nodeaware'], pct), flush=True)
    print('PASS: {} measurements ({} resumed, {} new), validated all three versions'.format(
        len(measurements), len(previous), len(measurements) - len(previous)), flush=True)


if __name__ == '__main__':
    main()
