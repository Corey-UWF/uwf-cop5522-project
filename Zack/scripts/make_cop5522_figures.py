#!/usr/bin/env python3
"""COP5522 sequential versus MPI figures from final Expanse benchmark CSVs.

Run from Final_Project root. Requires matplotlib (`python -m pip install matplotlib`).
Uses the median of 10 repetitions, with MPI setup-inclusive timings.
No timings or benchmark results are fabricated or embedded.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULTS = {
    'sequential': 'expanse-benchmarks/sequential-baseline-56025892.csv',
    'one': 'expanse-benchmarks/mpi-multinode-1nodes-56005681.csv',
    'two': 'expanse-benchmarks/mpi-multinode-2nodes-56009407.csv',
    'four': 'expanse-benchmarks/mpi-multinode-4nodes-resumed-56024513.csv',
}
DATASETS = ('large_regular', 'clustered_strong')
ORDERS = ('A(XW)', '(AX)W')
METHODS = ('adaptive', 'balanced', 'hybrid')


def rows_from(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing {}. Copy the CSV into the expected path or supply an override.'.format(path))
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        p.add_argument('--' + k, default=v)
    p.add_argument('--output-dir', default='figures')
    args = p.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    seq = defaultdict(list)
    for r in rows_from(args.sequential):
        seq[(r['dataset'], r['order'])].append(float(r['time_ms']))

    mpi = defaultdict(list)
    for file_key in ('one', 'two', 'four'):
        for r in rows_from(getattr(args, file_key)):
            if r.get('phase') not in ('', 'measured', None):
                continue
            key = (r['dataset'], r['order'], int(r['nodes']), int(r['ranks']), r['method'])
            mpi[key].append(float(r['mpi_setup_inclusive_s']) * 1000.0)

    for dataset in DATASETS:
        for order in ORDERS:
            if len(seq[(dataset, order)]) != 10:
                raise ValueError('Expected 10 sequential records for {} {}; found {}'.format(dataset, order, len(seq[(dataset, order)])))
            for nodes, ranks_list in ((1, (16,)), (2, (16, 32)), (4, (16, 32, 64))):
                for ranks in ranks_list:
                    for method in METHODS:
                        key = (dataset, order, nodes, ranks, method)
                        if len(mpi[key]) != 10:
                            raise ValueError('Expected 10 MPI records for {}; found {}'.format(key, len(mpi[key])))

    labels = ['Regular\nA(XW)', 'Regular\n(AX)W', 'Clustered\nA(XW)', 'Clustered\n(AX)W']
    workloads = [(d, o) for d in DATASETS for o in ORDERS]
    med_seq = [median(seq[k]) for k in workloads]
    best_mpi = []
    best_methods = []
    for d, o in workloads:
        candidates = [(median(mpi[(d, o, 1, 16, m)]), m) for m in METHODS]
        t, method = min(candidates)
        best_mpi.append(t)
        best_methods.append(method)

    # Figure 1: baseline compared with best 16-rank MPI on one node.
    x = list(range(4))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    a = ax.bar([i - .19 for i in x], med_seq, width=.38, label='Sequential (1 OpenBLAS thread)')
    b = ax.bar([i + .19 for i in x], best_mpi, width=.38, label='Best MPI method (16 ranks, 1 node)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median computation / MPI setup-inclusive time (ms)')
    ax.set_title('Sequential baseline versus 16-rank MPI (N = 40,000)')
    ax.legend()
    ax.grid(axis='y', alpha=.25)
    fig.tight_layout()
    fig.savefig(output / '01_sequential_vs_mpi.png', dpi=200)
    plt.close(fig)

    # Figure 2: fixed 16 ranks distributed over 1,2,4 nodes, by workload.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for d, o in workloads:
        label = '{} {}'.format('Regular' if d == 'large_regular' else 'Clustered', o)
        timings = [min(median(mpi[(d, o, n, 16, m)]) for m in METHODS) for n in (1, 2, 4)]
        ax.plot([1, 2, 4], timings, marker='o', label=label)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel('Number of nodes (16 MPI ranks total)')
    ax.set_ylabel('Median setup-inclusive time (ms)')
    ax.set_title('Fixed-rank multi-node overhead (best method per workload)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / '02_fixed_16_ranks_nodes.png', dpi=200)
    plt.close(fig)

    # Figure 3: hybrid/adaptive/balanced comparison for clustered (AX)W.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    configs = [(1, 16), (2, 16), (2, 32), (4, 16), (4, 32), (4, 64)]
    for m in METHODS:
        timings = [median(mpi[('clustered_strong', '(AX)W', n, r, m)]) for n, r in configs]
        ax.plot(range(len(configs)), timings, marker='o', label=m.capitalize())
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(['{}n/{}r'.format(n, r) for n, r in configs])
    ax.set_xlabel('Nodes / MPI ranks')
    ax.set_ylabel('Median setup-inclusive time (ms)')
    ax.set_title('Partitioning comparison: clustered matrix, (AX)W')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / '03_partitioning_comparison.png', dpi=200)
    plt.close(fig)

    with (output / 'sequential_mpi_comparison.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'order', 'sequential_median_ms', 'best_mpi_16r_1n_median_ms', 'best_mpi_method', 'speedup_sequential_over_mpi'])
        for (d, o), ts, tm, method in zip(workloads, med_seq, best_mpi, best_methods):
            w.writerow([d, o, '{:.6f}'.format(ts), '{:.6f}'.format(tm), method, '{:.6f}'.format(ts / tm)])

    print('Created 3 PNG figures and sequential_mpi_comparison.csv in {}'.format(output))
    print('Timing: median over 10; MPI uses mpi_setup_inclusive_s (excluding file I/O and MPI startup).')
    for (d, o), ts, tm, method in zip(workloads, med_seq, best_mpi, best_methods):
        print('{} {}: sequential {:.3f} ms; MPI {:.3f} ms ({}); speedup {:.3f}x'.format(d, o, ts, tm, method, ts / tm))


if __name__ == '__main__':
    main()
