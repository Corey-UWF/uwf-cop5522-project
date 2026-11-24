#!/usr/bin/env python3
"""Plot sequential-relative MPI speedup at ranks 1,2,4,8,16.

Inputs are original Expanse single-node benchmark CSVs. The old MPI timers
exclude partition setup and file I/O; the sequential timer covers the two
matrix multiplication stages only. Their timed regions are not identical.
Run from the Zack directory or pass explicit file paths.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANKS = (1, 2, 4, 8, 16)
DATASETS = ('large_regular', 'clustered_strong')
ORDER_SOURCES = (('A(XW)', 'large'), ('(AX)W', 'reverse'))
METHODS = ('adaptive', 'balanced', 'hybrid')


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing input CSV: {}'.format(path))
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequential', default='expanse-benchmarks/sequential-baseline-56025892.csv')
    p.add_argument('--large', default='expanse-benchmarks/mpi-large-single-55961832.csv')
    p.add_argument('--reverse', default='expanse-benchmarks/mpi-hybrid-single-56002005.csv',
                   help='(AX)W benchmark with all 3 strategies; use the hybrid experiment CSV')
    p.add_argument('--output-dir', default='figures')
    args = p.parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    seq = defaultdict(list)
    for r in read_csv(args.sequential):
        seq[(r['dataset'], r['order'])].append(float(r['time_ms']))

    mpi = defaultdict(list)
    for source, filename in [('large', args.large), ('reverse', args.reverse)]:
        expected_order = 'A(XW)' if source == 'large' else '(AX)W'
        for r in read_csv(filename):
            if r.get('phase', 'measured') != 'measured':
                continue
            order = r.get('order', expected_order) or expected_order
            if order != expected_order:
                raise ValueError('Unexpected order in {}: {}'.format(filename, order))
            key = (r['dataset'], order, int(r['ranks']), r['method'])
            mpi[key].append(1000 * float(r['mpi_elapsed_s']))

    for dataset in DATASETS:
        for order, _ in ORDER_SOURCES:
            if len(seq[(dataset, order)]) != 10:
                raise ValueError('Sequential {} {}: expected 10 repetitions; got {}'.format(
                    dataset, order, len(seq[(dataset, order)])))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    records = []
    for row_idx, dataset in enumerate(DATASETS):
        for col_idx, (order, _) in enumerate(ORDER_SOURCES):
            ax = axes[row_idx][col_idx]
            baseline = median(seq[(dataset, order)])
            for method in METHODS:
                points = []
                for ranks in RANKS:
                    samples = mpi[(dataset, order, ranks, method)]
                    if not samples:
                        continue
                    med = median(samples)
                    speedup = baseline / med
                    points.append((ranks, speedup))
                    records.append((dataset, order, method, ranks,
                                    len(samples), baseline, med, speedup))
                    if len(samples) != 10:
                        print('WARNING: {} {} {} p={} has {} runs, not 10'.format(
                            dataset, order, method, ranks, len(samples)))
                if points:
                    ax.plot([x for x, _ in points], [y for _, y in points],
                            marker='o', linewidth=1.8, label=method.capitalize())
            ax.axhline(1, linestyle='--', color='gray', alpha=.7)
            ax.set_xticks(RANKS)
            ax.set_xscale('log', base=2)
            ax.set_title('{} · {}'.format(
                'Regular' if dataset == 'large_regular' else 'Strongly clustered', order))
            ax.grid(alpha=.25)
            if row_idx == 1:
                ax.set_xlabel('MPI ranks on one node')
            if col_idx == 0:
                ax.set_ylabel('Sequential-relative speedup (×)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .005))
    fig.suptitle('Sequential-relative MPI speedup | N = 40,000', fontsize=14)
    fig.text(.5, .045,
             'Sequential: compute-only. MPI: internal elapsed, excludes partition setup and I/O. '
             'Timer boundaries differ.',
             ha='center', va='center', fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, .95))
    path_png = outdir / '04_single_node_sequential_speedup.png'
    fig.savefig(path_png, dpi=220)
    plt.close(fig)

    path_csv = outdir / 'single_node_sequential_speedup.csv'
    with path_csv.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'order', 'method', 'ranks', 'n_mpi',
                         'sequential_median_ms', 'mpi_elapsed_median_ms',
                         'sequential_relative_speedup'])
        for dataset, order, method, ranks, n, seq_ms, mpi_ms, sp in records:
            writer.writerow([dataset, order, method, ranks, n,
                             '{:.6f}'.format(seq_ms), '{:.6f}'.format(mpi_ms),
                             '{:.6f}'.format(sp)])
    print('Created:', path_png)
    print('Created:', path_csv)
    print('Recorded {} points. Speedup > 1 means MPI faster by these timing definitions.'.format(len(records)))
    print('Caution: the original MPI timer does not include partition setup.')


if __name__ == '__main__':
    main()
