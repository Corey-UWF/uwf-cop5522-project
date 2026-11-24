#!/usr/bin/env python3
"""Generate COP5522 diagnostic figures from original Expanse benchmark CSVs.

Run from Zack/ (or pass relative paths from Zack/scripts/).
All data points are based on median measured runs, never made-up timings.

05: single-node strong scaling / efficiency for regular A(XW) and clustered (AX)W.
06: instrumented MPI phase times at 16 ranks across 1, 2, and 4 nodes.
07: clustered (AX)W row-count / computation imbalance and runtime on one node.

Notes:
- Single-node historical 'mpi_elapsed_s' omits partition setup, whereas
  multi-node 'mpi_setup_inclusive_s' includes it; they are NOT spliced into
  one misleading continuous speedup line.
- MPI phase counters can be overlapping/critical-path durations, so the
  phase figure uses grouped bars, NOT a stacked bar claiming exact additivity.
- P=1 MPI time (not the sequential OpenBLAS time) is the baseline for the
  strong-scaling and efficiency plots.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('adaptive', 'balanced', 'hybrid')
RANKS = (1, 2, 4, 8, 16)


def load(path, expected_order=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing CSV: {}'.format(path))
    with path.open(newline='') as fp:
        records = list(csv.DictReader(fp))
    if not records:
        raise ValueError('Empty CSV: {}'.format(path))
    if expected_order:
        for r in records:
            r['order'] = r.get('order') or expected_order
    return [r for r in records if r.get('phase', 'measured') in ('', 'measured')]


def collect(data, field, filters):
    values = []
    for r in data:
        if all(str(r.get(k)) == str(v) for k, v in filters.items()):
            value = r.get(field, '')
            if value not in ('', None):
                values.append(float(value))
    return values


def get_median(data, field, filters):
    points = collect(data, field, filters)
    return median(points) if points else None


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Created:', path)


def scaling_figure(large, reverse, outdir):
    # The independent baseline for every series is the SAME method's P=1 MPI time.
    workloads = [
        ('Regular A(XW)', large, 'large_regular', 'A(XW)', ('adaptive', 'balanced')),
        ('Clustered (AX)W', reverse, 'clustered_strong', '(AX)W', METHODS),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    for i, (title, rows, dataset, order, methods) in enumerate(workloads):
        ax_s, ax_e = axes[i]
        for method in methods:
            baseline = get_median(rows, 'mpi_elapsed_s', dict(dataset=dataset, order=order, ranks=1, method=method))
            if baseline is None:
                print('WARNING: no P=1 baseline for {} {}'.format(title, method))
                continue
            rankvals, speedups, efficiency = [], [], []
            for ranks in RANKS:
                current = get_median(rows, 'mpi_elapsed_s', dict(dataset=dataset, order=order, ranks=ranks, method=method))
                if current is None:
                    continue
                s = baseline/current
                rankvals.append(ranks)
                speedups.append(s)
                efficiency.append(100*s/ranks)
            ax_s.plot(rankvals, speedups, marker='o', label=method.capitalize())
            ax_e.plot(rankvals, efficiency, marker='o', label=method.capitalize())
        ax_s.plot(RANKS, RANKS, linestyle=':', color='gray', label='Ideal speedup')
        ax_e.axhline(100, linestyle=':', color='gray', label='Ideal efficiency')
        for ax in (ax_s, ax_e):
            ax.set_xscale('log', base=2)
            ax.set_xticks(RANKS)
            ax.set_xticklabels([str(x) for x in RANKS])
            ax.grid(alpha=.25)
            ax.set_xlabel('MPI ranks on one node')
        ax_s.set_ylabel('Speedup vs. own P=1 MPI time (×)')
        ax_e.set_ylabel('Parallel efficiency (%)')
        ax_s.set_title(title + ': strong scaling')
        ax_e.set_title(title + ': efficiency')
        ax_s.legend(fontsize=8)
        ax_e.legend(fontsize=8)
    fig.suptitle('Single-node MPI strong scaling (N = 40,000; internal MPI elapsed time)', y=1.02)
    fig.tight_layout()
    save(fig, outdir/'05_mpi_strong_scaling_efficiency.png')


def phases_figure(node_csvs, outdir):
    # Grouped, not stacked: the MPI timing instrumentation does not guarantee
    # that all phases partition the elapsed wall time perfectly.
    key = dict(dataset='clustered_strong', order='(AX)W', method='hybrid', ranks=16)
    phase_cols = [('distribution_s','CSR distribution'),
                  ('root_xw_s','Root XW'), ('broadcast_s','Dense broadcast'),
                  ('local_compute_s','Local compute'), ('gather_s','Output gather')]
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    nodes = [1, 2, 4]
    width = .155
    for i, (field, label) in enumerate(phase_cols):
        ys = []
        for n in nodes:
            v = get_median(node_csvs[n], field, dict(key, nodes=n))
            ys.append(v*1000 if v is not None else float('nan'))
        ax.bar([x + (i-2)*width for x in range(3)], ys, width=width, label=label)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['1 node / 16 ranks', '2 nodes / 16 ranks', '4 nodes / 16 ranks'])
    ax.set_ylabel('Median reported phase time (ms)')
    ax.set_title('MPI phase timings: clustered (AX)W, hybrid, fixed 16 ranks')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=8, ncol=3)
    ax.text(.5, -.19, 'Individual phase timers may overlap; bars must not be summed into total runtime.',
            transform=ax.transAxes, ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, outdir/'06_mpi_phase_breakdown.png')


def balance_figure(reverse, outdir):
    # Use historical single-node comparison with same input and P=2,4,8,16.
    methods = METHODS
    ranks = [2, 4, 8, 16]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    keys = [('rows_min', 'rows_max', 'Max/min rows per rank'),
            ('rank_compute_avg_s', 'rank_compute_max_s', 'Max/avg rank compute time'),
            (None, 'mpi_elapsed_s', 'Total MPI elapsed time (ms)')]
    for i, (denom_col, num_col, title) in enumerate(keys):
        ax = axes[i]
        for method in methods:
            xs, ys = [], []
            for p in ranks:
                filt = dict(dataset='clustered_strong', order='(AX)W', method=method, ranks=p)
                numerator = get_median(reverse, num_col, filt)
                denominator = get_median(reverse, denom_col, filt) if denom_col else None
                if numerator is None or (denom_col and (denominator is None or denominator == 0)):
                    continue
                v = (numerator/denominator) if denom_col else numerator*1000
                xs.append(p)
                ys.append(v)
            if xs:
                ax.plot(xs, ys, marker='o', label=method.capitalize())
        ax.set_title(title)
        ax.set_xticks(ranks)
        ax.set_xlabel('MPI ranks on one node')
        ax.grid(alpha=.25)
        if i == 0:
            ax.set_ylabel('Ratio (larger = less balanced)')
        elif i == 1:
            ax.set_ylabel('Ratio (larger = less balanced)')
        else:
            ax.set_ylabel('Median runtime (ms)')
        ax.legend(fontsize=8)
    fig.suptitle('Partitioning effects: strongly clustered (AX)W (N = 40,000)', y=1.04)
    fig.tight_layout()
    save(fig, outdir/'07_mpi_load_balance.png')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--large', default='expanse-benchmarks/mpi-large-single-55961832.csv')
    p.add_argument('--reverse', default='expanse-benchmarks/mpi-hybrid-single-56002005.csv')
    p.add_argument('--one', default='expanse-benchmarks/mpi-multinode-1nodes-56005681.csv')
    p.add_argument('--two', default='expanse-benchmarks/mpi-multinode-2nodes-56009407.csv')
    p.add_argument('--four', default='expanse-benchmarks/mpi-multinode-4nodes-resumed-56024513.csv')
    p.add_argument('--output-dir', default='figures')
    a = p.parse_args()
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    large = load(a.large, expected_order='A(XW)')
    reverse = load(a.reverse, expected_order='(AX)W')
    multinode = {1:load(a.one), 2:load(a.two), 4:load(a.four)}
    scaling_figure(large, reverse, output)
    phases_figure(multinode, output)
    balance_figure(reverse, output)
    print('Finished. See figures 05, 06, and 07 in', output)


if __name__ == '__main__':
    main()
