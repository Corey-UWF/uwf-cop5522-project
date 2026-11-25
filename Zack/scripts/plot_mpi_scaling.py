#!/usr/bin/env python3
"""COP5522 Expanse: scaling, sequential speedup, and parallel efficiency plots.

Python 3.6+; requires matplotlib.

Examples (run in ~/Final_Project):
  python3 plot_mpi_scaling.py \
    --sweep-csv expanse-benchmarks/nodeshared-1nodes-56035121.csv \
    --log-dir . --output-dir figures-mpi

  python3 plot_mpi_scaling.py \
    --fixed-csv expanse-benchmarks/mpi-fixed16-1nodes-56041901.csv \
                expanse-benchmarks/mpi-fixed16-2nodes-56041902.csv \
                expanse-benchmarks/mpi-fixed16-4nodes-56041903.csv \
    --log-dir . --output-dir figures-mpi

Both options may be supplied together. Assumes 'elapsed' values are milliseconds,
'wall' values are seconds, and sequential compute-only baseline values are
recorded in the matching Slurm .out log. This script NEVER substitutes a baseline
from another job. The sequential comparison is compute-oriented, not a true
end-to-end application speedup measurement.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit('matplotlib not installed. Install locally: python3 -m pip install matplotlib')

BASELINE_RE = re.compile(
    r'==\s*([^:\s]+)\s*:\s*sequential\s+v21\s+median\s*\(compute only\)\s*=\s*([\d.]+)\s*ms',
    re.IGNORECASE,
)
NODE_RE = re.compile(r'-(1|2|4)nodes-')
JOB_RE = re.compile(r'-(\d+)\.csv$')
VARIANT_ORDER = ['nodeshared', 'nodeaware', 'replicated', 'hybrid']
DISPLAY = {'nodeshared': 'Node-shared', 'nodeaware': 'Node-aware',
           'replicated': 'Replicated CSR', 'hybrid': 'Original hybrid'}
DATASET_NAMES = {
    'regular_AXW': 'Regular A(XW)', 'clustered_AXW': 'Clustered A(XW)',
    'regular_AX_W': 'Regular (AX)W', 'clustered_AX_W': 'Clustered (AX)W',
}
DATASET_ORDER = ['regular_AXW', 'clustered_AXW', 'regular_AX_W', 'clustered_AX_W']


def read_csv_data(path):
    path = Path(path)
    records = defaultdict(list)
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        required = {'dataset', 'ranks', 'variant', 'elapsed'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError('{} missing columns: {}'.format(path, sorted(missing)))
        for line, row in enumerate(reader, start=2):
            try:
                rank = int(row['ranks'])
                elapsed = float(row['elapsed'])
            except (TypeError, ValueError) as exc:
                raise ValueError('{} row {}: {}'.format(path, line, exc))
            if rank <= 0 or elapsed <= 0:
                raise ValueError('{} row {}: ranks and elapsed must be positive'.format(path, line))
            records[(row['dataset'], rank, row['variant'])].append(elapsed)
    if not records:
        raise ValueError('{} contains no measurements'.format(path))
    return {key: median(values) for key, values in records.items()}


def infer_log(csv_path, log_dir):
    name = Path(csv_path).name
    job_match = JOB_RE.search(name)
    if not job_match:
        raise ValueError('Cannot infer job ID from {}; specify --log-dir with matching logs'.format(name))
    job = job_match.group(1)
    candidates = [Path(log_dir) / ('nodeshared-{}.out'.format(job)),
                  Path(log_dir) / ('mpi-fixed16-{}.out'.format(job))]
    matches = [p for p in candidates if p.exists()]
    if not matches:
        raise FileNotFoundError('Missing matching Slurm log for {}. Expected one of: {}'.format(
            name, ', '.join(str(p) for p in candidates)))
    return matches[0]


def read_baselines(log_path):
    text = Path(log_path).read_text(errors='replace')
    baselines = {d: float(t) for d, t in BASELINE_RE.findall(text)}
    if not baselines:
        raise ValueError('No sequential v21 medians found in {}'.format(log_path))
    return baselines


def load_experiment(csv_path, log_dir):
    summary = read_csv_data(csv_path)
    log_path = infer_log(csv_path, log_dir)
    seq = read_baselines(log_path)
    missing = sorted(set(k[0] for k in summary) - set(seq))
    if missing:
        raise ValueError('Log {} lacks sequential baseline for {}'.format(log_path, missing))
    return summary, seq, log_path


def style_axes(ax, ylabel=None):
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', labelsize=9)


def finish(fig, axes, title, footnote, outfile):
    # Deliberately reserve distinct areas for title, legend, plot, and note.
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for h, label in zip(handles, labels):
        if label not in unique:
            unique[label] = h
    fig.suptitle(title, fontsize=15, weight='bold', y=.985)
    fig.legend(list(unique.values()), list(unique.keys()), ncol=min(len(unique), 4),
               loc='upper center', bbox_to_anchor=(.5, .932), frameon=False, fontsize=10)
    fig.text(.5, .025, footnote, ha='center', va='center', fontsize=9, color='dimgray')
    fig.subplots_adjust(top=.79, bottom=.17, left=.075, right=.985,
                        wspace=.22, hspace=.38)
    fig.savefig(str(outfile), dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Created {}'.format(outfile))


def plot_sweep(summary, seq, outdir):
    datasets = [d for d in DATASET_ORDER if any(k[0] == d for k in summary)]
    if not datasets:
        raise ValueError('Single-node sweep CSV lacks known dataset labels')
    variants = [v for v in VARIANT_ORDER if any(k[2] == v for k in summary)]
    if not variants:
        raise ValueError('Single-node sweep CSV lacks expected variants')
    rank_values = sorted({k[1] for k in summary})
    print('Single-node sweep ranks: {}'.format(rank_values))
    # True strong-scaling trend (constant workload, 1 physical node, varying ranks)
    for kind, filename, title, ylab, foot in [
        ('runtime', '01_single_node_strong_scaling.png',
         'Single-node strong scaling (fixed input, increasing MPI ranks)',
         'Median MPI elapsed (ms)',
         'Lower is better. Fixed workload; same physical node. MPI elapsed excludes input/output.'),
        ('speedup', '02_single_node_speedup_vs_sequential.png',
         'Speedup relative to sequential OpenBLAS v21',
         'Sequential compute-only / MPI elapsed',
         'Compute-oriented comparison only: sequential and MPI timing boundaries differ.'),
        ('efficiency', '03_single_node_parallel_efficiency.png',
         'MPI strong-scaling efficiency (relative to 1 MPI rank)',
         'Efficiency = T(1 rank) / [P × T(P ranks)]',
         'MPI-only strong-scaling efficiency; each variant uses its own 1-rank baseline.'),
    ]:
        nrows = (len(datasets) + 1) // 2
        fig, aa = plt.subplots(nrows, 2, figsize=(12.6, 4.9 * nrows))
        axes = list(aa.flat) if hasattr(aa, 'flat') else [aa]
        for i, d in enumerate(datasets):
            ax = axes[i]
            for v in variants:
                rank_times = [(r, summary[(d, r, v)]) for r in rank_values if (d, r, v) in summary]
                if not rank_times:
                    continue
                if kind == 'efficiency' and (d, 1, v) not in summary:
                    print('WARNING: no rank-1 baseline for {} {}. Skipping efficiency.'.format(d, v))
                    continue
                x = [r for r, _ in rank_times]
                if kind == 'runtime':
                    y = [t for _, t in rank_times]
                elif kind == 'speedup':
                    y = [seq[d] / t for _, t in rank_times]
                else:
                    y = [summary[(d, 1, v)] / (r * t) for r, t in rank_times]
                ax.plot(x, y, marker='o', linewidth=2, label=DISPLAY.get(v, v))
            if kind == 'speedup':
                ax.axhline(1.0, linestyle=':', linewidth=1, color='gray')
            if kind == 'efficiency':
                ax.axhline(1.0, linestyle=':', linewidth=1, color='gray')
            ax.set_title(DATASET_NAMES.get(d, d), fontsize=11, weight='bold')
            ax.set_xticks(rank_values)
            ax.set_xlabel('MPI ranks')
            style_axes(ax, ylab)
        for ax in axes[len(datasets):]:
            ax.set_visible(False)
        finish(fig, axes, title, foot, outdir / filename)


def plot_fixed(experiments, outdir):
    counts = sorted(experiments)
    datasets = [d for d in DATASET_ORDER if any(any(k[0] == d for k in v[0]) for v in experiments.values())]
    variants = [v for v in VARIANT_ORDER if any(any(k[2] == v for k in z[0]) for z in experiments.values())]
    if not datasets or not variants:
        raise ValueError('No matching datasets or variants in fixed-rank CSVs')
    for kind, filename, title, ylab, foot in [
        ('elapsed', '04_fixed16_node_placement_runtime.png',
         'Fixed 16-rank MPI placement: algorithm time', 'Median MPI elapsed (ms)',
         'Same 16 MPI ranks and problem size at every node count. Not conventional strong scaling.'),
        ('seq_speedup', '05_fixed16_speedup_vs_sequential.png',
         'Fixed 16-rank placement: speedup vs sequential v21',
         'Sequential compute-only / MPI elapsed',
         'Compute-oriented comparison; sequential baseline is read from each matching job log.'),
    ]:
        nrows = (len(datasets) + 1) // 2
        fig, aa = plt.subplots(nrows, 2, figsize=(12.6, 4.9 * nrows))
        axes = list(aa.flat) if hasattr(aa, 'flat') else [aa]
        for i, d in enumerate(datasets):
            ax = axes[i]
            for v in variants:
                xs, ys = [], []
                for n in counts:
                    summary, seq = experiments[n]
                    # fixed16 experiment: only 16 rank records
                    if (d, 16, v) in summary:
                        t = summary[(d, 16, v)]
                        xs.append(n)
                        ys.append(t if kind == 'elapsed' else seq[d] / t)
                if xs:
                    ax.plot(xs, ys, marker='o', linewidth=2, label=DISPLAY.get(v, v))
            if kind == 'seq_speedup':
                ax.axhline(1, linestyle=':', linewidth=1, color='gray')
            ax.set_title(DATASET_NAMES.get(d, d), fontsize=11, weight='bold')
            ax.set_xticks(counts)
            ax.set_xlabel('Physical nodes (16 MPI ranks total)')
            style_axes(ax, ylab)
        for ax in axes[len(datasets):]:
            ax.set_visible(False)
        finish(fig, axes, title, foot, outdir / filename)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep-csv', help='Single-node rank sweep (nodeshared-1nodes-JOB.csv)')
    ap.add_argument('--fixed-csv', nargs='+', help='One or more mpi-fixed16-Nnodes-JOB.csv')
    ap.add_argument('--log-dir', default='.', help='Directory of matching Slurm .out files (default: current dir)')
    ap.add_argument('--output-dir', default='figures-mpi', help='Output folder for PNGs')
    args = ap.parse_args()
    if not args.sweep_csv and not args.fixed_csv:
        ap.error('provide --sweep-csv and/or --fixed-csv')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.sweep_csv:
        summary, seq, log = load_experiment(args.sweep_csv, args.log_dir)
        print('Single-node sweep matched log: {}'.format(log))
        plot_sweep(summary, seq, outdir)
    if args.fixed_csv:
        fixed = {}
        for path in args.fixed_csv:
            m = NODE_RE.search(Path(path).name)
            if not m:
                raise ValueError('Fixed CSV filename must include -1nodes-, -2nodes-, or -4nodes-: {}'.format(path))
            n = int(m.group(1))
            if n in fixed:
                raise ValueError('Duplicate fixed-rank dataset for {} nodes'.format(n))
            summary, seq, log = load_experiment(path, args.log_dir)
            print('{} nodes matched log: {}'.format(n, log))
            fixed[n] = (summary, seq)
        plot_fixed(fixed, outdir)
    print('Finished. Figures saved to {}'.format(outdir))


if __name__ == '__main__':
    main()
