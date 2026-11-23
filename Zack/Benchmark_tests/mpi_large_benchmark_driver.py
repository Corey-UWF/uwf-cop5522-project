#!/usr/bin/env python3
"""Large-input single-node MPI benchmarking driver for COP5522.
Records program-internal MPI elapsed time, which currently excludes partition
setup and filesystem read/write. Also records externally observed wall time.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

DATASETS = {
    "large_regular": "large-inputs/regular-A.csr",
    "shuffled_moderate": "large-inputs/shuffled-moderate-A.csr",
    "shuffled_strong": "large-inputs/shuffled-strong-A.csr",
    "clustered_moderate": "large-inputs/clustered-moderate-A.csr",
    "clustered_strong": "large-inputs/clustered-strong-A.csr",
}
PHASE_LABELS = {
    "CSR distribution": "distribution_s",
    "Root XW": "root_xw_s",
    "Dense broadcast": "broadcast_s",
    "Local computation": "local_compute_s",
    "Output gathering": "gather_s",
    "MPI elapsed time": "mpi_elapsed_s",
}
FIELDNAMES = [
    "dataset", "method", "ranks", "repetition", "phase", "mpi_elapsed_s",
    "distribution_s", "root_xw_s", "broadcast_s", "local_compute_s",
    "gather_s", "wall_s", "nonzero_imbalance", "rows_min", "rows_max",
    "nnz_min", "nnz_max", "rank_compute_min_s", "rank_compute_max_s",
    "rank_compute_avg_s", "job_id", "node",
]

def parse_output(output):
    record = {field: "" for field in FIELDNAMES}
    for label, key in PHASE_LABELS.items():
        match = re.search(r"^" + re.escape(label) + r"\s*=\s*([0-9.eE+-]+)\s+seconds", output, re.M)
        if not match:
            raise ValueError(f"Missing timing: {label}")
        record[key] = float(match.group(1))
    m = re.search(r"^Nonzero imbalance ratio\s*=\s*([0-9.eE+-]+)", output, re.M)
    if m:
        record["nonzero_imbalance"] = float(m.group(1))
    m = re.search(r"^Rows per rank: min=(\d+) max=(\d+)", output, re.M)
    if m:
        record["rows_min"], record["rows_max"] = map(int, m.groups())
    m = re.search(r"^Nonzeros per rank: min=(\d+) max=(\d+)", output, re.M)
    if m:
        record["nnz_min"], record["nnz_max"] = map(int, m.groups())
    m = re.search(r"^Rank-local computation: min=([0-9.eE+-]+) max=([0-9.eE+-]+) average=([0-9.eE+-]+) seconds", output, re.M)
    if m:
        record["rank_compute_min_s"], record["rank_compute_max_s"], record["rank_compute_avg_s"] = map(float, m.groups())
    return record

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--logdir", type=Path, required=True)
    args = ap.parse_args()
    if args.warmups < 0 or args.repetitions <= 0 or any(p <= 0 for p in args.ranks):
        ap.error("warmups must be >=0; repetitions and ranks must be positive")
    allocated = int(os.environ.get("SLURM_NTASKS", max(args.ranks)))
    if max(args.ranks) > allocated:
        ap.error(f"Requested {max(args.ranks)} ranks but only {allocated} tasks allocated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)
    output_y = args.logdir / "scratch-output.dense"  # Overwritten each run, excluded from internal timing
    written = 0
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()
        for dataset, csr in DATASETS.items():
            for p in args.ranks:
                for rep in range(-args.warmups, args.repetitions):
                    # Alternate order each repetition to partially reduce order bias.
                    methods = ("adaptive", "balanced") if rep % 2 == 0 else ("balanced", "adaptive")
                    for method in methods:
                        cmd = ["mpirun", "--bind-to", "core", "--map-by", "core", "-np", str(p),
                               f"./mpi-{method}", csr, "large-inputs/X.dense", "large-inputs/W.dense", str(output_y)]
                        t0 = time.perf_counter()
                        result = subprocess.run(cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        elapsed = time.perf_counter() - t0
                        if result.returncode != 0:
                            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
                        if rep < 0:
                            print(f"WARMUP {dataset} {method} p={p}", flush=True)
                            continue
                        rec = parse_output(result.stdout)
                        rec.update(dataset=dataset, method=method, ranks=p, repetition=rep + 1,
                                   phase="measured", wall_s=elapsed,
                                   job_id=os.environ.get("SLURM_JOB_ID", ""), node=os.environ.get("SLURMD_NODENAME", ""))
                        writer.writerow(rec)
                        f.flush()
                        written += 1
                        print(f"{dataset:19s} {method:8s} ranks={p:2d} rep={rep+1:2d}: "
                              f"MPI={rec['mpi_elapsed_s']*1000:.4f} ms wall={elapsed*1000:.3f} ms", flush=True)
    print(f"DONE: {written} measured runs. CSV: {args.output}", flush=True)

if __name__ == "__main__":
    main()
