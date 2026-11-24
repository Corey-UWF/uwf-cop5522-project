#!/usr/bin/env python3
"""Print median and spread from a mpi_benchmark_driver CSV (Python stdlib only)."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import sys

if len(sys.argv) != 2:
    raise SystemExit("Usage: python3 summarize_mpi_benchmarks.py expanse-benchmarks/mpi-single-node-JOBID.csv")
rows = list(csv.DictReader(Path(sys.argv[1]).open()))
groups = defaultdict(list)
for row in rows:
    groups[(row["dataset"], int(row["ranks"]), row["method"])].append(float(row["mpi_elapsed_s"]) * 1000)
print("dataset,ranks,method,n,median_ms,min_ms,max_ms")
for (dataset, ranks, method), vals in sorted(groups.items()):
    print(f"{dataset},{ranks},{method},{len(vals)},{statistics.median(vals):.6f},{min(vals):.6f},{max(vals):.6f}")
print("\nBalanced / equal-row median ratio (less than 1 means balanced faster):")
for dataset, ranks in sorted({(k[0], k[1]) for k in groups}):
    a = groups.get((dataset, ranks, "adaptive"))
    b = groups.get((dataset, ranks, "balanced"))
    if a and b:
        print(f"{dataset:19s} p={ranks:2d}  ratio={statistics.median(b)/statistics.median(a):.4f}")
