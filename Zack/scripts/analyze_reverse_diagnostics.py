import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

files = glob.glob("expanse-benchmarks/mpi-reverse-single-*.csv")
if not files:
    sys.exit("ERROR: Reverse benchmark CSV not found.")

filename = max(files, key=os.path.getmtime)
groups = defaultdict(list)

with open(filename) as f:
    for row in csv.DictReader(f):
        if row["phase"] == "measured":
            key = (row["dataset"], int(row["ranks"]), row["method"])
            groups[key].append(row)

def median(rows, field):
    values = []
    for row in rows:
        value = row.get(field, "")
        if value not in ("", None):
            values.append(float(value))
    return statistics.median(values) if values else None

def ms(rows, field):
    value = median(rows, field)
    return 1000 * value if value is not None else None

print("File:", filename)
print("Measured records:", sum(len(v) for v in groups.values()))

print("\n=== CLUSTERED MATRIX PHASE TIMINGS (ms) ===")
print("{:<19} {:>3} {:<9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "Dataset", "P", "Method", "Total", "Distrib",
    "RootXW", "Bcast", "Compute", "Gather", "Wall"
))

for dataset in ("clustered_moderate", "clustered_strong"):
    for p in (1, 2, 4, 8, 16):
        for method in ("adaptive", "balanced"):
            rows = groups.get((dataset, p, method), [])
            if not rows:
                continue
            fields = (
                "mpi_elapsed_s", "distribution_s", "root_xw_s",
                "broadcast_s", "local_compute_s", "gather_s", "wall_s"
            )
            values = [ms(rows, field) for field in fields]
            print("{:<19} {:>3} {:<9} {}".format(
                dataset, p, method,
                " ".join("{:>8.3f}".format(v) if v is not None
                         else "{:>8}".format("N/A") for v in values)
            ))

print("\n=== BALANCED WORKLOAD STATISTICS ===")
print("{:<19} {:>3} {:>11} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
    "Dataset", "P", "NNZ ratio", "Min rows", "Max rows",
    "Min compute", "Max compute", "Avg compute"
))

for dataset in ("clustered_moderate", "clustered_strong"):
    for p in (1, 2, 4, 8, 16):
        rows = groups.get((dataset, p, "balanced"), [])
        if not rows:
            continue
        fields = (
            "nonzero_imbalance", "rows_min", "rows_max",
            "rank_compute_min_s", "rank_compute_max_s",
            "rank_compute_avg_s"
        )
        values = [median(rows, field) for field in fields]
        values[3:] = [1000 * v if v is not None else None
                      for v in values[3:]]
        print("{:<19} {:>3} {}".format(
            dataset, p,
            " ".join("{:>12.3f}".format(v) if v is not None
                     else "{:>12}".format("N/A") for v in values)
        ))

print("\n=== BALANCED MINUS EQUAL-ROW PHASE DIFFERENCES (ms) ===")
print("Positive values mean balanced is slower.")
for dataset in ("clustered_moderate", "clustered_strong"):
    for p in (2, 4, 8, 16):
        a = groups.get((dataset, p, "adaptive"), [])
        b = groups.get((dataset, p, "balanced"), [])
        if not a or not b:
            continue
        print("\n{} | {} ranks".format(dataset, p))
        for field in (
            "mpi_elapsed_s", "distribution_s", "root_xw_s",
            "broadcast_s", "local_compute_s", "gather_s", "wall_s"
        ):
            av, bv = ms(a, field), ms(b, field)
            if av is not None and bv is not None:
                print("  {:<22} {:+.3f} ms".format(field, bv - av))
