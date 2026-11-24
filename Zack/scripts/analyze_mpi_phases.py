import csv
import statistics
from collections import defaultdict

filename = "expanse-benchmarks/mpi-single-node-55961672.csv"

metrics = [
    "mpi_elapsed_s",
    "distribution_s",
    "root_xw_s",
    "broadcast_s",
    "local_compute_s",
    "gather_s",
    "wall_s",
]

groups = defaultdict(list)

with open(filename) as f:
    for row in csv.DictReader(f):
        if row["phase"] != "measured":
            continue
        groups[(row["dataset"], row["ranks"], row["method"])].append(row)

print(
    "{:<20} {:>5} {:<9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "Dataset", "Ranks", "Method",
        "Total", "Distrib", "Root XW",
        "Bcast", "Compute", "Gather"
    )
)

for key in sorted(groups):
    dataset, ranks, method = key
    rows = groups[key]

    medians = {
        col: 1000 * statistics.median(float(r[col]) for r in rows)
        for col in metrics
    }

    print(
        "{:<20} {:>5} {:<9} {:>9.3f} {:>9.3f} {:>9.3f} "
        "{:>9.3f} {:>9.3f} {:>9.3f}".format(
            dataset, ranks, method,
            medians["mpi_elapsed_s"],
            medians["distribution_s"],
            medians["root_xw_s"],
            medians["broadcast_s"],
            medians["local_compute_s"],
            medians["gather_s"]
        )
    )
