#!/usr/bin/env python3
"""Resumable COP5522 multi-node MPI timing driver (Python >= 3.6)."""
import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import time

DATASETS = {"large_regular": "large-inputs/regular-A.csr",
            "clustered_strong": "large-inputs/clustered-strong-A.csr"}
ORDERS = {"axw": ("large-inputs/X.dense", "large-inputs/W.dense", "A(XW)"),
          "reverse": ("reverse-inputs/X.dense", "reverse-inputs/W.dense", "(AX)W")}
METHODS = ("adaptive", "balanced", "hybrid")
FIELDS = ["job_id", "nodes", "ranks", "ranks_per_node", "dataset", "order", "method",
          "repetition", "phase", "partition_setup_s", "mpi_elapsed_s",
          "mpi_setup_inclusive_s", "distribution_s", "root_xw_s", "broadcast_s",
          "local_compute_s", "gather_s", "wall_s", "nonzero_imbalance", "rows_min",
          "rows_max", "rank_compute_min_s", "rank_compute_max_s", "rank_compute_avg_s"]
LABELS = {"Partition setup": "partition_setup_s", "MPI elapsed time": "mpi_elapsed_s",
          "MPI setup-inclusive time": "mpi_setup_inclusive_s", "CSR distribution": "distribution_s",
          "Root XW": "root_xw_s", "Dense broadcast": "broadcast_s",
          "Local computation": "local_compute_s", "Output gathering": "gather_s"}


def parse_output(output):
    record = {k: "" for k in FIELDS}
    for label, field in LABELS.items():
        m = re.search(r"^" + re.escape(label) + r"\s*=\s*([\d.eE+-]+)\s+seconds", output, re.M)
        if m is None:
            raise RuntimeError("Missing required timing {} in:\n{}".format(label, output[-2000:]))
        record[field] = float(m.group(1))
    for pattern, keys in ((r"^Rows per rank: min=(\d+) max=(\d+)", ("rows_min", "rows_max")),
                          (r"^Nonzero imbalance ratio = ([\d.eE+-]+)", ("nonzero_imbalance",)),
                          (r"^Rank-local computation: min=([\d.eE+-]+) max=([\d.eE+-]+) average=([\d.eE+-]+) seconds",
                           ("rank_compute_min_s", "rank_compute_max_s", "rank_compute_avg_s"))):
        m = re.search(pattern, output, re.M)
        if m:
            for key, value in zip(keys, m.groups()):
                record[key] = float(value)
    return record


def key_for(row):
    return (row["dataset"], row["order"], int(row["ranks"]), row["method"], int(row["repetition"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, required=True, choices=(1, 2, 4))
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Previous partial CSV (never modified)")
    args = parser.parse_args()
    allocated = int(os.environ.get("SLURM_JOB_NUM_NODES", "0"))
    if allocated != args.nodes:
        parser.error("Allocated nodes {} != requested nodes {}".format(allocated, args.nodes))
    for ranks in args.ranks:
        if ranks % args.nodes or ranks // args.nodes > 16 or ranks < args.nodes:
            parser.error("Ranks {} must be divisible by nodes {} with <=16 ranks/node".format(ranks, args.nodes))
    if args.repetitions <= 0 or args.warmups < 0:
        parser.error("Expected positive repetitions and nonnegative warmups")
    if args.resume_from and args.output.resolve() == args.resume_from.resolve():
        parser.error("--output must differ from --resume-from to preserve original results")
    if args.output.exists():
        parser.error("Output already exists (refusing to overwrite): {}".format(args.output))

    prior = []
    seen = set()
    if args.resume_from:
        with args.resume_from.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                parser.error("Resume CSV columns do not match expected schema")
            for row in reader:
                try:
                    key = key_for(row)
                    if (int(row["nodes"]) != args.nodes or
                            int(row["ranks"]) not in args.ranks or
                            row["dataset"] not in DATASETS or
                            row["order"] not in ("A(XW)", "(AX)W") or
                            row["method"] not in METHODS or
                            not 1 <= key[-1] <= args.repetitions or
                            row["phase"] != "measured"):
                        parser.error("Unexpected row in resume CSV: {}".format(key))
                except (ValueError, KeyError) as exc:
                    parser.error("Invalid resume row: {}".format(exc))
                if key in seen:
                    parser.error("Duplicate measurement in resume file: {}".format(key))
                seen.add(key)
                prior.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.logdir.mkdir(parents=True, exist_ok=True)
    outfile = args.logdir / "scratch-output.dense"
    added = 0
    print("Loaded {} existing measurements; looking for missing repetitions".format(len(prior)), flush=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(prior)
        handle.flush()
        for dataset, csr in DATASETS.items():
            for order_key, (x, w, expected) in ORDERS.items():
                for ranks in args.ranks:
                    per_node = ranks // args.nodes
                    needed = any((dataset, expected, ranks, m, rep) not in seen
                                 for m in METHODS for rep in range(1, args.repetitions + 1))
                    if not needed:
                        print("SKIP completed: {} {} ranks={}".format(dataset, expected, ranks), flush=True)
                        continue
                    for repetition in range(-args.warmups, args.repetitions):
                        shift = (repetition + args.warmups) % 3
                        for method in METHODS[shift:] + METHODS[:shift]:
                            key = (dataset, expected, ranks, method, repetition + 1)
                            if repetition >= 0 and key in seen:
                                continue
                            executable = os.path.join(
                                os.environ.get("MPI_BUILD_DIR", "."),
                                "mpi-{}-timed".format(method))
                            cmd = ["mpirun", "-np", str(ranks), "--bind-to", "core",
                                   "--map-by", "ppr:{}:node".format(per_node),
                                   executable, csr, x, w, str(outfile)]
                            start = time.perf_counter()
                            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE,
                                                    universal_newlines=True)
                            wall = time.perf_counter() - start
                            if result.returncode:
                                raise RuntimeError("Command failed: {}\n{}\n{}".format(
                                    " ".join(cmd), result.stdout, result.stderr))
                            if not re.search(r"^Order=" + re.escape(expected) + r"\s", result.stdout, re.M):
                                raise RuntimeError("Unexpected multiplication order:\n" + result.stdout[:900])
                            if repetition < 0:
                                continue
                            row = parse_output(result.stdout)
                            row.update(job_id=os.environ.get("SLURM_JOB_ID", ""),
                                       nodes=args.nodes, ranks=ranks, ranks_per_node=per_node,
                                       dataset=dataset, order=expected, method=method,
                                       repetition=repetition + 1, phase="measured", wall_s=wall)
                            writer.writerow(row)
                            handle.flush()
                            seen.add(key)
                            added += 1
                            print("{} {} ranks={} nodes={} {} rep={} inclusive={:.3f}ms".format(
                                dataset, expected, ranks, args.nodes, method, repetition + 1,
                                1000 * row["mpi_setup_inclusive_s"]), flush=True)
    expected_total = len(DATASETS) * len(ORDERS) * len(args.ranks) * len(METHODS) * args.repetitions
    print("PASS: {} previous + {} new = {} of {} measurements".format(
        len(prior), added, len(seen), expected_total), flush=True)
    if len(seen) != expected_total:
        raise RuntimeError("Incomplete measurements: expected {}, got {}".format(expected_total, len(seen)))


if __name__ == "__main__":
    main()
