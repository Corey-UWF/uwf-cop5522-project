#!/usr/bin/env python3
"""Benchmark the v21-derived OpenMP sparse-row prototype on matched inputs."""

import argparse
import json
import os
import platform
import re
import statistics
from pathlib import Path

from benchmark_mpi import ROOT, compare, make_inputs, run, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blas-library", type=Path, required=True,
                        help="absolute shared-library path, e.g. /lib64/libopenblas.so.0")
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5522)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--schedules", nargs="+",
                        choices=("static", "dynamic", "guided"),
                        default=["static", "dynamic", "guided"])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if (args.n < 32 or args.features < 1 or args.hidden < 1 or
            args.runs < 1 or not args.threads or min(args.threads) < 1 or
            1 not in args.threads or len(set(args.threads)) != len(args.threads) or
            len(set(args.schedules)) != len(args.schedules)):
        parser.error("require N >= 32, positive dimensions/runs/threads and thread 1")
    if not args.blas_library.is_absolute() or not args.blas_library.is_file():
        parser.error("--blas-library must be an existing absolute file")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, details = make_inputs(output, args.n, args.features,
                                   args.hidden, args.seed)
    seq_src = ROOT / "Roly/src/project1_fileio_blas_adaptive_v21.c"
    omp_src = ROOT / "Kieren/project_openmp_v21.c"
    seq_bin, omp_bin = output / "v21", output / "openmp-v21"
    run(["gcc", "-O3", "-std=c99", str(seq_src),
         str(args.blas_library), "-o", str(seq_bin)])
    run(["gcc", "-O3", "-fopenmp", "-std=c99", str(omp_src),
         str(args.blas_library), "-o", str(omp_bin)])
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    raw, correctness = [], {}
    times = {workload: {schedule: {str(t): [] for t in args.threads}
                        for schedule in args.schedules} for workload in inputs}
    for workload, files in inputs.items():
        reference = output / f"reference-{workload}.dense"
        env["OMP_NUM_THREADS"] = "1"
        run([str(seq_bin), *map(str, files), str(reference)], env=env)
        combinations = [(schedule, t) for schedule in args.schedules
                        for t in args.threads]
        for iteration in range(args.runs + 1):
            order = (combinations[iteration % len(combinations):] +
                     combinations[:iteration % len(combinations)])
            for schedule, threads in order:
                env["OMP_NUM_THREADS"] = str(threads)
                env["OMP_SCHEDULE"] = schedule
                destination = output / f"output-{workload}-{schedule}-{threads}.dense"
                stdout = run([str(omp_bin), *map(str, files), str(destination)],
                             env=env)
                match = re.search(r"Time = ([\d.eE+-]+) us", stdout)
                if not match:
                    raise ValueError(f"Kernel timer absent: {stdout}")
                seconds = float(match.group(1)) * 1e-6
                phases = re.search(r"Dense = ([\d.eE+-]+) us\s+Sparse = ([\d.eE+-]+) us",
                                   stdout)
                if not phases:
                    raise ValueError(f"Stage timers absent: {stdout}")
                if iteration == 0:
                    result = compare(reference, destination)
                    correctness[f"{workload}_{schedule}_{threads}"] = result
                    if result["failures"]:
                        raise ValueError(f"Numerical mismatch: {workload} {schedule} "
                                         f"{threads}: {result}")
                else:
                    times[workload][schedule][str(threads)].append(seconds)
                raw.append({"workload": workload, "schedule": schedule,
                            "threads": threads, "round": iteration,
                            "warmup": iteration == 0, "seconds": seconds,
                            "dense_seconds": float(phases.group(1)) * 1e-6,
                            "sparse_seconds": float(phases.group(2)) * 1e-6})
    summary = {}
    for workload, schedules in times.items():
        summary[workload] = {}
        for schedule, thread_counts in schedules.items():
            base = statistics.median(thread_counts["1"])
            summary[workload][schedule] = {}
            for thread_count, durations in thread_counts.items():
                median = statistics.median(durations)
                speedup = base / median
                summary[workload][schedule][thread_count] = {
                    "median_seconds": median,
                    "median_dense_seconds": statistics.median(
                        row["dense_seconds"] for row in raw
                        if row["workload"] == workload and row["schedule"] == schedule
                        and row["threads"] == int(thread_count) and not row["warmup"]),
                    "median_sparse_seconds": statistics.median(
                        row["sparse_seconds"] for row in raw
                        if row["workload"] == workload and row["schedule"] == schedule
                        and row["threads"] == int(thread_count) and not row["warmup"]),
                    "speedup_vs_one_thread_same_schedule": speedup,
                    "parallel_efficiency": speedup / int(thread_count)}
    report = {
        "input": details,
        "input_sha256": {p.name: sha256(p) for paths in inputs.values() for p in paths},
        "source_sha256": {"v21": sha256(seq_src), "openmp_v21": sha256(omp_src)},
        "machine": {"hostname": platform.node(), "platform": platform.platform(),
                    "compiler": run(["gcc", "--version"]).splitlines()[0],
                    "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
                    "scheduler_node_list": os.environ.get("SLURM_JOB_NODELIST")},
        "protocol": {"runs": args.runs, "warmups": 1,
                     "threads": args.threads, "schedules": args.schedules,
                     "blas_library": str(args.blas_library),
                     "openblas_threads": 1,
                     "timing": "dense serial BLAS plus OpenMP sparse rows; file I/O excluded",
                     "speedup_baseline": "one-thread median of each schedule"},
        "correctness": correctness, "raw_runs": raw, "summary": summary}
    (output / "openmp-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"input": details, "summary": summary,
                      "correctness_failures": sum(v["failures"]
                                                  for v in correctness.values())},
                     indent=2))


if __name__ == "__main__":
    main()
