#!/usr/bin/env python3
"""Generate matched CSR inputs and benchmark the existing MPI baseline.

Example:
    python3 Kieren/benchmark_mpi.py --out /tmp/axw-mpi --ranks 1 2 4
    python3 Kieren/benchmark_mpi.py --out "$PWD/axw-mpi" --launcher srun \
        --ranks 1 2 4 8 --runs 5

Run the second command inside a scheduler allocation with enough MPI tasks.
The one-rank MPI measurement is the scaling baseline. The sequential program
is used to check numerical output, not to calculate MPI parallel efficiency.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def degree_sequence(n, total, seed):
    """Deterministic rank-weighted degrees; a synthetic skew, not a graph model."""
    rng = random.Random(seed)
    active = int(n * 0.98)
    weights = [(rank + 1) ** -0.9 for rank in range(active)]
    degrees = [0] * active
    remaining = total
    available = set(range(active))
    while remaining:
        weight_sum = sum(weights[i] for i in available)
        shares = [(i, remaining * weights[i] / weight_sum) for i in available]
        gained = 0
        for i, share in shares:
            increment = min(n - degrees[i], int(share))
            degrees[i] += increment
            gained += increment
        remaining -= gained
        available = {i for i in available if degrees[i] < n}
        if not gained:
            for i, _ in sorted(shares, key=lambda pair: pair[1] % 1, reverse=True):
                if not remaining:
                    break
                if i in available:
                    degrees[i] += 1
                    remaining -= 1
    degrees += [0] * (n - active)
    rng.shuffle(degrees)
    assert sum(degrees) == total and max(degrees) <= n
    return degrees


def write_csr(path, n, degrees, rng):
    rowptr = [0]
    columns, values = [], []
    for degree in degrees:
        selected = sorted(rng.sample(range(n), degree))
        columns.extend(selected)
        values.extend(f"{rng.uniform(-1, 1) / math.sqrt(degree):.9g}"
                      for _ in selected)
        rowptr.append(len(columns))
    with path.open("w") as handle:
        handle.write(f"{n} {n} {len(columns)}\n")
        handle.write(" ".join(map(str, rowptr)) + "\n")
        handle.write(" ".join(map(str, columns)) + "\n")
        handle.write(" ".join(values) + "\n")


def make_inputs(directory, n, features, hidden, seed):
    total = n * 16
    irregular = degree_sequence(n, total, seed)
    regular = [16] * n
    a_regular = directory / "regular-A.csr"
    a_irregular = directory / "irregular-A.csr"
    write_csr(a_regular, n, regular, random.Random(seed + 1))
    write_csr(a_irregular, n, irregular, random.Random(seed + 2))
    rng = random.Random(seed + 3)
    dense = []
    for name, rows, cols, scale in (("X.dense", n, features, 1.0),
                                   ("W.dense", features, hidden, 0.1)):
        path = directory / name
        with path.open("w") as handle:
            handle.write(f"{rows} {cols}\n")
            for _ in range(rows):
                handle.write(" ".join(f"{rng.uniform(-scale, scale):.9g}"
                                      for _ in range(cols)) + "\n")
        dense.append(path)
    ordered = sorted(irregular)
    details = {
        "N": n, "F": features, "H": hidden, "nnz": total, "seed": seed,
        "regular_degree": 16, "irregular_zero_rows": irregular.count(0),
        "irregular_median_degree": statistics.median(irregular),
        "irregular_p90_degree": ordered[int(.90 * (n - 1))],
        "irregular_p99_degree": ordered[int(.99 * (n - 1))],
        "irregular_max_degree": max(irregular),
    }
    return {"regular": [a_regular, *dense],
            "irregular": [a_irregular, *dense]}, details


def read_dense(path):
    with path.open() as handle:
        shape = tuple(map(int, handle.readline().split()))
        data = [float(value) for line in handle for value in line.split()]
    if len(data) != math.prod(shape):
        raise ValueError(f"Wrong output length in {path}")
    return shape, data


def compare(reference, candidate, atol=1e-4, rtol=1e-4):
    ref_shape, ref = read_dense(reference)
    got_shape, got = read_dense(candidate)
    if ref_shape != got_shape:
        raise ValueError(f"Output shape mismatch: {ref_shape} != {got_shape}")
    differences = [abs(a - b) for a, b in zip(ref, got)]
    return {"elements": len(ref), "failures": sum(
        difference > atol + rtol * abs(a)
        for a, difference in zip(ref, differences)),
        "max_absolute_error": max(differences, default=0),
        "atol": atol, "rtol": rtol}


def run(command, **kwargs):
    process = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {command}\n"
                           f"stdout: {process.stdout}\nstderr: {process.stderr}")
    return process.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=5522)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--launcher", choices=("mpirun", "srun"), default="mpirun")
    parser.add_argument("--generate-only", action="store_true",
                        help="generate and verify inputs without compiling MPI")
    args = parser.parse_args()
    if (args.n < 32 or args.features < 1 or args.hidden < 1 or
            args.runs < 1 or not args.ranks or min(args.ranks) < 1 or
            1 not in args.ranks or len(set(args.ranks)) != len(args.ranks)):
        parser.error("require N >= 32, positive dimensions/runs/ranks, and rank 1")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs, details = make_inputs(output, args.n, args.features, args.hidden,
                                   args.seed)
    if args.generate_only:
        manifest = {"input": details, "input_sha256": {
            path.name: sha256(path) for files in inputs.values() for path in files}}
        (output / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return
    binary_seq = output / "sequential-v15"
    binary_mpi = output / "mpi-baseline"
    seq_src = ROOT / "Roly/src/project1_fileio_portable_v15.c"
    mpi_src = ROOT / "Zack/mpi_baseline.c"
    run(["gcc", "-O3", str(seq_src), "-o", str(binary_seq)])
    run(["mpicc", "-O3", "-std=c11", str(mpi_src), "-o", str(binary_mpi)])
    timings = {name: {str(p): [] for p in args.ranks} for name in inputs}
    correctness = {}
    raw = []
    environment = os.environ.copy()
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    for workload, files in inputs.items():
        reference = output / f"reference-{workload}.dense"
        run([str(binary_seq), *map(str, files), str(reference)], env=environment)
        for iteration in range(args.runs + 1):
            ranks = args.ranks[iteration % len(args.ranks):] + args.ranks[:iteration % len(args.ranks)]
            for count in ranks:
                target = output / f"output-{workload}-{count}.dense"
                prefix = (["mpirun", "-np", str(count)] if args.launcher == "mpirun"
                          else ["srun", "-n", str(count)])
                stdout = run([*prefix, str(binary_mpi), *map(str, files),
                              str(target)], env=environment)
                match = re.search(r"MPI elapsed time = ([\d.eE+-]+) seconds", stdout)
                if not match:
                    raise ValueError(f"MPI timer absent: {stdout}")
                seconds = float(match.group(1))
                if iteration == 0:
                    checked = compare(reference, target)
                    correctness[f"{workload}_{count}_ranks"] = checked
                    if checked["failures"]:
                        raise ValueError(f"Numerical mismatch: {workload} {count}: {checked}")
                else:
                    timings[workload][str(count)].append(seconds)
                raw.append({"workload": workload, "ranks": count,
                            "round": iteration, "warmup": iteration == 0,
                            "seconds": seconds})
    summary = {}
    for workload, by_rank in timings.items():
        base = statistics.median(by_rank["1"])
        summary[workload] = {}
        for count, values in by_rank.items():
            median = statistics.median(values)
            speedup = base / median
            summary[workload][count] = {"median_seconds": median,
                "speedup_vs_one_rank": speedup,
                "parallel_efficiency": speedup / int(count)}
    report = {"input": details,
              "input_sha256": {p.name: sha256(p) for case in inputs.values() for p in case},
              "source_sha256": {"v15": sha256(seq_src), "mpi": sha256(mpi_src)},
              "machine": {"hostname": platform.node(), "platform": platform.platform(),
                          "compiler": run(["gcc", "--version"]).splitlines()[0],
                          "mpi_compiler": run(["mpicc", "--version"]).splitlines()[0],
                          "scheduler_job_id": os.environ.get("SLURM_JOB_ID"),
                          "scheduler_node_list": os.environ.get("SLURM_JOB_NODELIST")},
              "protocol": {"runs": args.runs, "warmups": 1,
                           "launcher": args.launcher, "ranks": args.ranks,
                           "timer": "maximum rank elapsed; distribution, computation and gather; file I/O excluded",
                           "speedup_baseline": "one-rank MPI median"},
              "correctness": correctness, "raw_runs": raw, "summary": summary}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"input": details, "correctness": correctness,
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
