#!/usr/bin/env python3
"""Times MPI variants of Y = A X W on identical inputs and prints median phase times.

Every variant is first checked against the sequential reference, then timed with
warm-ups and repetitions (variant order alternates to avoid ordering bias).
Compatible with Python 3.6.

Example (inside an allocation):
  python3 scripts/compare_variants.py --reference ./sequential-v21 \
      --variant hybrid=./mpi-hybrid-timed --variant nodeshared=./mpi-nodeshared \
      --dataset regular large-inputs/regular-A.csr large-inputs/X.dense large-inputs/W.dense \
      --ranks 1 4 16 --mpiexec "mpirun --bind-to core"
"""
import argparse
import csv
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

PHASES = [("distribution", "CSR distribution"), ("xw", "Root XW"),
          ("broadcast", "Dense broadcast"), ("compute", "Local computation"),
          ("gather", "Output gathering"), ("elapsed", "MPI elapsed time")]
HERE = Path(__file__).resolve().parent
# Works from the repository layout (tests/ next to scripts/) and from a flat directory.
VALIDATOR = next((p for p in (HERE.parent / "tests" / "validate_large_outputs.py",
                              HERE / "validate_large_outputs.py") if p.is_file()),
                 HERE / "validate_large_outputs.py")


def run(command):
    start = time.perf_counter()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode != 0:
        sys.exit("Command failed: %s\n%s\n%s" % (" ".join(command), result.stdout, result.stderr))
    return result.stdout, time.perf_counter() - start


def parse(output):
    values = {}
    for key, label in PHASES:
        match = re.search(r"^%s\s*=\s*([0-9.eE+-]+) seconds" % re.escape(label), output, re.M)
        if match is None:
            sys.exit("Missing '%s' in output:\n%s" % (label, output[-1500:]))
        values[key] = float(match.group(1)) * 1000.0
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="sequential v21 executable")
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=EXE")
    parser.add_argument("--dataset", action="append", nargs=4, required=True,
                        metavar=("NAME", "A.csr", "X.dense", "W.dense"))
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, help="write every measurement to this file")
    args = parser.parse_args()
    variants = [v.split("=", 1) for v in args.variant]
    launcher = shlex.split(args.mpiexec)
    args.workdir.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, a, x, w in args.dataset:
        expected = str(args.workdir / "expected.dense")
        scratch = str(args.workdir / "scratch.dense")
        seq_times = []
        for _ in range(args.repetitions):
            out, _wall = run([args.reference, a, x, w, expected])
            seq_times.append(float(re.search(r"Time = ([0-9.eE+-]+) us", out).group(1)) / 1000.0)
        seq = statistics.median(seq_times)
        print("\n== %s: sequential v21 median (compute only) = %.3f ms" % (name, seq), flush=True)
        print("%5s %-11s %9s %8s %8s %8s %8s %8s %8s %8s" % (
            "ranks", "variant", "elapsed", "distrib", "rootXW", "bcast", "compute", "gather",
            "wall(s)", "vs seq"), flush=True)
        for ranks in args.ranks:
            def command(exe):
                return launcher + ["-n", str(ranks), exe, a, x, w, scratch]
            for vname, exe in variants:                  # correctness first
                out, _wall = run(command(exe))
                run([sys.executable, str(VALIDATOR), expected, scratch])
            samples = dict((vname, []) for vname, _ in variants)
            for rep in range(-args.warmups, args.repetitions):
                for vname, exe in (variants if rep % 2 == 0 else variants[::-1]):
                    out, wall = run(command(exe))
                    if rep >= 0:
                        sample = parse(out)
                        sample["wall"] = wall
                        samples[vname].append(sample)
                        rows.append(dict(dataset=name, ranks=ranks, variant=vname, **sample))
            for vname, _ in variants:
                med = dict((k, statistics.median(s[k] for s in samples[vname]))
                           for k in [p[0] for p in PHASES] + ["wall"])
                print("%5d %-11s %9.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.2f %7.2fx" % (
                    ranks, vname, med["elapsed"], med["distribution"], med["xw"],
                    med["broadcast"], med["compute"], med["gather"], med["wall"],
                    seq / med["elapsed"]), flush=True)
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
