"""Compare balanced MPI output and diagnostics against adaptive MPI and v21."""

import argparse
import bisect
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile


def read_matrix(path):
    tokens = path.read_text().split()
    shape = tuple(map(int, tokens[:2]))
    values = list(map(float, tokens[2:]))
    if len(shape) != 2 or len(values) != math.prod(shape):
        raise AssertionError(f"Invalid output shape in {path}")
    if not all(math.isfinite(value) for value in values):
        raise AssertionError(f"Nonfinite output in {path}")
    return shape, values


def compare(expected_path, actual_path):
    shape, expected = read_matrix(expected_path)
    actual_shape, actual = read_matrix(actual_path)
    if actual_shape != shape:
        raise AssertionError(f"Shape mismatch: {shape} != {actual_shape}")
    failures = sum(abs(a - b) > 1e-4 + 1e-4 * abs(b)
                   for a, b in zip(actual, expected))
    if failures:
        raise AssertionError(f"{failures} numerical tolerance failures")
    return max(abs(a - b) for a, b in zip(actual, expected))


def write_inputs(directory, degrees, features, hidden):
    n = len(degrees)
    pointers = [0]
    columns = []
    values = []
    for row, degree in enumerate(degrees):
        pointers.append(pointers[-1] + degree)
        columns.extend((row + p * 37) % n for p in range(degree))
        values.extend(0.125 * ((p % 7) + 1) for p in range(degree))
    a = directory / "A.csr"
    x = directory / "X.dense"
    w = directory / "W.dense"
    a.write_text(f"{n} {n} {len(columns)}\n" + "\n".join(
        " ".join(map(str, sequence)) for sequence in (pointers, columns, values)) + "\n")
    x_values = [((i + 3 * j) % 17 + 1) / 17
                for i in range(n) for j in range(features)]
    w_values = [((2 * i + 5 * j) % 19 + 1) / 19
                for i in range(features) for j in range(hidden)]
    x.write_text(f"{n} {features}\n" + " ".join(f"{v:.9g}" for v in x_values))
    w.write_text(f"{features} {hidden}\n" + " ".join(f"{v:.9g}" for v in w_values))
    return a, x, w, pointers


def expected_partitions(pointers, ranks):
    n = len(pointers) - 1
    nnz = pointers[-1]
    boundaries = [0]
    for p in range(1, ranks):
        if nnz == 0:
            boundary = p * (n // ranks) + min(p, n % ranks)
        else:
            scaled = [value * ranks for value in pointers]
            target = nnz * p
            boundary = bisect.bisect_left(scaled, target)
            if boundary and target - scaled[boundary - 1] <= scaled[boundary] - target:
                boundary -= 1
        boundaries.append(boundary)
    boundaries.append(n)
    rows = [end - start for start, end in zip(boundaries, boundaries[1:])]
    counts = [pointers[end] - pointers[start]
              for start, end in zip(boundaries, boundaries[1:])]
    return rows, counts


def check_diagnostics(output, pointers, ranks, features, hidden):
    rows, counts = expected_partitions(pointers, ranks)
    n = len(pointers) - 1
    nnz = pointers[-1]
    average = nnz / ranks
    assert f"Rows={n} Nonzeros={nnz} Ranks={ranks}" in output
    assert f"Rows per rank: min={min(rows)} max={max(rows)}" in output
    assert f"Nonzeros per rank: min={min(counts)} max={max(counts)} average={average:.6f}" in output
    order = "A(XW)" if hidden <= features else "(AX)W"
    assert f"Order={order}" in output and "OpenBLAS_threads=1" in output
    if nnz == 0:
        assert "Nonzero imbalance ratio = N/A (zero nonzeros)" in output
    else:
        assert f"Nonzero imbalance ratio = {max(counts) / average:.6f}" in output
    phases = re.findall(r"^.* = ([0-9.]+) seconds$", output, re.MULTILINE)
    assert len(phases) == 6
    times = list(map(float, phases))
    assert all(math.isfinite(t) and t >= 0 for t in times)
    assert times[-1] >= max(times[:-1])
    match = re.search(r"Rank-local computation: min=([0-9.]+) max=([0-9.]+) average=([0-9.]+)", output)
    assert match
    minimum, maximum, mean = map(float, match.groups())
    assert 0 <= minimum <= mean <= maximum
    assert maximum == times[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced", default="./mpi-balanced")
    parser.add_argument("--adaptive", default="./mpi-adaptive")
    parser.add_argument("--reference", required=True, help="Compiled sequential v21 executable")
    parser.add_argument("--mpiexec", default="mpirun")
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4])
    args = parser.parse_args()
    if any(rank <= 0 for rank in args.ranks):
        parser.error("Rank counts must be positive")
    balanced, adaptive, reference = (str(Path(path).resolve())
                                     for path in (args.balanced, args.adaptive, args.reference))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")

    def run(command):
        result = subprocess.run(command, env=env, capture_output=True, text=True,
                                timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    cases = [
        ("primary", [16] * 4000, 128, 64),
        ("reverse", [16] * 4000, 64, 128),
        ("empty-rows", [0, 0, 4, 0, 4, 0], 7, 3),
        ("dominant-row", [1, 128, 0, 1], 3, 7),
        ("excess-ranks", [1, 0, 2], 4, 4),
        ("zero", [0] * 5, 3, 7),
        ("block-tail", [0, 1, 2, 1, 0], 131, 130),
    ]
    power_law = [0 if i % 11 == 10 else max(1, 64 // ((i + 1) ** 2))
                 for i in range(64)]
    cases.extend([("power-law-xw", power_law, 7, 3),
                  ("power-law-ax", power_law, 3, 7)])
    with tempfile.TemporaryDirectory(prefix="balanced-mpi-tests-") as temporary:
        directory = Path(temporary)
        for name, degrees, features, hidden in cases:
            a, x, w, pointers = write_inputs(directory, degrees, features, hidden)
            seq_output = directory / "reference.dense"
            run([reference, str(a), str(x), str(w), str(seq_output)])
            for ranks in args.ranks:
                balanced_output = directory / "balanced.dense"
                adaptive_output = directory / "adaptive.dense"
                command = [args.mpiexec, "-np", str(ranks)]
                output = run(command + [balanced, str(a), str(x), str(w), str(balanced_output)])
                run(command + [adaptive, str(a), str(x), str(w), str(adaptive_output)])
                error = compare(seq_output, balanced_output)
                compare(adaptive_output, balanced_output)
                check_diagnostics(output, pointers, ranks, features, hidden)
                print(f"PASS {name}: ranks={ranks}, maximum absolute error vs v21={error:.9g}")


if __name__ == "__main__":
    main()
