"""Checks mpi-nodeshared against sequential v21.

Covers both multiplication orders, empty rows, a dominant row, more ranks than
rows, zero nonzeros, odd rank counts, and emulated multi-node runs. The
partition the program reports is also compared with an independent calculation.

Example:
  python3 tests/test_nodeshared.py --executable ./mpi-nodeshared \
      --reference ./sequential-v21 --mpiexec "mpirun --oversubscribe"
"""
import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_balanced_mpi import compare, write_inputs  # noqa: E402

POWER_LAW = [0 if i % 11 == 10 else max(1, 64 // ((i + 1) ** 2)) for i in range(64)]
# (name, nonzeros in each row, features F, hidden H)
CASES = [
    ("regular A(XW)", [16] * 300, 128, 64),
    ("regular (AX)W", [16] * 300, 64, 128),
    ("F equals H", [3, 1, 4, 1, 5, 9, 2, 6], 5, 5),
    ("empty rows", [0, 0, 4, 0, 4, 0], 7, 3),
    ("dominant row", [1, 128, 0, 1], 3, 7),
    ("more ranks than rows", [1, 0, 2], 4, 4),
    ("single row", [5], 3, 2),
    ("zero nonzeros", [0] * 5, 3, 7),
    ("block tail", [0, 1, 2, 1, 0], 131, 130),
    ("power law A(XW)", POWER_LAW, 7, 3),
    ("power law (AX)W", POWER_LAW, 3, 7),
]


def expected_bounds(pointers, workers, use_xw_first, hidden):
    n = len(pointers) - 1
    row_cost = 1 + (0 if use_xw_first else hidden // 2)
    total = pointers[-1] + row_cost * n
    bounds, row = [0], 0
    for w in range(1, workers + 1):
        while row < n and (pointers[row] + row_cost * row) * workers < total * w:
            row += 1
        bounds.append(row)
    return bounds


def run(command, pretend_nodes=0):
    env = dict(os.environ)
    env.pop("NODESHARED_TEST_NODES", None)
    if pretend_nodes:
        env["NODESHARED_TEST_NODES"] = str(pretend_nodes)
    return subprocess.run(command, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True, timeout=180)


def check_report(output, pointers, ranks, nodes, features, hidden):
    use_xw_first = hidden <= features
    assert "Order=" + ("A(XW)" if use_xw_first else "(AX)W") in output, output
    assert "Ranks=%d Nodes=%d OpenBLAS_threads=1" % (ranks, nodes) in output, output
    bounds = expected_bounds(pointers, ranks, use_xw_first, hidden)
    rows = [b - a for a, b in zip(bounds, bounds[1:])]
    counts = [pointers[b] - pointers[a] for a, b in zip(bounds, bounds[1:])]
    assert "Rows per rank: min=%d max=%d" % (min(rows), max(rows)) in output, output
    assert "Nonzeros per rank: min=%d max=%d" % (min(counts), max(counts)) in output, output
    seconds = [float(v) for v in re.findall(r"= ([0-9.]+) seconds", output)]
    assert len(seconds) >= 9 and all(math.isfinite(v) and v >= 0 for v in seconds), output
    phases, total, inclusive = seconds[:5], seconds[5], seconds[6]
    assert total >= max(phases) * 0.999, output
    assert inclusive >= total, output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--executable", default="./mpi-nodeshared")
    parser.add_argument("--reference", required=True, help="sequential v21 executable")
    parser.add_argument("--mpiexec", default="mpirun", help="launcher command, may include options")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 7, 16])
    parser.add_argument("--nodes", type=int, nargs="+", default=[0, 2, 3],
                        help="0 = real node layout; k > 0 pretends there are k nodes")
    args = parser.parse_args()
    launcher = shlex.split(args.mpiexec)
    executable = str(Path(args.executable).resolve())
    reference = str(Path(args.reference).resolve())

    passed = 0
    with tempfile.TemporaryDirectory(prefix="nodeshared-tests-") as temp:
        directory = Path(temp)
        for name, degrees, features, hidden in CASES:
            a, x, w, pointers = write_inputs(directory, degrees, features, hidden)
            worst = 0.0
            expected = directory / "expected.dense"
            result = run([reference, str(a), str(x), str(w), str(expected)])
            assert result.returncode == 0, result.stderr
            for pretend in args.nodes:
                for ranks in args.ranks:
                    actual = directory / "actual.dense"
                    result = run(launcher + ["-n", str(ranks), executable, str(a), str(x),
                                             str(w), str(actual)], pretend)
                    assert result.returncode == 0, "%s ranks=%d: %s" % (name, ranks, result.stderr)
                    nodes = min(pretend, ranks) if pretend else 1
                    worst = max(worst, compare(expected, actual))
                    check_report(result.stdout, pointers, ranks, nodes, features, hidden)
                    passed += 1
            print("PASS %-22s max abs error vs v21 = %.3g" % (name, worst))

        # Bad input must stop every rank with an error, not hang.
        a, x, w, _ = write_inputs(directory, [2, 2, 2], 3, 2)
        a.write_text(a.read_text().replace("0 2 4 6", "0 4 2 6", 1))
        result = run(launcher + ["-n", "3", executable, str(a), str(x), str(w),
                                 str(directory / "bad.dense")])
        assert result.returncode != 0, "bad input was accepted"
        assert not (directory / "bad.dense").exists(), "output written for bad input"
        print("PASS bad CSR input is rejected")
    print("ALL PASSED: %d program runs matched sequential v21" % passed)


if __name__ == "__main__":
    main()
