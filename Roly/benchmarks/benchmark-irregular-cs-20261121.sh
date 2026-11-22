#!/bin/bash

set -euo pipefail

cd /home/rv58/COP5522/project1/fileio-target-tests

WORKDIR="/tmp/project1-irregular-$$"
mkdir -p "$WORKDIR"

cleanup()
{
    rm -rf "$WORKDIR"
}

trap cleanup EXIT

export OPENBLAS_CORETYPE=SkylakeX
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

echo "===== SYSTEM ====="
hostname
gcc --version | head -1
lscpu | grep -E 'Model name|Socket|Core'

echo
echo "===== VERIFY SOURCES ====="

check_hash()
{
    file="$1"
    expected="$2"

    actual="$(
        sha256sum "$file" |
        awk '{print $1}'
    )"

    echo "$actual  $file"

    if [ "$actual" != "$expected" ]; then
        echo "ERROR: hash mismatch for $file"
        exit 1
    fi
}

check_hash \
    project1_fileio_portable_v15.c \
    a917e22cce0ae405ebfcad35a4fd1edd8e51337ad7e71222aad54a770c9bd536

check_hash \
    project1_fileio_blas_v18.c \
    46ba9344cd026ec3b10bfb767461cbff96fc7550b152bfa96e907e2217e27361

check_hash \
    project1_fileio_blas_reassociated_noinline_v20.c \
    33d4dc6cab032946b9626c3971d1e57d2f331c4b53568745c2e30959ad038b7d

echo
echo "===== VERIFY MATRICES ====="

check_hash \
    A-benchmark.csr \
    f5d6f60abd92e084880a48622991db445011f5c0fb95917df65452ce1df3b617

check_hash \
    A-benchmark-irregular-a060.csr \
    6f764bc27adce2495c54d7920f969e8345f15634d816000f2272ff52c1e5bd04

check_hash \
    A-benchmark-irregular-a075.csr \
    5c538be78eeeefd1c23e9ecd1fdf284471eed06574b47710694b6fcaa3f355d9

echo
echo "===== MATRIX DEGREE STATISTICS ====="

python3 - <<'PY'
from pathlib import Path
import statistics

matrices = (
    ("regular", "A-benchmark.csr"),
    ("a060", "A-benchmark-irregular-a060.csr"),
    ("a075", "A-benchmark-irregular-a075.csr"),
)

for name, filename in matrices:
    tokens = Path(filename).read_text().split()

    rows = int(tokens[0])
    cols = int(tokens[1])
    nnz = int(tokens[2])

    rowptr = [
        int(x)
        for x in tokens[3:3 + rows + 1]
    ]

    degrees = [
        rowptr[i + 1] - rowptr[i]
        for i in range(rows)
    ]

    print(
        "{:<7s} mean={:6.2f} median={:6.2f} "
        "min={:4d} max={:4d} stdev={:8.3f}".format(
            name,
            statistics.mean(degrees),
            statistics.median(degrees),
            min(degrees),
            max(degrees),
            statistics.pstdev(degrees)
        )
    )
PY

echo
echo "===== COMPILE ====="

gcc -O3 -Wall -Wextra -Wpedantic -std=c99 \
    project1_fileio_portable_v15.c \
    -o "$WORKDIR/v15"

gcc -O3 -Wall -Wextra -Wpedantic -std=c99 \
    project1_fileio_blas_v18.c \
    -lopenblas \
    -o "$WORKDIR/v18"

gcc -O3 -Wall -Wextra -Wpedantic -std=c99 \
    project1_fileio_blas_reassociated_noinline_v20.c \
    -lopenblas \
    -o "$WORKDIR/v20"

echo "PASS: compiled v15, v18, v20"

echo
echo "===== V20 NOINLINE SYMBOL ====="

nm -S "$WORKDIR/v20" |
    grep csr_dense_multiply

run_case()
{
    version="$1"
    matrix="$2"
    output="$3"

    case "$matrix" in
        regular)
            afile="A-benchmark.csr"
            ;;
        a060)
            afile="A-benchmark-irregular-a060.csr"
            ;;
        a075)
            afile="A-benchmark-irregular-a075.csr"
            ;;
        *)
            echo "ERROR: unknown matrix $matrix"
            return 1
            ;;
    esac

    taskset -c 0 \
        "$WORKDIR/$version" \
        "$afile" \
        X-benchmark.dense \
        W-benchmark.dense \
        "$output"
}

echo
echo "===== CORRECTNESS ====="

for matrix in regular a060 a075
do
    echo "--- $matrix ---"

    for version in v15 v18 v20
    do
        run_case \
            "$version" \
            "$matrix" \
            "$WORKDIR/Y-${matrix}-${version}.dense" \
            >/dev/null
    done

    python3 - "$WORKDIR" "$matrix" <<'PY'
from pathlib import Path
import sys

workdir = Path(sys.argv[1])
matrix = sys.argv[2]

ABS_TOL = 1.0e-4
REL_TOL = 1.0e-4

def read_dense(path):
    tokens = path.read_text().split()
    rows = int(tokens[0])
    cols = int(tokens[1])
    values = [float(x) for x in tokens[2:]]
    return rows, cols, values

rr, rc, reference = read_dense(
    workdir / ("Y-" + matrix + "-v15.dense")
)

for version in ("v18", "v20"):
    rows, cols, values = read_dense(
        workdir / ("Y-" + matrix + "-" + version + ".dense")
    )

    if (rows, cols) != (rr, rc):
        raise SystemExit(
            "{} {} dimension mismatch".format(
                matrix,
                version
            )
        )

    max_abs = 0.0
    max_rel = 0.0
    failures = 0

    for expected, actual in zip(reference, values):
        abs_error = abs(actual - expected)

        scale = max(
            abs(expected),
            abs(actual),
            1.0e-30
        )

        rel_error = abs_error / scale

        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, rel_error)

        if abs_error > ABS_TOL and rel_error > REL_TOL:
            failures += 1

    print(
        "{} {}: max_abs={:.9g} max_rel={:.9g} failures={}".format(
            matrix,
            version,
            max_abs,
            max_rel,
            failures
        )
    )

    if failures:
        raise SystemExit(
            "ERROR: correctness failure"
        )
PY
done

echo "PASS: all correctness checks completed"

echo
echo "===== THREE WARMUPS PER CASE ====="

for matrix in regular a060 a075
do
    for version in v15 v18 v20
    do
        echo "--- $matrix / $version ---"

        for warmup in 1 2 3
        do
            run_case \
                "$version" \
                "$matrix" \
                "$WORKDIR/Y-temp.dense" \
                >/dev/null
        done
    done
done

RESULTS="$WORKDIR/results.tsv"

printf "round\tmatrix\tversion\ttime_us\n" > "$RESULTS"

orders=(
    "regular:v15 regular:v18 regular:v20 a060:v15 a060:v18 a060:v20 a075:v15 a075:v18 a075:v20"
    "a060:v18 a060:v20 a075:v15 a075:v18 a075:v20 regular:v15 regular:v18 regular:v20 a060:v15"
    "a075:v20 regular:v15 a060:v20 regular:v18 a075:v15 a060:v15 regular:v20 a075:v18 a060:v18"
    "regular:v20 a075:v18 a060:v15 regular:v15 a075:v20 a060:v18 regular:v18 a075:v15 a060:v20"
    "a075:v15 a060:v18 regular:v20 a075:v20 regular:v18 a060:v15 a075:v18 regular:v15 a060:v20"
    "a060:v20 regular:v18 a075:v18 a060:v15 regular:v20 a075:v15 a060:v18 regular:v15 a075:v20"
    "a075:v20 a075:v18 a075:v15 a060:v20 a060:v18 a060:v15 regular:v20 regular:v18 regular:v15"
)

echo
echo "===== SEVEN MEASURED ROUNDS ====="

round=0

for order in "${orders[@]}"
do
    round=$((round + 1))

    echo
    echo "----- ROUND $round -----"

    for case_name in $order
    do
        matrix="${case_name%%:*}"
        version="${case_name##*:}"

        line=$(
            run_case \
                "$version" \
                "$matrix" \
                "$WORKDIR/Y-temp.dense"
        )

        echo "$matrix/$version: $line"

        time_us=$(
            printf '%s\n' "$line" |
            awk '
            {
                for (i = 1; i <= NF; i++) {
                    if ($i == "Time") {
                        print $(i + 2)
                        exit
                    }
                }
            }'
        )

        if [ -z "$time_us" ]; then
            echo "ERROR: failed to parse runtime"
            exit 1
        fi

        printf "%d\t%s\t%s\t%s\n" \
            "$round" \
            "$matrix" \
            "$version" \
            "$time_us" \
            >> "$RESULTS"
    done
done

echo
echo "===== FINAL MEDIANS ====="

python3 - "$RESULTS" <<'PY'
import csv
import statistics
import sys

matrices = ("regular", "a060", "a075")
versions = ("v15", "v18", "v20")

data = {
    (matrix, version): []
    for matrix in matrices
    for version in versions
}

with open(sys.argv[1], newline="") as f:
    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:
        key = (
            row["matrix"],
            row["version"]
        )

        data[key].append(
            float(row["time_us"])
        )

med = {
    key: statistics.median(values)
    for key, values in data.items()
}

print(
    "{:<8s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
        "Matrix",
        "v15 ms",
        "v18 ms",
        "v20 ms",
        "v20 speedup"
    )
)

for matrix in matrices:
    v15 = med[(matrix, "v15")]
    v18 = med[(matrix, "v18")]
    v20 = med[(matrix, "v20")]

    print(
        "{:<8s} {:>10.3f} {:>10.3f} {:>10.3f} {:>11.3f}x".format(
            matrix,
            v15 / 1000.0,
            v18 / 1000.0,
            v20 / 1000.0,
            v15 / v20
        )
    )

print()
print("===== IRREGULARITY EFFECT VS REGULAR =====")

for version in versions:
    regular = med[("regular", version)]

    for matrix in ("a060", "a075"):
        irregular = med[(matrix, version)]

        change = (
            irregular / regular - 1.0
        ) * 100.0

        print(
            "{} {} vs regular: {:+.2f}%".format(
                version,
                matrix,
                change
            )
        )

print()
print("===== V20 VS V18 =====")

for matrix in matrices:
    v18 = med[(matrix, "v18")]
    v20 = med[(matrix, "v20")]

    if v20 < v18:
        print(
            "{}: v20 {:.3f}x faster, {:.2f}% lower runtime".format(
                matrix,
                v18 / v20,
                (1.0 - v20 / v18) * 100.0
            )
        )
    else:
        print(
            "{}: v18 {:.3f}x faster, {:.2f}% lower runtime".format(
                matrix,
                v20 / v18,
                (1.0 - v18 / v20) * 100.0
            )
        )
PY

echo
echo "===== COMPLETE ====="
