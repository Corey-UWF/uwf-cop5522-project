#!/bin/bash
#SBATCH --job-name=p1-reverse
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=expanse-reverse-%j.out
#SBATCH --error=expanse-reverse-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

WORKDIR="${TMPDIR:-/tmp}/project1-reverse-${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

echo "===== JOB ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

echo
echo "===== SYSTEM ====="
hostname
gcc --version | head -1
lscpu | grep -E 'Model name|Socket|Core'

echo
echo "===== VERIFY SOURCES ====="

sha256sum \
    project1_fileio_blas_v18.c \
    project1_fileio_blas_reassociated_noinline_v20.c

echo
echo "===== VERIFY INPUTS ====="

head -1 X-benchmark-F64.dense
head -1 W-benchmark-F64-H128.dense

sha256sum \
    X-benchmark-F64.dense \
    W-benchmark-F64-H128.dense

echo
echo "===== COMPILE ====="

gcc -O3 -Wall -Wextra -Wpedantic -std=c99 \
    project1_fileio_blas_v18.c \
    /lib64/libopenblas.so.0 \
    -o "$WORKDIR/v18"

gcc -O3 -Wall -Wextra -Wpedantic -std=c99 \
    project1_fileio_blas_reassociated_noinline_v20.c \
    /lib64/libopenblas.so.0 \
    -o "$WORKDIR/v20"

echo
echo "===== VERIFY V20 NOINLINE ====="

nm -S "$WORKDIR/v20" |
    grep csr_dense_multiply

echo
echo "===== OPENBLAS ====="

for version in v18 v20
do
    printf "%-4s: " "$version"

    ldd "$WORKDIR/$version" |
        grep 'libopenblas.so.0'
done

run_version()
{
    version="$1"
    output="$2"

    srun \
        --ntasks=1 \
        --cpus-per-task=1 \
        --cpu-bind=cores \
        "$WORKDIR/$version" \
        A-benchmark.csr \
        X-benchmark-F64.dense \
        W-benchmark-F64-H128.dense \
        "$output"
}

echo
echo "===== CORRECTNESS ====="

run_version \
    v18 \
    "$WORKDIR/Y-v18.dense" \
    >/dev/null

run_version \
    v20 \
    "$WORKDIR/Y-v20.dense" \
    >/dev/null

python3 - "$WORKDIR" <<'PY'
from pathlib import Path
import sys

workdir = Path(sys.argv[1])

def read_dense(path):
    tokens = path.read_text().split()

    rows = int(tokens[0])
    cols = int(tokens[1])
    values = [float(x) for x in tokens[2:]]

    return rows, cols, values

r1, c1, a = read_dense(
    workdir / "Y-v18.dense"
)

r2, c2, b = read_dense(
    workdir / "Y-v20.dense"
)

if (r1, c1) != (r2, c2):
    raise SystemExit(
        "ERROR: output dimension mismatch"
    )

max_abs = 0.0
max_rel = 0.0
failures = 0

for expected, actual in zip(a, b):
    abs_error = abs(actual - expected)

    scale = max(
        abs(expected),
        abs(actual),
        1.0e-30
    )

    rel_error = abs_error / scale

    max_abs = max(max_abs, abs_error)
    max_rel = max(max_rel, rel_error)

    if abs_error > 1.0e-4 and rel_error > 1.0e-4:
        failures += 1

print("max_abs =", max_abs)
print("max_rel =", max_rel)
print("failures =", failures)

if failures:
    raise SystemExit(
        "FAIL: numerical comparison"
    )

print("PASS: v18 and v20 agree within tolerance")
PY

echo
echo "===== THREE WARMUPS ====="

for version in v18 v20
do
    for warmup in 1 2 3
    do
        run_version \
            "$version" \
            "$WORKDIR/Y-temp.dense" \
            >/dev/null
    done
done

RESULTS="$WORKDIR/results.tsv"

printf "round\tversion\ttime_us\n" > "$RESULTS"

orders=(
    "v18 v20"
    "v20 v18"
    "v18 v20"
    "v20 v18"
    "v18 v20"
    "v20 v18"
    "v18 v20"
)

echo
echo "===== SEVEN MEASURED ROUNDS ====="

round=0

for order in "${orders[@]}"
do
    round=$((round + 1))

    echo
    echo "----- ROUND $round: $order -----"

    for version in $order
    do
        line=$(
            run_version \
                "$version" \
                "$WORKDIR/Y-temp.dense"
        )

        echo "$version: $line"

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

        printf "%d\t%s\t%s\n" \
            "$round" \
            "$version" \
            "$time_us" \
            >> "$RESULTS"
    done
done

echo
echo "===== REVERSE-SHAPE MEDIANS ====="

python3 - "$RESULTS" <<'PY'
import csv
import statistics
import sys

data = {
    "v18": [],
    "v20": []
}

with open(sys.argv[1], newline="") as f:
    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:
        data[row["version"]].append(
            float(row["time_us"])
        )

med = {
    version: statistics.median(values)
    for version, values in data.items()
}

flops = {
    "v18": 73728000.0,
    "v20": 81920000.0,
}

for version in ("v18", "v20"):
    seconds = med[version] / 1.0e6
    gflops = flops[version] / seconds / 1.0e9

    print(
        "{}: {:.3f} us = {:.3f} ms, {:.3f} GFLOP/s".format(
            version,
            med[version],
            med[version] / 1000.0,
            gflops
        )
    )

print()
print("===== ASSOCIATION COMPARISON =====")

if med["v18"] < med["v20"]:
    print(
        "v18 faster by : {:.2f}%".format(
            (
                1.0 -
                med["v18"] / med["v20"]
            ) * 100.0
        )
    )

    print(
        "v20/v18 ratio : {:.3f}x".format(
            med["v20"] / med["v18"]
        )
    )
else:
    print(
        "v20 faster by : {:.2f}%".format(
            (
                1.0 -
                med["v20"] / med["v18"]
            ) * 100.0
        )
    )

    print(
        "v18/v20 ratio : {:.3f}x".format(
            med["v18"] / med["v20"]
        )
    )

print()
print(
    "v18 FLOPs = {}".format(
        int(flops["v18"])
    )
)

print(
    "v20 FLOPs = {}".format(
        int(flops["v20"])
    )
)
PY

echo
echo "===== COMPLETE ====="
