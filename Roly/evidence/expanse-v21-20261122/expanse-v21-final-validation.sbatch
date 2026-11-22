#!/bin/bash
#SBATCH --job-name=p1-v21-final
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --constraint=lustre
#SBATCH --output=expanse-v21-final-%j.out
#SBATCH --error=expanse-v21-final-%j.err
#SBATCH --no-requeue

set -euo pipefail

SUBMIT_DIR="$SLURM_SUBMIT_DIR"
WORKDIR="/scratch/$USER/job_${SLURM_JOB_ID}/project1-v21-final"
RESULTS="$WORKDIR/results.csv"

mkdir -p "$WORKDIR"

echo "===== JOB ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

echo
echo "===== SYSTEM ====="

hostname
gcc --version | head -1

lscpu | grep -E \
    'CPU\(s\):|On-line CPU|Thread|Core|Socket|Model name|NUMA node[0-9]+ CPU'

echo
echo "===== STAGE EXACT SOURCES / INPUTS ====="

cp \
    "$SUBMIT_DIR/project1_fileio_portable_v15.c" \
    "$SUBMIT_DIR/project1_fileio_blas_v18.c" \
    "$SUBMIT_DIR/project1_fileio_blas_reassociated_noinline_v20.c" \
    "$SUBMIT_DIR/project1_fileio_blas_adaptive_v21.c" \
    "$SUBMIT_DIR/A-benchmark.csr" \
    "$SUBMIT_DIR/X-benchmark.dense" \
    "$SUBMIT_DIR/W-benchmark.dense" \
    "$SUBMIT_DIR/X-benchmark-F64.dense" \
    "$SUBMIT_DIR/W-benchmark-F64-H128.dense" \
    "$WORKDIR/"

cd "$WORKDIR"

echo
echo "===== VERIFY SOURCE HASHES ====="

sha256sum -c <<'SUMS'
a917e22cce0ae405ebfcad35a4fd1edd8e51337ad7e71222aad54a770c9bd536  project1_fileio_portable_v15.c
46ba9344cd026ec3b10bfb767461cbff96fc7550b152bfa96e907e2217e27361  project1_fileio_blas_v18.c
33d4dc6cab032946b9626c3971d1e57d2f331c4b53568745c2e30959ad038b7d  project1_fileio_blas_reassociated_noinline_v20.c
d06dfafef6c9465cdfb86d0f75b7434b05614209c94ccdc6d67d27d79514769a  project1_fileio_blas_adaptive_v21.c
SUMS

echo
echo "===== VERIFY INPUT HASHES ====="

sha256sum -c <<'SUMS'
f5d6f60abd92e084880a48622991db445011f5c0fb95917df65452ce1df3b617  A-benchmark.csr
ec4b3708cc5005aec3202b4061e3038506dcd247f73acbb0578a1288d91701d6  X-benchmark.dense
95f4c7a2a69e491040238bf5703fa82071246563e0445fdec6695cd29b234bab  W-benchmark.dense
e73fa225d26e6ef53308e6aa3d01b1d0f4eb7c7aef73991116e835d724cea0b5  X-benchmark-F64.dense
03942e902f5916d3467b33502028ad3629a1bff22d2ca47cf741a51d1cf0cb70  W-benchmark-F64-H128.dense
SUMS

echo
echo "===== INPUT DIMENSIONS ====="

echo "--- primary ---"
head -1 A-benchmark.csr
head -1 X-benchmark.dense
head -1 W-benchmark.dense

echo
echo "--- reverse ---"
head -1 A-benchmark.csr
head -1 X-benchmark-F64.dense
head -1 W-benchmark-F64-H128.dense

echo
echo "===== VERIFY SERIAL OPENBLAS ====="

if [ ! -r /lib64/libopenblas.so.0 ]; then
    echo "ERROR: /lib64/libopenblas.so.0 is unavailable."
    exit 1
fi

ls -l /lib64/libopenblas.so.0

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

unset OPENBLAS_CORETYPE || true

echo
echo "===== COMPILE ====="

gcc \
    -O3 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -std=c99 \
    project1_fileio_portable_v15.c \
    -o v15

gcc \
    -O3 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -std=c99 \
    project1_fileio_blas_v18.c \
    /lib64/libopenblas.so.0 \
    -o v18

gcc \
    -O3 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -std=c99 \
    project1_fileio_blas_reassociated_noinline_v20.c \
    /lib64/libopenblas.so.0 \
    -o v20

gcc \
    -O3 \
    -Wall \
    -Wextra \
    -Wpedantic \
    -std=c99 \
    project1_fileio_blas_adaptive_v21.c \
    /lib64/libopenblas.so.0 \
    -o v21-base

gcc \
    -O3 \
    -march=native \
    -mtune=native \
    -Wall \
    -Wextra \
    -Wpedantic \
    -std=c99 \
    project1_fileio_blas_adaptive_v21.c \
    /lib64/libopenblas.so.0 \
    -o v21-native

echo "PASS: all five binaries compiled."

echo
echo "===== NATIVE TARGET ====="

gcc \
    -march=native \
    -mtune=native \
    -Q \
    --help=target 2>/dev/null |
    grep -E \
        'march=|mtune=|mavx|mfma' |
    head -40 || true

echo
echo "===== CSR SYMBOLS ====="

for version in v20 v21-base v21-native
do
    printf "%-10s " "$version"

    nm -S "$version" |
        grep csr_dense_multiply
done

echo
echo "===== OPENBLAS LINKAGE ====="

for version in v18 v20 v21-base v21-native
do
    printf "%-10s " "$version"

    ldd "$version" |
        grep -i openblas
done

run_version()
{
    local version="$1"
    local shape="$2"
    local output="$3"

    local x_file
    local w_file

    if [ "$shape" = "primary" ]; then
        x_file="X-benchmark.dense"
        w_file="W-benchmark.dense"
    else
        x_file="X-benchmark-F64.dense"
        w_file="W-benchmark-F64-H128.dense"
    fi

    srun \
        -n 1 \
        -c 1 \
        --cpu-bind=cores \
        "./$version" \
        A-benchmark.csr \
        "$x_file" \
        "$w_file" \
        "$output"
}

get_time()
{
    local version="$1"
    local shape="$2"

    run_version \
        "$version" \
        "$shape" \
        "$WORKDIR/Y-${version}-${shape}.dense" |
    sed -n \
        's/.*Time = \([^ ]*\) us.*/\1/p'
}

echo
echo "===== CORRECTNESS OUTPUTS ====="

for shape in primary reverse
do
    echo "--- $shape ---"

    for version in v15 v18 v20 v21-base v21-native
    do
        run_version \
            "$version" \
            "$shape" \
            "$WORKDIR/Y-${version}-${shape}.dense" \
            >/dev/null
    done

    sha256sum \
        "$WORKDIR"/Y-*-"$shape".dense
done

echo
echo "===== NUMERICAL CORRECTNESS VS v15 ====="

python3 - "$WORKDIR" <<'PY'
from pathlib import Path
import sys

workdir = Path(sys.argv[1])

ABS_TOL = 1.0e-4
REL_TOL = 1.0e-4

def read_dense(path):
    tokens = path.read_text().split()

    rows = int(tokens[0])
    cols = int(tokens[1])

    values = [
        float(value)
        for value in tokens[2:]
    ]

    expected = rows * cols

    if len(values) != expected:
        raise SystemExit(
            "{}: expected {} values, found {}".format(
                path,
                expected,
                len(values)
            )
        )

    return rows, cols, values


def compare(reference_path, test_path, label):
    rr, rc, reference = read_dense(reference_path)
    tr, tc, test = read_dense(test_path)

    if (rr, rc) != (tr, tc):
        raise SystemExit(
            "{}: dimension mismatch".format(label)
        )

    max_abs = 0.0
    max_rel = 0.0
    failures = 0

    for a, b in zip(reference, test):
        abs_error = abs(a - b)

        scale = max(
            abs(a),
            abs(b)
        )

        rel_error = (
            abs_error / scale
            if scale != 0.0
            else 0.0
        )

        max_abs = max(
            max_abs,
            abs_error
        )

        max_rel = max(
            max_rel,
            rel_error
        )

        if (
            abs_error > ABS_TOL
            and rel_error > REL_TOL
        ):
            failures += 1

    print(
        "{}: max_abs={:.9g} max_rel={:.9g} failures={}".format(
            label,
            max_abs,
            max_rel,
            failures
        )
    )

    if failures != 0:
        raise SystemExit(
            "ERROR: {} failed correctness.".format(label)
        )


for shape in ("primary", "reverse"):
    reference = (
        workdir /
        ("Y-v15-" + shape + ".dense")
    )

    for version in (
        "v18",
        "v20",
        "v21-base",
        "v21-native",
    ):
        compare(
            reference,
            workdir /
            ("Y-" + version + "-" + shape + ".dense"),
            "{} {} vs v15".format(
                shape,
                version
            )
        )

print("PASS: all correctness checks completed.")
PY

echo
echo "===== THREE WARMUPS PER VERSION / SHAPE ====="

for shape in primary reverse
do
    echo "--- $shape ---"

    for version in \
        v15 \
        v18 \
        v20 \
        v21-base \
        v21-native
    do
        echo "warming $shape/$version"

        for warmup in 1 2 3
        do
            run_version \
                "$version" \
                "$shape" \
                "$WORKDIR/Y-warmup.dense" \
                >/dev/null
        done
    done
done

echo
echo "===== 21 MEASURED ROUNDS ====="

echo \
    "shape,round,position,version,time_us" \
    > "$RESULTS"

run_measured()
{
    local shape="$1"
    local round="$2"
    local position="$3"
    local version="$4"

    local time_us

    time_us="$(
        get_time \
            "$version" \
            "$shape"
    )"

    if [ -z "$time_us" ]; then
        echo "ERROR: missing timing for $shape/$version"
        exit 1
    fi

    printf \
        '%s,%d,%d,%s,%s\n' \
        "$shape" \
        "$round" \
        "$position" \
        "$version" \
        "$time_us" \
        >> "$RESULTS"

    printf \
        '%-7s round=%2d pos=%d %-10s %10s us\n' \
        "$shape" \
        "$round" \
        "$position" \
        "$version" \
        "$time_us"
}

run_shape_round()
{
    local shape="$1"
    local round="$2"

    local order

    case $(( (round - 1) % 5 )) in
        0)
            order="v15 v18 v20 v21-base v21-native"
            ;;
        1)
            order="v18 v20 v21-base v21-native v15"
            ;;
        2)
            order="v20 v21-base v21-native v15 v18"
            ;;
        3)
            order="v21-base v21-native v15 v18 v20"
            ;;
        4)
            order="v21-native v15 v18 v20 v21-base"
            ;;
    esac

    local position=0
    local version

    for version in $order
    do
        position=$((position + 1))

        run_measured \
            "$shape" \
            "$round" \
            "$position" \
            "$version"
    done
}

for round in $(seq 1 21)
do
    echo
    echo "----- ROUND $round -----"

    if [ $((round % 2)) -eq 1 ]; then
        run_shape_round primary "$round"
        run_shape_round reverse "$round"
    else
        run_shape_round reverse "$round"
        run_shape_round primary "$round"
    fi
done

echo
echo "===== MEDIAN / PAIRED ANALYSIS ====="

python3 - "$RESULTS" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]

data = defaultdict(
    lambda: defaultdict(dict)
)

with open(path, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        shape = row["shape"]
        version = row["version"]
        round_no = int(row["round"])
        time_us = float(row["time_us"])

        data[shape][version][round_no] = time_us


def med(shape, version):
    return statistics.median(
        data[shape][version].values()
    )


def compare(shape, baseline, candidate):
    baseline_data = data[shape][baseline]
    candidate_data = data[shape][candidate]

    rounds = sorted(
        set(baseline_data) &
        set(candidate_data)
    )

    paired = [
        baseline_data[round_no] /
        candidate_data[round_no]
        for round_no in rounds
    ]

    wins = sum(
        candidate_data[round_no] <
        baseline_data[round_no]
        for round_no in rounds
    )

    losses = sum(
        candidate_data[round_no] >
        baseline_data[round_no]
        for round_no in rounds
    )

    baseline_median = med(
        shape,
        baseline
    )

    candidate_median = med(
        shape,
        candidate
    )

    print(
        "{} -> {}: {:.4f}x, "
        "candidate runtime change {:+.2f}%, "
        "paired median {:.4f}x, "
        "wins/losses {}/{}".format(
            baseline,
            candidate,
            baseline_median /
            candidate_median,
            (
                candidate_median /
                baseline_median -
                1.0
            ) * 100.0,
            statistics.median(paired),
            wins,
            losses
        )
    )


for shape in ("primary", "reverse"):
    print()
    print(
        "===== {} =====".format(
            shape.upper()
        )
    )

    for version in (
        "v15",
        "v18",
        "v20",
        "v21-base",
        "v21-native",
    ):
        print(
            "{:<10s} median = {:9.3f} us".format(
                version,
                med(shape, version)
            )
        )

    print()

    print("Adaptive comparison:")

    if shape == "primary":
        compare(
            shape,
            "v20",
            "v21-base"
        )
    else:
        compare(
            shape,
            "v18",
            "v21-base"
        )

    print()
    print("Native compiler comparison:")

    compare(
        shape,
        "v21-base",
        "v21-native"
    )

    print()
    print("Final speedup vs portable baseline:")

    compare(
        shape,
        "v15",
        "v21-native"
    )

print()
print("===== EXPECTED ADAPTIVE PATHS =====")
print("primary F=128 H=64  -> A(XW)")
print("reverse F=64 H=128 -> (AX)W")
PY

echo
echo "===== PRESERVE RAW RESULTS ====="

cp \
    "$RESULTS" \
    "$SUBMIT_DIR/expanse-v21-final-${SLURM_JOB_ID}.csv"

echo
echo "===== FINAL SOURCE HASH ====="

sha256sum \
    project1_fileio_blas_adaptive_v21.c

echo
echo "===== COMPLETE ====="
