#!/bin/bash
#SBATCH --job-name=mpi-correctness
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --output=mpi-correctness-%j.out
#SBATCH --error=mpi-correctness-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load cpu/0.17.3b
module load gcc/10.2.0/npcyll4
module load openmpi/4.1.1
module load openblas/0.3.18
module load slurm/expanse/23.02.7

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
unset OPENBLAS_CORETYPE

mkdir -p expanse-validation

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "MPI compiler: $(which mpicc)"
module list

echo "=== Sequential reference ==="
./sequential-v21 \
    A-benchmark.csr \
    X-benchmark.dense \
    W-benchmark.dense \
    expanse-validation/Y-sequential.dense

for method in adaptive balanced; do
    for p in 1 2 4; do
        echo "=== $method : $p ranks ==="

        mpirun -np "$p" "./mpi-$method" \
            A-benchmark.csr \
            X-benchmark.dense \
            W-benchmark.dense \
            "expanse-validation/Y-${method}-${p}.dense"
    done
done

echo "=== Numerical correctness ==="

python3 - <<'PY'
from pathlib import Path
import math

folder = Path("expanse-validation")

def load(name):
    tokens = (folder / name).read_text().split()
    rows, cols = map(int, tokens[:2])
    values = [float(v) for v in tokens[2:]]
    assert len(values) == rows * cols
    return (rows, cols), values

shape, expected = load("Y-sequential.dense")
passed = 0

for method in ("adaptive", "balanced"):
    for p in (1, 2, 4):
        actual_shape, actual = load(f"Y-{method}-{p}.dense")

        if actual_shape != shape:
            print(method, p, "FAIL dimensions")
            continue

        failures = sum(
            not (math.isfinite(a) and math.isfinite(b))
            or abs(a-b) > 1e-4 + 1e-4 * abs(a)
            for a, b in zip(expected, actual)
        )

        if failures == 0:
            passed += 1

        print(
            f"{method} {p} ranks: "
            f"failures={failures}, "
            f"{'PASS' if failures == 0 else 'FAIL'}"
        )

print(f"TOTAL: {passed}/6 PASS")
raise SystemExit(0 if passed == 6 else 1)
PY

echo "Correctness test completed."

