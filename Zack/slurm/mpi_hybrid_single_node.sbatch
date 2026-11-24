#!/bin/bash
#SBATCH --job-name=mpi-hybrid
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mpi-hybrid-%j.out
#SBATCH --error=mpi-hybrid-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cpu/0.17.3b
module load gcc/10.2.0/npcyll4
module load openmpi/4.1.1
module load openblas/0.3.18
module load slurm/expanse/23.02.7
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset OPENBLAS_CORETYPE || true
unset HYBRID_ROW_WEIGHT || true

for f in mpi_hybrid.c mpi_hybrid_benchmark_driver.py mpi-adaptive mpi-balanced \
         sequential-v21 validate_large_outputs.py generate_large_inputs.py \
         generate_reverse_inputs.py large-inputs/SHA256SUMS.txt \
         reverse-inputs/X.dense reverse-inputs/W.dense; do
    test -e "$f" || { echo "Missing $f" >&2; exit 1; }
done
mkdir -p expanse-benchmarks
python3 -m py_compile mpi_hybrid_benchmark_driver.py
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py --verify

# Exactly the same OpenBLAS library already used by the earlier MPI binaries.
OPENBLAS_SO=$(ldd ./mpi-balanced | awk '/libopenblas.so.0/ {print $3; exit}')
if [ -z "$OPENBLAS_SO" ] || [ ! -f "$OPENBLAS_SO" ]; then
    echo "Cannot resolve libopenblas.so.0 from mpi-balanced" >&2; exit 1
fi
OPENBLAS_LIB=$(dirname "$OPENBLAS_SO")
mpicc -O3 -Wall -Wextra -std=c11 -o mpi-hybrid mpi_hybrid.c \
    -L"$OPENBLAS_LIB" -Wl,-rpath,"$OPENBLAS_LIB" -lopenblas -lm
ldd ./mpi-hybrid | grep libopenblas
sha256sum mpi_adaptive.c mpi_balanced.c mpi_hybrid.c \
    mpi_hybrid_benchmark_driver.py > "expanse-benchmarks/hybrid-sources-${SLURM_JOB_ID}.sha256"

# Numerical correctness before any benchmarking: two irregularities, all three
# methods, 1/2/4/16 ranks. Preserve validation logs, remove only dense scratch.
validation="expanse-benchmarks/hybrid-validation-${SLURM_JOB_ID}"
mkdir -p "$validation"
for dataset in regular clustered-strong; do
    csr="large-inputs/${dataset}-A.csr"
    ./sequential-v21 "$csr" reverse-inputs/X.dense reverse-inputs/W.dense \
        "$validation/reference.dense"
    for p in 1 2 4 16; do
        for method in adaptive balanced hybrid; do
            echo "VALIDATE dataset=$dataset method=$method ranks=$p"
            mpirun --bind-to core --map-by core -np "$p" "./mpi-$method" \
                "$csr" reverse-inputs/X.dense reverse-inputs/W.dense \
                "$validation/candidate.dense" \
                > "$validation/${dataset}-${method}-p${p}.log"
            grep -q '^Order=(AX)W ' "$validation/${dataset}-${method}-p${p}.log" || {
                echo "Incorrect multiplication order: $method p=$p" >&2; exit 1;
            }
            python3 validate_large_outputs.py \
                "$validation/reference.dense" "$validation/candidate.dense"
        done
    done
done
rm -f "$validation/reference.dense" "$validation/candidate.dense"

# 5 datasets x 5 ranks x 3 methods x 10 reps = 750 recorded measurements.
python3 mpi_hybrid_benchmark_driver.py \
    --warmups 2 --repetitions 10 --ranks 1 2 4 8 16 \
    --output "expanse-benchmarks/mpi-hybrid-single-${SLURM_JOB_ID}.csv" \
    --logdir "expanse-benchmarks/hybrid-logs-${SLURM_JOB_ID}"
echo "PASS: 750 measured runs completed"
