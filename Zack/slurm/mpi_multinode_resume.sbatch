#!/bin/bash
#SBATCH --job-name=mpi-resume4
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mpi-resume4-%j.out
#SBATCH --error=mpi-resume4-%j.err
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
for f in mpi_adaptive_timed.c mpi_balanced_timed.c mpi_hybrid_timed.c \
         mpi_multinode_driver_resume.py mpi-balanced sequential-v21 \
         validate_large_outputs.py generate_large_inputs.py \
         generate_reverse_inputs.py \
         expanse-benchmarks/mpi-multinode-4nodes-56010108.csv; do
    test -f "$f" || { echo "Missing file: $f" >&2; exit 1; }
done
python3 -m py_compile mpi_multinode_driver_resume.py
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py --verify
OPENBLAS_SO=$(ldd ./mpi-balanced | awk '/libopenblas.so.0/ {print $3; exit}')
test -f "$OPENBLAS_SO"
OPENBLAS_LIB=$(dirname "$OPENBLAS_SO")
BUILD_DIR="$PWD/expanse-build-${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR"
for method in adaptive balanced hybrid; do
    mpicc -O3 -Wall -Wextra -std=c11 \
        -o "$BUILD_DIR/mpi-${method}-timed" \
        "mpi_${method}_timed.c" \
        -L"$OPENBLAS_LIB" \
        -Wl,-rpath,"$OPENBLAS_LIB" -lopenblas -lm
done
export MPI_BUILD_DIR="$BUILD_DIR"
mkdir -p expanse-benchmarks
sha256sum mpi_*_timed.c mpi_multinode_driver_resume.py > \
    "expanse-benchmarks/multinode-resume-sources-${SLURM_JOB_ID}.sha256"
# Correctness smoke test with all three methods across all 4 nodes.
validation="expanse-benchmarks/multinode-resume-validation-${SLURM_JOB_ID}"
mkdir -p "$validation"
./sequential-v21 large-inputs/clustered-strong-A.csr \
    reverse-inputs/X.dense reverse-inputs/W.dense "$validation/reference.dense"
for method in adaptive balanced hybrid; do
    mpirun -np 4 --bind-to core --map-by ppr:1:node \
        "$MPI_BUILD_DIR/mpi-${method}-timed" \
        large-inputs/clustered-strong-A.csr \
        reverse-inputs/X.dense reverse-inputs/W.dense \
        "$validation/${method}.dense" > "$validation/${method}.log"
    python3 validate_large_outputs.py "$validation/reference.dense" \
        "$validation/${method}.dense"
done
rm -f "$validation"/*.dense
python3 mpi_multinode_driver_resume.py \
    --nodes 4 --ranks 16 32 64 \
    --warmups 2 --repetitions 10 \
    --resume-from expanse-benchmarks/mpi-multinode-4nodes-56010108.csv \
    --output "expanse-benchmarks/mpi-multinode-4nodes-resumed-${SLURM_JOB_ID}.csv" \
    --logdir "expanse-benchmarks/multinode-resume-logs-${SLURM_JOB_ID}"
