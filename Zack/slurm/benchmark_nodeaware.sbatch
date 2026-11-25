#!/bin/bash
#SBATCH --job-name=nodeaware-test
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --output=nodeaware-%j.out
#SBATCH --error=nodeaware-%j.err
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
python3 -m py_compile benchmark_nodeaware.py
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py --verify
for f in mpi_hybrid_timed.c mpi_hybrid_replicated_timed.c mpi_hybrid_nodeaware_timed.c \
    mpi-balanced sequential-v21 validate_large_outputs.py; do
    test -f "$f" || { echo "Missing $f" >&2; exit 1; }
done
OPENBLAS_SO=$(ldd ./mpi-balanced | awk '/libopenblas.so.0/ {print $3; exit}')
test -f "$OPENBLAS_SO"
OPENBLAS_LIB=$(dirname "$OPENBLAS_SO")
BUILD_DIR="$PWD/expanse-build-nodeaware-${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" expanse-benchmarks
for variant in original replicated nodeaware; do
    case "$variant" in
       original) source_file=mpi_hybrid_timed.c ;;
       replicated) source_file=mpi_hybrid_replicated_timed.c ;;
       nodeaware) source_file=mpi_hybrid_nodeaware_timed.c ;;
    esac
    mpicc -O3 -Wall -Wextra -std=c11 \
        -o "$BUILD_DIR/mpi-hybrid-${variant}-timed" \
        "$source_file" -L"$OPENBLAS_LIB" \
        -Wl,-rpath,"$OPENBLAS_LIB" -lopenblas -lm
done
sha256sum mpi_hybrid_timed.c mpi_hybrid_replicated_timed.c mpi_hybrid_nodeaware_timed.c \
    benchmark_nodeaware.py > "expanse-benchmarks/nodeaware-sources-${SLURM_JOB_ID}.sha256"
# Hold total MPI ranks at 16 while checking node-aware against the matching reference.
python3 benchmark_nodeaware.py \
    --nodes "$SLURM_JOB_NUM_NODES" --ranks 16 \
    --warmups 2 --repetitions 10 \
    --build-dir "$BUILD_DIR" \
    --output "expanse-benchmarks/nodeaware-${SLURM_JOB_NUM_NODES}nodes-${SLURM_JOB_ID}.csv"
