#!/bin/bash
#SBATCH --job-name=mpi-e2e-ab
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=mpi-e2e-%j.out
#SBATCH --error=mpi-e2e-%j.err
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
python3 -m py_compile benchmark_mpi_e2e.py
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py --verify
for f in mpi_hybrid_replicated_e2e.c mpi_hybrid_replicated_warmup_e2e.c \
         mpi_hybrid_nodeaware_e2e.c mpi-balanced sequential-v21 validate_large_outputs.py; do
    test -f "$f" || { echo "Missing $f" >&2; exit 1; }
done
OPENBLAS_SO=$(ldd ./mpi-balanced | awk '/libopenblas.so.0/ {print $3; exit}')
test -f "$OPENBLAS_SO"
OPENBLAS_LIB=$(dirname "$OPENBLAS_SO")
BUILD_DIR="$PWD/expanse-build-mpi-e2e-${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" expanse-benchmarks
for variant in replicated replicated-warmup nodeaware; do
    case "$variant" in
      replicated) source_file=mpi_hybrid_replicated_e2e.c ;;
      replicated-warmup) source_file=mpi_hybrid_replicated_warmup_e2e.c ;;
      nodeaware) source_file=mpi_hybrid_nodeaware_e2e.c ;;
    esac
    mpicc -O3 -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L \
       -o "$BUILD_DIR/mpi-hybrid-${variant}-e2e" \
       "$source_file" -L"$OPENBLAS_LIB" \
       -Wl,-rpath,"$OPENBLAS_LIB" -lopenblas -lm
done
sha256sum mpi_hybrid_replicated_e2e.c mpi_hybrid_replicated_warmup_e2e.c \
          mpi_hybrid_nodeaware_e2e.c benchmark_mpi_e2e.py \
          > "expanse-benchmarks/mpi-e2e-sources-${SLURM_JOB_ID}.sha256"
# Optional resume source, supplied at submission with --export=ALL,RESUME_FROM=...
resume_args=()
if [[ -n "${RESUME_FROM:-}" ]]; then
    test -f "$RESUME_FROM" || { echo "Missing resume CSV: $RESUME_FROM" >&2; exit 1; }
    resume_args=(--resume-from "$RESUME_FROM")
fi
python3 benchmark_mpi_e2e.py \
    --nodes "$SLURM_JOB_NUM_NODES" --ranks 16 \
    --warmups 2 --repetitions 10 \
    --build-dir "$BUILD_DIR" \
    "${resume_args[@]}" \
    --mpi-retries 3 --retry-delay 5 \
    --output "expanse-benchmarks/mpi-e2e-${SLURM_JOB_NUM_NODES}nodes-${SLURM_JOB_ID}.csv"
