#!/bin/bash
#SBATCH --job-name=mpi-multinode
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mpi-multinode-%j.out
#SBATCH --error=mpi-multinode-%j.err
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
         mpi_multinode_driver.py mpi-balanced sequential-v21 \
         validate_large_outputs.py generate_large_inputs.py \
         generate_reverse_inputs.py; do
    test -f "$f" || { echo "Missing file: $f" >&2; exit 1; }
done
python3 -m py_compile mpi_multinode_driver.py
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py --verify
OPENBLAS_SO=$(ldd ./mpi-balanced | awk '/libopenblas.so.0/ {print $3; exit}')
test -f "$OPENBLAS_SO"
OPENBLAS_LIB=$(dirname "$OPENBLAS_SO")
for method in adaptive balanced hybrid; do
    mpicc -O3 -Wall -Wextra -std=c11 -o "mpi-${method}-timed" \
        "mpi_${method}_timed.c" -L"$OPENBLAS_LIB" \
        -Wl,-rpath,"$OPENBLAS_LIB" -lopenblas -lm
done
mkdir -p expanse-benchmarks
sha256sum mpi_*_timed.c mpi_multinode_driver.py > \
    "expanse-benchmarks/multinode-sources-${SLURM_JOB_ID}.sha256"

# Small correctness smoke test in this allocation before measuring.
# Use cluster-strong plus reverse order to exercise weighted partition logic.
validation="expanse-benchmarks/multinode-validation-${SLURM_JOB_ID}"
mkdir -p "$validation"
./sequential-v21 large-inputs/clustered-strong-A.csr \
    reverse-inputs/X.dense reverse-inputs/W.dense "$validation/reference.dense"
for method in adaptive balanced hybrid; do
    mpirun -np "$SLURM_JOB_NUM_NODES" --bind-to core --map-by ppr:1:node \
        "./mpi-${method}-timed" large-inputs/clustered-strong-A.csr \
        reverse-inputs/X.dense reverse-inputs/W.dense "$validation/${method}.dense" \
        > "$validation/${method}.log"
    python3 validate_large_outputs.py "$validation/reference.dense" \
        "$validation/${method}.dense"
done
rm -f "$validation"/*.dense

case "$SLURM_JOB_NUM_NODES" in
  1) ranks=(16) ;;
  2) ranks=(16 32) ;;
  4) ranks=(16 32 64) ;;
  *) echo "Only 1, 2, 4 nodes supported" >&2; exit 1 ;;
esac
python3 mpi_multinode_driver.py \
    --nodes "$SLURM_JOB_NUM_NODES" --ranks "${ranks[@]}" \
    --warmups 2 --repetitions 10 \
    --output "expanse-benchmarks/mpi-multinode-${SLURM_JOB_NUM_NODES}nodes-${SLURM_JOB_ID}.csv" \
    --logdir "expanse-benchmarks/multinode-logs-${SLURM_JOB_ID}"
