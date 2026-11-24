#!/bin/bash
#SBATCH --job-name=mpi-large-single
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mpi-large-single-%j.out
#SBATCH --error=mpi-large-single-%j.err
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
unset OPENBLAS_CORETYPE || true
for f in mpi-adaptive mpi-balanced sequential-v21 \
         mpi_large_benchmark_driver.py validate_large_outputs.py \
         generate_large_inputs.py generate_irregular_inputs.py \
         large-inputs/SHA256SUMS.txt large-inputs/metadata.json; do
    test -e "$f" || { echo "Missing required file: $f" >&2; exit 1; }
done
python3 generate_large_inputs.py --output-dir large-inputs --verify
mkdir -p expanse-benchmarks
printf 'Job %s, node %s, tasks %s\n' "$SLURM_JOB_ID" "$(hostname)" "$SLURM_NTASKS"
module list
sha256sum mpi_adaptive.c mpi_balanced.c project1_fileio_blas_adaptive_v21.c \
    mpi_large_benchmark_driver.py generate_large_inputs.py generate_irregular_inputs.py \
    > "expanse-benchmarks/large-source-hashes-${SLURM_JOB_ID}.txt"

# Verify against sequential baseline before collecting timings. Avoid running
# validation on the login node. Values are checked with mixed absolute/relative
# tolerance to accommodate floating-point evaluation order.
validation="expanse-benchmarks/large-validation-${SLURM_JOB_ID}"
mkdir -p "$validation"
for dataset in regular clustered-strong; do
    case "$dataset" in
        regular) csr=large-inputs/regular-A.csr ;;
        clustered-strong) csr=large-inputs/clustered-strong-A.csr ;;
    esac
    echo "REFERENCE $dataset"
    ./sequential-v21 "$csr" large-inputs/X.dense large-inputs/W.dense "$validation/reference.dense"
    for p in 1 16; do
        for method in adaptive balanced; do
            echo "VALIDATE $dataset $method p=$p"
            mpirun --bind-to core --map-by core -np "$p" "./mpi-$method" \
                "$csr" large-inputs/X.dense large-inputs/W.dense "$validation/candidate.dense"
            python3 validate_large_outputs.py "$validation/reference.dense" "$validation/candidate.dense"
        done
    done
done
rm -f "$validation/reference.dense" "$validation/candidate.dense"

# Full controlled experiment: 5 datasets x 2 methods x 5 ranks x 10 reps.
python3 mpi_large_benchmark_driver.py \
    --warmups 2 --repetitions 10 \
    --ranks 1 2 4 8 16 \
    --output "expanse-benchmarks/mpi-large-single-${SLURM_JOB_ID}.csv" \
    --logdir "expanse-benchmarks/large-logs-${SLURM_JOB_ID}"
