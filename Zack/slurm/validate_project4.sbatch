#!/bin/bash
#SBATCH --job-name=project4-final
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=project4-final-%j.out
#SBATCH --error=project4-final-%j.err

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

echo "=== Stage 1: Full correctness suite ==="

python3 test_nodeshared.py \
    --executable ./project-4 \
    --reference ./sequential-v21 \
    --mpiexec "mpirun --bind-to none" \
    --ranks 1 2 4 8 16

echo "=== Stage 2: Three-input interface ==="

work="$SLURM_SUBMIT_DIR/expanse-validation/project4-${SLURM_JOB_ID}"
mkdir -p "$work"

project="$SLURM_SUBMIT_DIR"

cd "$work"

"$project/sequential-v21" \
    "$project/large-inputs/regular-A.csr" \
    "$project/large-inputs/X.dense" \
    "$project/large-inputs/W.dense" \
    reference.dense

mpirun -np 16 --bind-to none \
    "$project/project-4" \
    "$project/large-inputs/regular-A.csr" \
    "$project/large-inputs/X.dense" \
    "$project/large-inputs/W.dense"

test -s Y.dense

python3 "$project/validate_large_outputs.py" \
    reference.dense Y.dense

echo "PASS: Three-input interface produced correct output"
echo "ALL PASSED: project-4 final validation"
