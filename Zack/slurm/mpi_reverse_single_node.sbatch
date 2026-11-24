#!/bin/bash
#SBATCH --job-name=mpi-reverse
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=mpi-reverse-%j.out
#SBATCH --error=mpi-reverse-%j.err
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

for f in mpi-adaptive mpi-balanced sequential-v21 \
         generate_reverse_inputs.py generate_large_inputs.py generate_irregular_inputs.py \
         mpi_reverse_benchmark_driver.py validate_large_outputs.py \
         large-inputs/SHA256SUMS.txt; do
    test -e "$f" || { echo "Missing $f" >&2; exit 1; }
done
mkdir -p expanse-benchmarks
python3 --version
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py  # generate only if absent; otherwise verify
python3 generate_reverse_inputs.py --verify
printf 'Job %s, node %s, tasks %s\n' "$SLURM_JOB_ID" "$(hostname)" "$SLURM_NTASKS"
module list
sha256sum mpi_adaptive.c mpi_balanced.c project1_fileio_blas_adaptive_v21.c \
    generate_large_inputs.py generate_reverse_inputs.py mpi_reverse_benchmark_driver.py \
    > "expanse-benchmarks/reverse-source-hashes-${SLURM_JOB_ID}.txt"

validation="expanse-benchmarks/reverse-validation-${SLURM_JOB_ID}"
mkdir -p "$validation"
for dataset in regular clustered-strong; do
    csr="large-inputs/${dataset}-A.csr"
    echo "REFERENCE (AX)W $dataset"
    ./sequential-v21 "$csr" reverse-inputs/X.dense reverse-inputs/W.dense "$validation/reference.dense"
    for p in 1 16; do
        for method in adaptive balanced; do
            echo "VALIDATE (AX)W $dataset $method p=$p"
            mpirun --bind-to core --map-by core -np "$p" "./mpi-$method" \
                "$csr" reverse-inputs/X.dense reverse-inputs/W.dense "$validation/candidate.dense" \
                > "$validation/${dataset}-${method}-p${p}.log"
            grep -q '^Order=(AX)W ' "$validation/${dataset}-${method}-p${p}.log" || {
                echo "FAIL: wrong multiplication order" >&2; exit 1;
            }
            python3 validate_large_outputs.py "$validation/reference.dense" "$validation/candidate.dense"
        done
    done
done
rm -f "$validation/reference.dense" "$validation/candidate.dense"

python3 mpi_reverse_benchmark_driver.py \
    --warmups 2 --repetitions 10 \
    --ranks 1 2 4 8 16 \
    --output "expanse-benchmarks/mpi-reverse-single-${SLURM_JOB_ID}.csv" \
    --logdir "expanse-benchmarks/reverse-logs-${SLURM_JOB_ID}"
