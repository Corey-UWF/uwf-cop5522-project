#!/bin/bash
#SBATCH --job-name=mpi-single-node
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --mem=8G
#SBATCH --output=mpi-single-node-%j.out
#SBATCH --error=mpi-single-node-%j.err

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

for f in mpi-adaptive mpi-balanced A-benchmark.csr X-benchmark.dense W-benchmark.dense \
         irregular-moderate-A.csr irregular-strong-A.csr \
         clustered-inputs/irregular-moderate-A.csr clustered-inputs/irregular-strong-A.csr \
         mpi_benchmark_driver.py; do
    test -e "$f" || { echo "Missing file: $f" >&2; exit 1; }
done

sha256sum -c SHA256SUMS.txt
sha256sum -c irregular-inputs.sha256
sha256sum -c clustered-inputs.sha256

mkdir -p expanse-benchmarks
printf 'Slurm job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
printf 'Allocated nodes: %s; tasks: %s\n' "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS"
module list

python3 mpi_benchmark_driver.py \
    --warmups 2 --repetitions 10 \
    --ranks 1 2 4 8 16 \
    --output "expanse-benchmarks/mpi-single-node-${SLURM_JOB_ID}.csv" \
    --logdir "expanse-benchmarks/logs-${SLURM_JOB_ID}"
