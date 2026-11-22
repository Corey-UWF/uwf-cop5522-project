#!/bin/bash
#SBATCH --job-name=axw-mpi-one-node
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=Kieren/mpi-one-node-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cpu gcc mvapich2 slurm
RUN_DIR="/scratch/$USER/axw-mpi-one-node-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" Kieren/results
module list
hostname
mpicc --version | head -1
python3 Kieren/benchmark_mpi.py --out "$RUN_DIR" --launcher srun \
  --n 4000 --ranks 1 2 4 --runs 5 | tee "$RUN_DIR/summary.txt"
cp "$RUN_DIR/results.json" "Kieren/results/mpi-one-node-$SLURM_JOB_ID.json"
