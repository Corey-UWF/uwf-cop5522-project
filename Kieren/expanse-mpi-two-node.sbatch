#!/bin/bash
#SBATCH --job-name=axw-mpi-two-node
#SBATCH --partition=compute
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=Kieren/mpi-two-node-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cpu gcc mvapich2 slurm
RUN_DIR="/scratch/$USER/axw-mpi-two-node-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" Kieren/results
module list
hostname
mpicc --version | head -1
echo "Rank placement across nodes:"
srun -n 8 hostname | sort | uniq -c
python3 Kieren/benchmark_mpi.py --out "$RUN_DIR" --launcher srun \
  --n 20000 --ranks 1 2 4 8 --runs 5 | tee "$RUN_DIR/summary.txt"
cp "$RUN_DIR/results.json" "Kieren/results/mpi-two-node-$SLURM_JOB_ID.json"
