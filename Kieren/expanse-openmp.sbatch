#!/bin/bash
#SBATCH --job-name=axw-openmp
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=Kieren/openmp-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cpu gcc slurm
RUN_DIR="/scratch/$USER/axw-openmp-$SLURM_JOB_ID"
mkdir -p "$RUN_DIR" Kieren/results
module list
hostname
gcc --version | head -1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
python3 Kieren/benchmark_openmp.py --out "$RUN_DIR" \
  --blas-library /lib64/libopenblas.so.0 \
  --n 12000 --threads 1 2 4 --runs 5 | tee "$RUN_DIR/summary.txt"
cp "$RUN_DIR/openmp-results.json" "Kieren/results/openmp-$SLURM_JOB_ID.json"
