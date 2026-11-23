#!/bin/bash
#SBATCH --job-name=mpi-large-inputs
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --output=mpi-large-inputs-%j.out
#SBATCH --error=mpi-large-inputs-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python3 --version
python3 generate_large_inputs.py --output-dir large-inputs
python3 generate_large_inputs.py --output-dir large-inputs --verify
sha256sum generate_large_inputs.py generate_irregular_inputs.py > large-inputs/generator-sources.sha256
