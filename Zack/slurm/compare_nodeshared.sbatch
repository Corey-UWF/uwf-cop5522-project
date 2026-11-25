#!/bin/bash
# Compares the hybrid MPI version with mpi_nodeshared.c on 1, 2 or 4 Expanse nodes.
# Submit with the node count you want, for example:  sbatch --nodes=2 compare_nodeshared.sbatch
# Needs these files in the submit directory (the same flat layout as the other jobs):
#   mpi_hybrid_timed.c mpi_nodeshared.c sequential-v21 compare_variants.py
#   validate_large_outputs.py generate_large_inputs.py generate_reverse_inputs.py
#   generate_irregular_inputs.py large-inputs/ reverse-inputs/
#SBATCH --job-name=nodeshared-compare
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=nodeshared-%j.out
#SBATCH --error=nodeshared-%j.err
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
for f in mpi_hybrid_timed.c mpi_nodeshared.c sequential-v21 compare_variants.py \
    validate_large_outputs.py; do
    test -f "$f" || { echo "Missing $f" >&2; exit 1; }
done
python3 generate_large_inputs.py --verify
python3 generate_reverse_inputs.py --verify

OPENBLAS_SO=$(ldd ./sequential-v21 | awk '/libopenblas.so.0/ {print $3; exit}')
OPENBLAS_LIB=$(dirname "$OPENBLAS_SO")
BUILD_DIR="$PWD/expanse-build-nodeshared-${SLURM_JOB_ID}"
mkdir -p "$BUILD_DIR" expanse-benchmarks
for program in mpi_hybrid_timed mpi_nodeshared; do
    mpicc -O3 -Wall -Wextra -std=c11 -o "$BUILD_DIR/${program}" "${program}.c" \
        -L"$OPENBLAS_LIB" -Wl,-rpath,"$OPENBLAS_LIB" -lopenblas -lm
done
sha256sum mpi_hybrid_timed.c mpi_nodeshared.c compare_variants.py \
    > "expanse-benchmarks/nodeshared-sources-${SLURM_JOB_ID}.sha256"

NODES=$SLURM_JOB_NUM_NODES
if [ "$NODES" -eq 1 ]; then
    RANKS="1 2 4 8 16"
    MPIEXEC="mpirun --bind-to core"
else
    RANKS=$((NODES * 16))   # 16 ranks on every node
    MPIEXEC="mpirun --bind-to core --map-by ppr:16:node"
fi
L=large-inputs
R=reverse-inputs
python3 compare_variants.py --reference ./sequential-v21 \
    --variant "hybrid=$BUILD_DIR/mpi_hybrid_timed" \
    --variant "nodeshared=$BUILD_DIR/mpi_nodeshared" \
    --dataset regular_AXW "$L/regular-A.csr" "$L/X.dense" "$L/W.dense" \
    --dataset clustered_AXW "$L/clustered-strong-A.csr" "$L/X.dense" "$L/W.dense" \
    --dataset regular_AX_W "$L/regular-A.csr" "$R/X.dense" "$R/W.dense" \
    --dataset clustered_AX_W "$L/clustered-strong-A.csr" "$R/X.dense" "$R/W.dense" \
    --ranks $RANKS --warmups 2 --repetitions 10 \
    --mpiexec "$MPIEXEC" \
    --workdir "$BUILD_DIR/out" \
    --csv "expanse-benchmarks/nodeshared-${NODES}nodes-${SLURM_JOB_ID}.csv"
