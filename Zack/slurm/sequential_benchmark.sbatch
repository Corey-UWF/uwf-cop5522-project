#!/bin/bash
#SBATCH --job-name=seq-baseline
#SBATCH --account=fsu125
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=seq-baseline-%j.out
#SBATCH --error=seq-baseline-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load cpu/0.17.3b
module load gcc/10.2.0/npcyll4
module load openblas/0.3.18

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
unset OPENBLAS_CORETYPE || true

mkdir -p expanse-benchmarks

csv="expanse-benchmarks/sequential-baseline-${SLURM_JOB_ID}.csv"
output="expanse-benchmarks/sequential-temp-${SLURM_JOB_ID}.dense"

echo "job_id,dataset,order,repetition,time_us,time_ms" > "$csv"

for dataset in large_regular clustered_strong; do
    if [[ "$dataset" == "large_regular" ]]; then
        csr="large-inputs/regular-A.csr"
    else
        csr="large-inputs/clustered-strong-A.csr"
    fi

    for order in "A(XW)" "(AX)W"; do
        if [[ "$order" == "A(XW)" ]]; then
            x="large-inputs/X.dense"
            w="large-inputs/W.dense"
        else
            x="reverse-inputs/X.dense"
            w="reverse-inputs/W.dense"
        fi

        echo "Benchmark: $dataset $order"

        # Two warmup repetitions
        for rep in 1 2; do
            ./sequential-v21 "$csr" "$x" "$w" "$output" > /dev/null
        done

        # Ten measured repetitions
        for rep in $(seq 1 10); do
            result=$(./sequential-v21 "$csr" "$x" "$w" "$output")

            time_us=$(printf '%s\n' "$result" | awk \
                '/Time[[:space:]]*=/ {
                    for (i=1; i<=NF; i++)
                        if ($i=="Time" && $(i+1)=="=") {
                            print $(i+2)
                            exit
                        }
                }')

            if [[ -z "$time_us" ]]; then
                echo "Could not parse timing: $result" >&2
                exit 1
            fi

            time_ms=$(awk -v t="$time_us" 'BEGIN {printf "%.6f",t/1000}')

            echo "$SLURM_JOB_ID,$dataset,$order,$rep,$time_us,$time_ms" >> "$csv"
            echo "$dataset $order rep=$rep time=${time_ms}ms"
        done
    done
done

rm -f "$output"

echo "PASS: sequential benchmark finished"
echo "Results: $csv"
