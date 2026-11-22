# Kieren's MPI correctness and scaling experiment

This experiment tests the team's existing `Zack/mpi_baseline.c` on matched
regular and skewed CSR inputs. It uses `Roly/src/project1_fileio_portable_v15.c`
as a numerical reference. All sources are from the same repository checkout.
The MPI code computes `(AX)W` with contiguous equal-row partitions. Rolando's
adaptive v21 can use a different operation order, so MPI speedup is calculated
against the one-rank MPI median, not against v21.

The generator uses `N=4000`, `F=128`, `H=64`, 16 nonzeros per row for the
regular case, and a fixed-seed skewed row distribution with exactly the same
total nonzeros for the irregular case. These are **new synthetic inputs**, not
Rolando's frozen A060 and A075 matrices. The script records dimensions, row
statistics, input checksums, raw runs, compiler, host and scheduler metadata.
It compares each MPI output against v15 using
`abs(mpi-reference) <= 1e-4 + 1e-4*abs(reference)` and stops on any failure.

The timed MPI interval includes distribution, computation and gathering. It
excludes input parsing and output writing. Run one warm-up and five measured
runs per rank count; compare medians. The one-rank MPI median defines speedup
and efficiency for a given workload. Do not directly combine the one-node
`N=4000` table with the two-node `N=20000` table into one speedup curve.

## Local preparation

From the repository root, this generates and fingerprints the inputs without
needing an MPI development installation:

```sh
python3 Kieren/benchmark_mpi.py --out /tmp/axw-input-check --generate-only
```

Where MPI headers and libraries are installed, a small smoke test is:

```sh
python3 Kieren/benchmark_mpi.py --out /tmp/axw-mpi-check --ranks 1 2 4
```

## Expanse runs

Clone or stage this branch on Expanse, then from the repository root use your
course allocation name in place of `YOUR_ACCOUNT`:

```sh
sbatch --account=YOUR_ACCOUNT Kieren/expanse-mpi-one-node.sbatch
sbatch --account=YOUR_ACCOUNT Kieren/expanse-mpi-two-node.sbatch
```

The first job uses the shared partition and a 4,000-row input for correctness
and one-node scaling. The second requests two compute nodes and a larger
20,000-row input for the course's multi-node requirement. It runs 1, 2, 4 and
8 ranks inside the *same two-node allocation*, so those numbers form a
consistent scaling curve. The compute partition allocates whole nodes even
though this experiment uses only four tasks per node; record this resource
cost and the actual rank placement in the final analysis.

Both jobs save compact `Kieren/results/mpi-*.json` files and write Slurm logs
to `Kieren/mpi-*.out`. Generated input and output matrices remain in the
per-job scratch directory reported by the script. Check `correctness` for zero
failures before reporting any timing. If MPI speedup is poor, inspect input
size, per-rank nonzero counts and communication timing before claiming a
specific cause; the baseline currently does not report a phase breakdown.

The benchmark does not provide the planned nonzero-balanced MPI variant or
OpenMP results. Those require working implementations before their comparisons
can be measured.
