# Zack's Additions

## MPI baseline

`mpi_baseline.c` is an initial MPI implementation for the assigned distributed-memory
work. It partitions rows of the CSR matrix as evenly as possible, replicates `X`
and `W` on each rank, computes `Y = (A X) W` for local rows, then gathers the
output on rank 0. This is a correctness-first baseline; it does not yet balance
work by nonzero count or compare the alternate multiplication order `A (X W)`.

### Build and run

Build with an MPI C compiler. The starter accepts the same four-file command
line and matrix encodings as Rolando's sequential versions:

```sh
make
mpirun -np 4 ./mpi-baseline A.csr X.dense W.dense Y.dense
```

The files contain whitespace-separated text values:

- `A.csr`: `rows columns nnz`, followed by `rows+1` zero-based CSR row
  pointers, `nnz` zero-based column indices, then `nnz` single-precision values.
- `X.dense` and `W.dense`: `rows columns`, followed by row-major values.
- `Y.dense`: `rows columns`, followed by row-major values.

The implementation expects `A` to be `N x N`, `X` to be `N x F`, and `W` to
be `F x H`. It computes `(A X) W`, uses contiguous equal-row partitions, and
replicates `X` and `W` on all ranks. The output is written by rank 0; elapsed
distributed time is printed separately to standard output.

### Current limitations and next steps

This is a basic correctness-oriented MPI version, not the final optimized
parallel program. Validate its output against the sequential reference first.
Then compare the current equal-row partition with one balanced by CSR nonzero
counts, and measure multiple process counts on multiple nodes. The current
timing includes MPI distribution, multiplication, and output gathering, but
excludes input-file reading and writing.

## Generating a larger regular workload

`generate_regular_inputs.c` exports the deterministic single-precision formulas
from Rolando's `project1_portable_sequential_v12.c`. It produces a square CSR
matrix with `N=4000`, exactly 16 nonzeros per row (64,000 total), and dense
matrices with `F=128` and `H=64`. Nine significant digits preserve the float
values when the text files are read back.

This is a **v12-derived regular synthetic workload**, not a confirmed reproduction
of the frozen inputs behind Rolando's v15-v17 results. It is not power-law data
and does not test irregular row-load balance.

Upload the generator and updated Makefile to your Zack directory on Bridges-2.
The following commands are for the remote Linux shell:

```sh
make generate-regular-inputs CC=gcc
./generate-regular-inputs regular-A.csr regular-X.dense regular-W.dense
sha256sum regular-A.csr regular-X.dense regular-W.dense > regular-inputs.sha256
```

The generator overwrites its three output files. Use distinct filenames that do
not refer to existing data you need to keep. Retain the checksums with benchmark
results so each implementation can be tested on identical inputs.

Inside a compute allocation, with the matching compiler/MPI modules loaded:

```sh
./sequential-reference regular-A.csr regular-X.dense regular-W.dense Y-seq-regular.dense
mpirun -np 1 ./mpi-baseline regular-A.csr regular-X.dense regular-W.dense Y-mpi-regular-1.dense
mpirun -np 2 ./mpi-baseline regular-A.csr regular-X.dense regular-W.dense Y-mpi-regular-2.dense
mpirun -np 4 ./mpi-baseline regular-A.csr regular-X.dense regular-W.dense Y-mpi-regular-4.dense
```

Adjust the sequential executable path if it is in a sibling directory. Compare
each MPI output with the sequential output; use a numerical tolerance if text
comparison shows floating-point differences. Once correctness is checked, perform
one warm-up and at least five measured runs per supported process count.
Report median elapsed times, speedup relative to one MPI rank, and efficiency.
Record node count, rank placement, modules, compiler flags, and timing boundaries.
This workload may still be too small for useful multi-node scaling; measure before
drawing conclusions or increasing the workload size.
