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
