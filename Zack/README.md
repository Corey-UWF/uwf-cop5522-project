# Zack's Additions

## MPI baseline

`mpi_baseline.c` is an initial MPI implementation for the assigned distributed-memory
work. It partitions rows of the CSR matrix as evenly as possible, replicates `X`
and `W` on each rank, computes `Y = (A X) W` for local rows, then gathers the
output on rank 0. This is a correctness-first baseline; it does not yet balance
work by nonzero count or compare the alternate multiplication order `A (X W)`.

### Build and run

Build with an MPI C compiler:

```sh
make
mpirun -np 4 ./mpi-baseline A_file X_file W_file
```

The current input and output encodings are provisional until the course reference
files and required output format are available:

- `A_file`: `N nnz`, followed by `N+1` zero-based CSR row pointers, `nnz`
  zero-based column indices, then `nnz` values.
- `X_file` and `W_file`: `rows columns`, followed by row-major values.
- Standard output: `N H`, followed by the `N x H` result in row-major order.

Values are whitespace-separated text numbers. The implementation expects
`A` to be `N x N`, `X` to be `N x F`, and `W` to be `F x H`. Update the
provisional readers and output writer once the official file format is confirmed.
