COP 5522 Term Project 1
Sequential Primary Benchmark Inputs

These are the exact frozen input files used for the primary sequential
performance benchmark.

Operation:

    Y = A X W

Primary benchmark dimensions:

    A: 4000 x 4000 sparse CSR
    nnz(A): 64000

    X: 4000 x 128 dense

    W: 128 x 64 dense

    Y: 4000 x 64 dense

Files:

    A-benchmark.csr
    X-benchmark.dense
    W-benchmark.dense

The SHA-256 checksums in SHA256SUMS.txt can be used to verify that the
files are identical to the benchmark inputs used during testing.

The sequential implementation was tested as a single-process,
single-threaded implementation. Dense matrix multiplication used serial
OpenBLAS in the BLAS-based versions.
