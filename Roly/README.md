# Rolando Veliz - Sequential Performance Engineering

## Role

My primary responsibility for the COP 5522 term project is sequential
performance engineering for the sparse-dense-dense computation

    Y = A X W

where:

- A is a sparse matrix stored in CSR format.
- X is a dense feature matrix.
- W is a dense weight matrix.
- Y is the resulting dense matrix.

The project compares the two mathematically equivalent multiplication
orders:

    (AX)W
    A(XW)

The broader team project will also investigate OpenMP shared-memory
parallelism and MPI distributed-memory parallelism.

## Current Progress

A controlled sequential benchmark has been developed and validated.

The benchmark:

- checks that both multiplication orders produce matching results;
- performs one warm-up run;
- performs five measured runs;
- reports median execution times;
- reports GFLOP/s for individual stages;
- keeps allocation, initialization, verification, and reporting outside
  the timed multiplication regions.

Several sequential optimization avenues have been investigated rather
than stopping at the first successful optimization.

Experiments completed include:

- compiler optimization levels;
- multiplication-order comparison;
- pointer `restrict`;
- sparse feature blocking;
- software prefetching;
- compiler vectorization analysis;
- generated-assembly inspection;
- dense loop-order optimization;
- cross-platform validation.

## Main Finding So Far

The most important portable optimization has been changing dense matrix
multiplication from the traditional:

    i -> j -> k

loop order to:

    i -> k -> j

This improves row-major memory locality and gives the compiler a much
simpler contiguous inner loop to vectorize.

In the final controlled comparison on the COP 5522 development VM, the
portable sequential version achieved approximately:

- 6.06x speedup for (AX)W;
- 7.56x speedup for A(XW);
- 8.21x speedup for the dense (AX) * W stage;
- 8.70x speedup for the dense X * W stage.

Sparse optimizations were much more architecture-dependent.

## Frozen Sequential Reference

The current portable sequential benchmark configuration is:

    FEATURE_BLOCK=128
    PREFETCH_DISTANCE=0
    DENSE_IKJ=1

Frozen source:

    src/project1_portable_sequential_v12.c

Authoritative source SHA-256:

    2deda5af6e3c95efa72e363d8812fcaabfaa757b8544f4fbd1c0341da16ce627

The frozen source is a performance benchmark/reference implementation.
It is not yet the final Project 2 submission program.

## Cross-Platform Testing

Sequential experiments have been performed on:

- the COP 5522 Linux development VM;
- UWF cs-ssh;
- PSC Bridges-2;
- SDSC Expanse.

The dense i-k-j optimization generalized strongly across the tested
systems. Sparse blocking and software prefetching behaved differently
depending on the processor/compiler environment.

This reinforced an important project principle: optimizations should be
measured rather than assumed to be portable.

## Current Limitation

The current benchmark uses a synthetic sparse matrix with exactly
16 nonzeros per row.

This regular workload is useful for controlled performance experiments,
but it does not yet represent the irregular degree distribution expected
from the power-law graph workload that motivates the project.

## Next Steps

The professor has released the project reference implementation and the
required file-based interface:

    project A.csr X.dense W.dense Y.dense

The next work will:

1. inspect and reproduce the professor's exact CSR and dense text formats;
2. integrate the required file-based interface;
3. transplant the validated sequential kernels into that implementation;
4. test irregular/power-law sparse workloads;
5. continue profiling and performance analysis;
6. evaluate OpenMP thread counts and scheduling strategies;
7. evaluate MPI data distribution and communication;
8. measure speedup, efficiency, load balance, and scalability.

Relevant optimization avenues from the course lectures will continue to
be considered systematically. Techniques that do not improve performance
will also be documented because negative results help identify which
optimizations actually matter for this workload.

## Files

- `src/project1_portable_sequential_v12.c`
  - frozen portable sequential benchmark source.

- `checksums/project1_portable_sequential_v12.sha256`
  - checksum identifying the frozen source.

- `docs/LOG.txt`
  - detailed experimental log, including successful and unsuccessful
    optimization attempts.

- `results/sequential-results.md`
  - summarized sequential performance results and conclusions.
