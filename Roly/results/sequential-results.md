# Sequential Performance Results

## Goal

My part of the project focuses on improving the sequential performance
of:

    Y = A X W

with `A` stored in CSR format and `X` and `W` stored as row-major dense
matrices.

There are two mathematically equivalent ways to organize the
computation:

    (AX)W
    A(XW)

Rather than assuming one order or one optimization would always be
better, I have been testing the different options and measuring the
results.

## Current File-Driven Versions

### v15 - Baseline

    project1_fileio_portable_v15.c

This is the portable file-driven baseline and computes:

    (AX)W

It follows the professor-style file-driven structure while using a
portable local timer.

### v16 - Optimized (AX)W

    project1_fileio_optimized_v16.c

This version uses:

    FEATURE_BLOCK=128
    PREFETCH_DISTANCE=0
    dense loop order = i-k-j

and still computes:

    (AX)W

### v17 - Optimized A(XW)

    project1_fileio_reassociated_v17.c

This version keeps the optimized kernels from v16 but changes the
association to:

    A(XW)

## Test Problem

The current benchmark uses:

    N = 4000
    F = 128
    H = 64
    nnz(A) = 64000
    16 nonzeros per row

This is a regular synthetic sparse matrix. Every row has the same number
of nonzeros.

## How I Measured It

For the current cs-ssh and Expanse tests:

- all versions used the same input files
- all versions were compiled with GCC 8.5.0 and `-O3`
- file I/O was outside the timed multiplication region
- each version received one warm-up run
- each version was measured five times
- the execution order was rotated between versions
- the median of the five measured runs was used for comparison

## Correctness

Before comparing performance, I checked that the optimized versions were
still producing the expected result.

v15 and v16 produced byte-for-byte identical output.

Their output SHA-256 was:

    8e708a154ae67ca79a153c388526aa1d34e5b08abed998c3a1d87361bf16da29

v17 changes the order of the floating-point operations, so small
rounding differences are expected.

I compared all 256,000 output values from v16 and v17.

Results:

    Maximum absolute difference: 6.11e-05
    Maximum relative difference: 7.49260208e-07
    Tolerance failures: 0

The comparison used:

    absolute tolerance = 1e-4
    relative tolerance = 1e-4

## UWF cs-ssh

CPU:

    Intel Xeon Gold 6338N

Measured runtimes:

| Version | Order | Median | GFLOP/s | Speedup vs v15 |
|---|---|---:|---:|---:|
| v15 | `(AX)W` | 32.369 ms | 2.531 | 1.000x |
| v16 | `(AX)W` | 7.886 ms | 10.389 | 4.105x |
| v17 | `A(XW)` | 5.187 ms | 14.213 | 6.240x |

Compared with v16, v17 was:

    1.520x faster
    34.22% lower runtime

## SDSC Expanse

Performance node:

    exp-1-11

CPU:

    AMD EPYC 7742 64-Core Processor

Measured runtimes:

| Version | Order | Median | GFLOP/s | Speedup vs v15 |
|---|---|---:|---:|---:|
| v15 | `(AX)W` | 32.284 ms | 2.537 | 1.000x |
| v16 | `(AX)W` | 7.226 ms | 11.336 | 4.468x |
| v17 | `A(XW)` | 5.745 ms | 12.834 | 5.620x |

Compared with v16, v17 was:

    1.258x faster
    20.51% lower runtime

## Why A(XW) Helps for This Test

For `(AX)W`, the sparse multiplication works across `F=128` columns.

For `A(XW)`, the dense multiplication happens first and the sparse
multiplication works across `H=64` columns.

For the current dimensions, the nominal operation counts are:

    v16 (AX)W = 81,920,000 FLOPs
    v17 A(XW) = 73,728,000 FLOPs

That is a 10% reduction in nominal floating-point work.

The measured runtime improvement is larger than 10% on both systems, so
the smaller operation count does not appear to explain the entire
difference.

Other factors such as memory access, locality, vectorization, or the
size of the intermediate matrix may also matter, but I do not want to
claim a specific cause until it has been profiled.

## Cross-Platform Comparison

| Version | cs-ssh | Expanse |
|---|---:|---:|
| v15 | 32.369 ms | 32.284 ms |
| v16 | 7.886 ms | 7.226 ms |
| v17 | 5.187 ms | 5.745 ms |

The baseline results were almost identical.

Expanse was faster for v16, while cs-ssh was faster for v17.

That is another reminder that the effect of an optimization can depend
on the processor and the way a particular kernel uses the hardware.

## Earlier Optimization Work

Before moving to the professor's file interface, I used the v12
synthetic benchmark to test several optimization ideas.

The most useful portable change was the dense `i-k-j` loop ordering.

I also tested:

- compiler optimization levels
- `restrict`
- sparse feature blocking
- software prefetching
- GCC vectorization
- assembly output
- hardware counters

Not every optimization helped.

For example, `restrict` simplified the compiler's alias analysis but
made the measured runtime worse, so I did not keep it.

Sparse feature blocking and prefetching helped in some environments but
did not transfer consistently across all of the systems I tested.

The detailed chronological results are kept in:

    Roly/docs/LOG.txt

## Current Takeaway

For the current regular workload:

    N=4000
    F=128
    H=64

v17 `A(XW)` has been the fastest sequential version I have tested on
both cs-ssh and Expanse.

I do not consider that a universal rule.

Earlier testing with:

    F=64
    H=128

showed that `(AX)W` can become faster when the dimensions are reversed.

The better association therefore depends on the dimensions of the
problem, not just on one fixed multiplication order.

## What Still Needs Testing

The current matrix has exactly 16 nonzeros per row.

A real graph can have a much more uneven degree distribution, so I still
want to test an irregular or power-law sparse matrix.

That will give us a better idea of how well the sequential choices hold
up when the sparse workload is less regular, and it will also give the
team a more realistic workload for later parallel scheduling tests.
