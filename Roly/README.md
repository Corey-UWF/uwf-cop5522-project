# Rolando Veliz - Sequential Performance Work

## My Part of the Project

My main responsibility for the COP 5522 term project is the sequential
performance work for:

    Y = A X W

where:

- `A` is a sparse matrix stored in CSR format
- `X` is a dense feature matrix
- `W` is a dense weight matrix
- `Y` is the resulting dense matrix

I have been looking at both mathematically equivalent ways to perform
the computation:

    (AX)W
    A(XW)

My goal has been to first establish a correct sequential baseline and
then use measurements, compiler analysis, and profiling to determine
which optimizations actually help instead of assuming that a change
will automatically make the program faster.

## Earlier Sequential Work

I started with a synthetic benchmark so I could experiment with the
kernels in a controlled environment.

That work is preserved as:

    src/project1_portable_sequential_v12.c

Some of the areas I tested were:

- compiler optimization levels
- sparse feature blocking
- software prefetching
- pointer `restrict`
- dense loop ordering
- GCC vectorization reports
- generated assembly
- hardware performance counters
- cross-platform behavior

The biggest portable improvement came from changing the dense matrix
multiplication loop order from:

    i -> j -> k

to:

    i -> k -> j

With row-major storage, the new inner loop walks through the dense rows
contiguously. GCC also showed that it could vectorize this version much
more effectively.

Sparse blocking and software prefetching were less consistent across
machines. A configuration that helped on the development VM did not
always help on the other systems.

The portable v12 checkpoint uses:

    FEATURE_BLOCK=128
    PREFETCH_DISTANCE=0
    DENSE_IKJ=1

I am keeping v12 in the repository because it documents the earlier
optimization process, but it is no longer the newest project version.

## File-Driven Project Versions

After the professor released the required file interface:

    project A.csr X.dense W.dense Y.dense

I moved the sequential work into that format.

### v15 - File-Driven Baseline

Source:

    src/project1_fileio_portable_v15.c

This is the portable professor-style baseline. The main computational
order is:

    (AX)W

The main portability change was replacing the course-specific timing
header with a local `CLOCK_MONOTONIC` timer. The matrix operations,
file formats, and timing boundaries were otherwise kept consistent with
the professor's reference structure.

### v16 - Optimized (AX)W

Source:

    src/project1_fileio_optimized_v16.c

This version keeps the same multiplication order as v15 but uses the
portable optimizations that survived the earlier testing:

    FEATURE_BLOCK=128
    PREFETCH_DISTANCE=0
    dense loop order = i-k-j

### v17 - Optimized A(XW)

Source:

    src/project1_fileio_reassociated_v17.c

This version uses the same optimized kernels as v16 but changes the
association to:

    A(XW)

For the current test dimensions, `F=128` and `H=64`, this means the
sparse multiplication in v17 works across 64 columns instead of 128.

## Current Performance Results

The current file-driven benchmark uses:

    N = 4000
    F = 128
    H = 64
    nnz(A) = 64000
    16 nonzeros per row

Each version was given one warm-up run followed by five measured runs.
The execution order was rotated between versions, and the median of the
five measured runs was used for comparison.

File I/O was kept outside the timed multiplication region.

### UWF cs-ssh

CPU:

    Intel Xeon Gold 6338N

| Version | Median runtime | Speedup vs v15 |
|---|---:|---:|
| v15 baseline `(AX)W` | 32.369 ms | 1.000x |
| v16 optimized `(AX)W` | 7.886 ms | 4.105x |
| v17 optimized `A(XW)` | 5.187 ms | 6.240x |

For this workload, v17 was about 1.52x faster than v16 on cs-ssh.

### SDSC Expanse

Performance node:

    exp-1-11

CPU:

    AMD EPYC 7742 64-Core Processor

| Version | Median runtime | Speedup vs v15 |
|---|---:|---:|
| v15 baseline `(AX)W` | 32.284 ms | 1.000x |
| v16 optimized `(AX)W` | 7.226 ms | 4.468x |
| v17 optimized `A(XW)` | 5.745 ms | 5.620x |

For this workload, v17 was about 1.26x faster than v16 on Expanse.

The same reassociation helped on both systems, although the amount of
improvement was different on each machine.

## Correctness

v15 and v16 produced byte-for-byte identical output.

The output SHA-256 was:

    8e708a154ae67ca79a153c388526aa1d34e5b08abed998c3a1d87361bf16da29

v17 changes the order of the floating-point operations, so I did not
expect its output to be byte-for-byte identical.

A full comparison of all 256,000 output values between v16 and v17
showed:

    Maximum absolute difference: 6.11e-05
    Maximum relative difference: 7.49260208e-07
    Tolerance failures: 0

using:

    absolute tolerance = 1e-4
    relative tolerance = 1e-4

## What I Have Learned So Far

For the current workload with:

    N=4000, F=128, H=64

v17 `A(XW)` has been the fastest sequential version on both cs-ssh and
Expanse.

That does not mean `A(XW)` will always be faster.

Earlier testing with `F=64` and `H=128` showed that the preferred order
can reverse. The matrix dimensions affect how much work is done in the
sparse stage, so the better order depends on the shape of the problem.

The results also showed that some optimizations are much more portable
than others. The dense `i-k-j` loop change helped consistently, while
sparse blocking and software prefetching depended much more on the
machine.

## Current Limitation

The current sparse matrix has exactly 16 nonzeros in every row.

That makes it useful for controlled testing, but it does not behave like
the irregular or power-law graphs that motivate the project.

The next useful sequential test is an irregular sparse workload so we
can see whether the same conclusions still hold when the row lengths
are not uniform.

## Files

### Source

- `src/project1_portable_sequential_v12.c`
  - earlier portable optimization checkpoint

- `src/project1_fileio_portable_v15.c`
  - file-driven sequential baseline

- `src/project1_fileio_optimized_v16.c`
  - optimized `(AX)W`

- `src/project1_fileio_reassociated_v17.c`
  - optimized `A(XW)`

### Results

- `results/sequential-results.md`
  - summary of the current sequential results

- `results/sequential-cross-platform-v15-v17.txt`
  - detailed cs-ssh and Expanse comparison

### Experiment History

- `docs/LOG.txt`
  - chronological notes from the sequential experiments, including
    approaches that helped and approaches that did not

### Checksums

The `checksums/` directory contains SHA-256 files for the frozen source
versions so the exact tested files can be identified later.
