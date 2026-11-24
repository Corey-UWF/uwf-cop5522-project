# Rolando Veliz - Sequential Performance Work

## My Part of the Project

My main responsibility for the COP 5522 term project is the sequential
performance work for:

    Y = A X W

where:

- `A` is an `N x N` sparse matrix stored in CSR format
- `X` is an `N x F` dense feature matrix
- `W` is an `F x H` dense weight matrix
- `Y` is the resulting `N x H` dense matrix

I studied both mathematically equivalent evaluation orders:

    (AX)W
    A(XW)

My goal was to begin with a correct sequential baseline that follows the
professor's required interface, then use controlled measurements,
compiler analysis, profiling, and cross-platform testing to determine
which optimizations actually improved performance.

All of my sequential work remained single-process and single-threaded,
while the team handles the MPI implementation separately.

---

## Final Sequential Implementation

The final portable sequential source is:

    src/project1_fileio_blas_adaptive_v21.c

v21 combines:

- the optimized CSR-dense kernel developed during the earlier sequential
  work
- serial OpenBLAS for dense SGEMM
- the `noinline` CSR-kernel decision established by the v19/v20 compiler
  investigation
- adaptive selection between the two mathematically equivalent evaluation
  orders

The adaptive rule is:

    if H <= F:
        use A(XW)
    else:
        use (AX)W

For the tested dimension regimes, this applies the sparse multiplication
to the narrower dense width.

For the primary benchmark:

    N = 4000
    F = 128
    H = 64
    nnz(A) = 64000

v21 selects:

    A(XW)

For the reverse benchmark:

    N = 4000
    F = 64
    H = 128
    nnz(A) = 64000

v21 selects:

    (AX)W

On cs-ssh, compiling the exact v21 source with
`-march=native -mtune=native` reduced the primary-shape runtime from
2.097 ms to 1.889 ms relative to the portable `-O3` build, a 9.91%
reduction.

On Expanse, the native build reduced the primary-shape runtime from
1.968 ms to 1.898 ms, a 3.54% reduction. The final native result was
16.682x faster than the portable v15 baseline on that primary workload.

For the reverse shape on Expanse, portable v21 was 30.17% faster than
the v20 fixed `A(XW)` checkpoint in the controlled 21-round experiment,
with v21 winning all 21 paired measurements. Because v20 and v21 both
retain the `noinline` CSR-kernel decision, this provides a cleaner
comparison of the evaluation-order change itself.

For additional context, v21 was also 20.07% faster than v18 on the same
reverse workload.

All tested correctness comparisons passed.

The final v21 validation record is:

    results/sequential-v21-adaptive-20261122.md

The earlier v15-v20 results remain preserved in:

    results/sequential-final-20261121.md

---

## Optimization History

### v12 - Earlier Portable Optimization Checkpoint

Source:

    src/project1_portable_sequential_v12.c

Before moving to the professor's file-driven interface, I used a
controlled synthetic benchmark to investigate:

- compiler optimization
- sparse feature blocking
- software prefetching
- pointer `restrict`
- dense loop ordering
- GCC vectorization
- generated assembly
- hardware performance counters
- cross-platform behavior

One of the largest portable early improvements came from changing the
dense matrix multiplication loop order from:

    i -> j -> k

to:

    i -> k -> j

With row-major storage, the new inner loop accesses the dense rows
contiguously. GCC also showed that this inner loop could be vectorized
more effectively.

Other optimizations were less portable. For example, `restrict`,
software prefetching, and some sparse blocking configurations either
regressed or produced architecture-dependent results.

The retained v12 configuration used:

    FEATURE_BLOCK=128
    PREFETCH_DISTANCE=0
    DENSE_IKJ=1

v12 is preserved because it documents the optimization process that led
to the later file-driven versions.

---

## File-Driven Sequential Versions

### v15 - Portable Baseline

Source:

    src/project1_fileio_portable_v15.c

Association:

    (AX)W

This is the professor-style portable file-driven baseline.

The main portability change was replacing the course-specific timing
header with a local `CLOCK_MONOTONIC` timer while keeping the matrix
operations, file formats, and timing boundaries consistent with the
reference structure.

### v16 - Optimized (AX)W

Source:

    src/project1_fileio_optimized_v16.c

Association:

    (AX)W

Retained optimizations:

    FEATURE_BLOCK=128
    PREFETCH_DISTANCE=0
    dense loop order = i-k-j

### v17 - Optimized A(XW)

Source:

    src/project1_fileio_reassociated_v17.c

Association:

    A(XW)

v17 keeps the optimized kernels from v16 but changes the algebraic
association.

For the primary dimensions `F=128` and `H=64`, this reduces the width of
the sparse stage from 128 columns to 64 columns.

### v18 - BLAS (AX)W

Source:

    src/project1_fileio_blas_v18.c

Association:

    (AX)W

v18 replaces the hand-written dense matrix multiplication with serial
OpenBLAS SGEMM while keeping the optimized sparse kernel and the `(AX)W`
evaluation order.

### v19 - BLAS A(XW)

Source:

    src/project1_fileio_blas_reassociated_v19.c

Association:

    A(XW)

v19 combines algebraic reassociation with serial OpenBLAS.

During testing, v19 exposed an important compiler behavior: the CSR
kernel could retire substantially more instructions depending on how
GCC inlined it at the call site.

### v20 - BLAS A(XW) with CSR noinline

Source:

    src/project1_fileio_blas_reassociated_noinline_v20.c

Association:

    A(XW)

v20 differs from v19 by applying:

    __attribute__((noinline))

to the CSR-dense multiplication function.

Controlled tests showed that keeping this function out of line
substantially improved the performance of the generated code on both
Intel Xeon and AMD EPYC.

`noclone` alone did not provide the same improvement, so I did not keep
it in the subsequent implementation.

### v21 - Adaptive BLAS Evaluation Order

Source:

    src/project1_fileio_blas_adaptive_v21.c

v21 retains the optimized sparse kernel, serial OpenBLAS, and the
`noinline` CSR-kernel decision from v20.

The new change is adaptive evaluation-order selection:

    if H <= F:
        use A(XW)
    else:
        use (AX)W

For `(AX)W`, the sparse multiplication operates across width `F`.

For `A(XW)`, the sparse multiplication operates across width `H`.

The adaptive rule therefore chooses the tested association that applies
the sparse operation to the narrower dense width.

The primary `F=128, H=64` workload selects `A(XW)`, while the reverse
`F=64, H=128` workload selects `(AX)W`.

Controlled tests on both cs-ssh and Expanse showed that the adaptive
implementation preserved correctness. The longer paired tests also
showed that v21 matched the best fixed choice on the primary shape and
provided a clear improvement when the dimensions were reversed.

The exact same portable source was also tested with:

    -O3 -march=native -mtune=native

Native compilation improved both tested shapes on both Intel and AMD.
Those flags are treated as an optional target-specific build
optimization rather than a requirement of the portable source.

---

## Final Six-Version Performance

### UWF cs-ssh

CPU:

    Intel Xeon Gold 6338N

Compiler:

    GCC 8.5.0

| Version | Association | Median runtime | GFLOP/s | Speedup vs v15 |
|---|---|---:|---:|---:|
| v15 | `(AX)W` | 32.329 ms | 2.534 | 1.000x |
| v16 | `(AX)W` | 7.303 ms | 11.218 | 4.427x |
| v17 | `A(XW)` | 5.209 ms | 14.155 | 6.207x |
| v18 | `(AX)W` + BLAS | 2.921 ms | 28.046 | 11.068x |
| v19 | `A(XW)` + BLAS | 3.231 ms | 22.822 | 10.007x |
| v20 | `A(XW)` + BLAS + noinline | **1.961 ms** | **37.597** | **16.486x** |

For the final comparison, v20 was:

    1.489x faster than v18
    1.647x faster than v19

### SDSC Expanse

CPU:

    AMD EPYC 7742 64-Core Processor

Compiler:

    GCC 8.5.0

| Version | Association | Median runtime | GFLOP/s | Speedup vs v15 |
|---|---|---:|---:|---:|
| v15 | `(AX)W` | 31.861 ms | 2.571 | 1.000x |
| v16 | `(AX)W` | 6.891 ms | 11.888 | 4.624x |
| v17 | `A(XW)` | 5.492 ms | 13.424 | 5.801x |
| v18 | `(AX)W` + BLAS | 3.512 ms | 23.325 | 9.072x |
| v19 | `A(XW)` + BLAS | 2.899 ms | 25.430 | 10.990x |
| v20 | `A(XW)` + BLAS + noinline | **2.314 ms** | **31.855** | **13.766x** |

For the final comparison, v20 was:

    1.517x faster than v18
    1.253x faster than v19

---

## Measurement Method

The final controlled six-version comparison used:

- the same frozen sources and benchmark inputs
- GCC 8.5.0
- `-O3`
- file I/O outside the timed multiplication region
- one process and one computational thread
- controlled CPU affinity
- three warm-up executions per version
- seven measured rounds
- rotated version order
- median runtime as the primary result
- independent numerical correctness checks

Runtime is the primary metric because the two evaluation orders can
perform different numbers of nominal floating-point operations.

### OpenBLAS on cs-ssh

Serial OpenBLAS 0.3.15 was used with:

    OPENBLAS_NUM_THREADS=1
    OPENBLAS_CORETYPE=SkylakeX

OpenBLAS automatic detection selected its much slower Prescott kernel on
this host, so I explicitly selected the appropriate SkylakeX
implementation.

### OpenBLAS on Expanse

The serial OpenBLAS library was used with:

    OPENBLAS_NUM_THREADS=1

The native AMD Zen implementation was used. No Intel-specific core type
was forced on Expanse.

---

## Correctness

The final implementations were checked against the portable baseline
using:

    absolute tolerance = 1e-4
    relative tolerance = 1e-4

For the primary `N=4000, F=128, H=64` benchmark, every comparison had:

    tolerance failures = 0

Small numerical differences are expected after algebraic reassociation
or BLAS because those changes alter the order of floating-point
operations.

v15 and v16 produced byte-for-byte identical output in the earlier
comparison.

Their verified output SHA-256 was:

    8e708a154ae67ca79a153c388526aa1d34e5b08abed998c3a1d87361bf16da29

---

## Dimension Sensitivity

One of the strongest findings was that the best evaluation order depends
on the dense dimensions.

### Primary Shape

    F = 128
    H = 64

For this shape:

    (AX)W sparse width = 128
    A(XW) sparse width = 64

The `A(XW)` order was faster on both tested architectures.

### Reversed Shape

I then reversed the dense dimensions:

    F = 64
    H = 128

Now:

    (AX)W sparse width = 64
    A(XW) sparse width = 128

The performance preference also reversed.

#### cs-ssh

    v18 (AX)W = 2.815 ms
    v20 A(XW) = 3.715 ms

`(AX)W` was approximately 24.25% faster.

#### Expanse

    v18 (AX)W = 2.926 ms
    v20 A(XW) = 3.501 ms

`(AX)W` was approximately 16.42% faster.

The preferred association therefore reversed on both Intel and AMD.

For the workloads I tested, the best results came from applying the
sparse multiplication to the narrower dense intermediate.

For:

    F=128, H=64

`A(XW)` applies the sparse multiplication to 64 columns.

For:

    F=64, H=128

`(AX)W` applies the sparse multiplication to 64 columns.

This is an experimental result for the tested workloads, not a universal
rule for every sparse matrix or processor.

---

## Compiler Inlining Result

The v19 investigation initially made it look as though the sparse stage
became slower simply because it followed SGEMM.

Further testing did not support that explanation.

The investigation included:

- equal-work matrix dimensions
- alternative OpenBLAS core types
- delays after SGEMM
- pre-touching the intermediate matrix
- hardware performance counters
- compiler vectorization reports
- `noinline`
- `noclone`
- `noinline,noclone`

The important observation was that the slower context retired
approximately 1.91x as many instructions for the CSR stage even though
the C kernel and nominal CSR work were the same.

On cs-ssh:

    original v19 = 3.344 ms
    noinline     = 2.350 ms
    noclone      = 3.340 ms
    both         = 2.354 ms

`noinline` reduced runtime by approximately 29.73%.

On Expanse:

    original = 2.787 ms
    noinline = 2.240 ms
    both     = 2.259 ms

`noinline` reduced runtime by approximately 19.63%.

Based on these results, preventing GCC from inlining the CSR-dense
kernel materially improved performance for this workload on both tested
architectures.

I did not isolate the exact compiler-level cause beyond the inlining
decision, so I do not make a stronger claim than the measurements
support.

---

## Synthetic Irregular Sparse Matrices

The primary sparse benchmark is regular:

    mean degree   = 16
    median degree = 16
    minimum       = 16
    maximum       = 16
    standard deviation = 0

I also tested two deterministic synthetic irregular sparse matrices
while keeping:

    N = 4000
    nnz(A) = 64000
    mean degree = 16

### A060

    median degree = 10
    minimum = 7
    maximum = 955
    standard deviation = 28.304

### A075

    median degree = 8
    minimum = 4
    maximum = 2256
    standard deviation = 55.019

These matrices change both the row-degree distribution and the
column-index pattern. Because of that, I treat them as tests of
sensitivity to changes in sparse structure rather than experiments that
isolate row irregularity by itself.

### cs-ssh

| Matrix | v15 | v18 | v20 | v20 speedup vs v15 |
|---|---:|---:|---:|---:|
| Regular | 32.403 ms | 2.879 ms | **2.176 ms** | 14.889x |
| A060 | 32.017 ms | 3.447 ms | **2.110 ms** | 15.172x |
| A075 | 32.177 ms | 3.387 ms | **2.083 ms** | 15.445x |

### Expanse

| Matrix | v15 | v18 | v20 | v20 speedup vs v15 |
|---|---:|---:|---:|---:|
| Regular | 31.824 ms | 3.499 ms | **2.156 ms** | 14.760x |
| A060 | 32.227 ms | 3.777 ms | **2.616 ms** | 12.319x |
| A075 | 32.234 ms | 3.758 ms | **2.600 ms** | 12.397x |

v20 remained the fastest tested sequential implementation for every
regular and synthetic irregular matrix on both architectures.

However, the effect of the sparse-structure changes was
architecture-dependent.

On cs-ssh, v20 was essentially unchanged in these measurements, while
v18 slowed by roughly 18-20%.

On Expanse, v20 slowed by about 21% on the irregular cases, while v18
slowed by about 7-8%.

For that reason, I do not claim that v20 is universally less sensitive
to irregular sparse structure.

The results show that v20 retained the best overall runtime across the
tested sparse structures, while the size and direction of the
sparse-structure effect depended on the architecture.

---

## Main Technical Lessons

Several useful findings came out of the sequential optimization work.

The dense `i-k-j` traversal was a strong portable optimization because
it improved contiguous row-major access and compiler vectorization.

Serial BLAS provided another major improvement, although selecting the
correct OpenBLAS architecture mattered on cs-ssh.

Algebraically equivalent evaluation orders were not performance
equivalent. The preferred association changed when `F` and `H` were
reversed.

v21 turns that observation into an adaptive decision. For the tested
dimension regimes, it selects the association that gives the sparse
multiplication the narrower dense width.

Compiler decisions also mattered. Preventing GCC from inlining the CSR
kernel improved the BLAS implementation on both Intel and AMD.
Target-specific `-march=native -mtune=native` compilation then provided
an additional measurable improvement on both tested architectures.

Finally, sparse-structure sensitivity differed between architectures.
This reinforces the need to validate optimization decisions on more than
one system instead of assuming that behavior on one machine will
generalize everywhere.

The main takeaway from the sequential experiments is:

> Algebraically equivalent evaluation orders do not necessarily have
> equivalent performance. The evaluation order determines the width of
> the sparse stage, and compiler inlining can sometimes hide that
> advantage. For the matrix shapes I tested, applying the sparse
> multiplication to the narrower intermediate dimension produced the
> better or equivalent result after controlling for the inlining effect.
> v21 uses that result to select the evaluation order adaptively.

---

## Repository Files

### Sequential Sources

- `src/project1_portable_sequential_v12.c`
  - earlier portable optimization checkpoint

- `src/project1_fileio_portable_v15.c`
  - portable file-driven baseline

- `src/project1_fileio_optimized_v16.c`
  - optimized `(AX)W`

- `src/project1_fileio_reassociated_v17.c`
  - optimized `A(XW)`

- `src/project1_fileio_blas_v18.c`
  - BLAS `(AX)W`

- `src/project1_fileio_blas_reassociated_v19.c`
  - BLAS `A(XW)`

- `src/project1_fileio_blas_reassociated_noinline_v20.c`
  - prior fixed-association BLAS + noinline checkpoint

- `src/project1_fileio_blas_adaptive_v21.c`
  - final portable adaptive sequential implementation

### Results

- `results/sequential-results.md`
  - earlier v15-v17 performance record

- `results/sequential-cross-platform-v15-v17.txt`
  - detailed earlier cs-ssh and Expanse comparison

- `results/sequential-final-20261121.md`
  - frozen consolidated v15-v20 checkpoint and experiments

- `results/sequential-v21-adaptive-20261122.md`
  - final adaptive v21 validation and native-build results

### Benchmark and Reproducibility Files

- `benchmarks/benchmark-irregular-cs-20261121.sh`
  - cs-ssh regular/A060/A075 controlled benchmark script

- `evidence/expanse-20261121/`
  - preserved Expanse reverse-shape and irregular-structure job scripts,
    outputs, empty error files, and checksum manifest

- `evidence/expanse-v21-20261122/`
  - final Expanse v21 job script, output, empty error file, raw 21-round
    CSV, and SHA-256 manifest

### Checksums

The `checksums/` directory contains SHA-256 records for the frozen
sequential sources and benchmark artifacts. It also contains the checksum
record for the final v21 source.

### Experiment History

- `docs/LOG.txt`
  - chronological notes from the earlier sequential optimization work

---

## Final Status

The sequential experimentation and benchmarking phase is complete.

The final portable sequential implementation is:

    src/project1_fileio_blas_adaptive_v21.c

Frozen SHA-256:

    d06dfafef6c9465cdfb86d0f75b7434b05614209c94ccdc6d67d27d79514769a

The final v21 validation record is:

    results/sequential-v21-adaptive-20261122.md

Frozen SHA-256:

    ef1cf2bfe1bf1ce97bee5d642167732eb89bcd06c3e33dc51a386746807bedd1

The prior v20 consolidated results remain preserved as the historical
checkpoint:

    results/sequential-final-20261121.md

Frozen SHA-256:

    001fda4a4849004c0b34744e1db66bb6ff9eacf9cf90d1d3d5025574060bb5fe

For portable builds, the default recommendation remains:

    -O3

When compiling directly for a known execution system, the tested
target-specific performance build is:

    -O3 -march=native -mtune=native

OpenBLAS remains single-threaded for the sequential implementation.
