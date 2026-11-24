# Final Sequential Performance Record

## Scope

This document consolidates the final sequential performance work for my
part of the COP 5522 term project.

The target computation is:

    Y = A X W

where:

- `A` is an `N x N` sparse matrix stored in CSR format
- `X` is an `N x F` dense matrix
- `W` is an `F x H` dense matrix
- `Y` is an `N x H` dense matrix

The two mathematically equivalent evaluation orders studied were:

    (AX)W
    A(XW)

The sequential work remained single-process and single-threaded.
MPI and the parallel implementation are handled separately by the team.

---

## Final Sequential Versions

### v15 - Portable Baseline

Source:

    Roly/src/project1_fileio_portable_v15.c

Association:

    (AX)W

This is the professor-style portable file-driven baseline.

### v16 - Optimized Sequential (AX)W

Source:

    Roly/src/project1_fileio_optimized_v16.c

Association:

    (AX)W

Main retained optimizations:

    FEATURE_BLOCK = 128
    PREFETCH_DISTANCE = 0
    dense traversal = i-k-j

### v17 - Optimized Sequential A(XW)

Source:

    Roly/src/project1_fileio_reassociated_v17.c

Association:

    A(XW)

This version keeps the optimized kernels from v16 but changes the
evaluation order.

### v18 - BLAS (AX)W

Source:

    Roly/src/project1_fileio_blas_v18.c

Association:

    (AX)W

The dense multiplication is performed with serial OpenBLAS SGEMM while
the sparse CSR kernel remains the same optimized sequential kernel.

### v19 - BLAS A(XW)

Source:

    Roly/src/project1_fileio_blas_reassociated_v19.c

Association:

    A(XW)

This version combines algebraic reassociation with serial OpenBLAS.

### v20 - Final Sequential Version

Source:

    Roly/src/project1_fileio_blas_reassociated_noinline_v20.c

Association:

    A(XW)

v20 differs from v19 by preventing GCC from inlining the CSR-dense
multiplication function:

    __attribute__((noinline))

This change was retained because controlled experiments showed that the
inlined version generated substantially worse execution behavior for
this workload on both tested architectures.

---

## Primary Benchmark

The primary regular benchmark uses:

    N = 4000
    F = 128
    H = 64
    nnz(A) = 64000
    nnz per row = 16

The regular sparse matrix therefore has exactly 16 nonzeros in every row.

For this shape:

    (AX)W nominal FLOPs = 81,920,000
    A(XW) nominal FLOPs = 73,728,000

The reassociated form performs 10% fewer nominal floating-point
operations because the sparse stage operates across `H=64` columns
instead of `F=128`.

Because the versions do not all execute the same number of nominal
floating-point operations, runtime is the primary comparison metric.

---

## Final Controlled Measurement Method

For the final six-version comparison:

- the same frozen source files were used
- the same frozen benchmark inputs were used
- GCC 8.5.0 was used on both target systems
- compiler optimization level was `-O3`
- file I/O was outside the timed multiplication region
- execution was single-process and single-threaded
- each version received three warm-up runs
- seven measured rounds were executed
- version order was rotated between rounds
- the median of seven measurements was used as the primary result
- correctness was checked independently from performance
- CPU affinity was controlled

For OpenBLAS:

### cs-ssh

Serial OpenBLAS 0.3.15 was used with:

    OPENBLAS_NUM_THREADS=1
    OPENBLAS_CORETYPE=SkylakeX

The explicit SkylakeX core type was used because OpenBLAS automatic
detection selected the much slower Prescott kernel on this system.

### Expanse

The serial OpenBLAS 0.3.15 library was used with:

    OPENBLAS_NUM_THREADS=1

The native Zen implementation was used. No Intel-specific core type was
forced on Expanse.

---

# Final Six-Version Results

## UWF cs-ssh

System:

    Intel Xeon Gold 6338N
    GCC 8.5.0

| Version | Association | Median runtime | GFLOP/s | Speedup vs v15 |
|---|---|---:|---:|---:|
| v15 | `(AX)W` | 32.329 ms | 2.534 | 1.000x |
| v16 | `(AX)W` | 7.303 ms | 11.218 | 4.427x |
| v17 | `A(XW)` | 5.209 ms | 14.155 | 6.207x |
| v18 | `(AX)W` + BLAS | 2.921 ms | 28.046 | 11.068x |
| v19 | `A(XW)` + BLAS | 3.231 ms | 22.822 | 10.007x |
| v20 | `A(XW)` + BLAS + noinline | **1.961 ms** | **37.597** | **16.486x** |

v20 reduced runtime relative to v15 by approximately:

    93.93%

v20 was also:

    3.724x faster than v16
    2.656x faster than v17
    1.489x faster than v18
    1.647x faster than v19

Compared with v19, v20 reduced runtime by approximately:

    39.30%

---

## SDSC Expanse

System:

    AMD EPYC 7742 64-Core Processor
    GCC 8.5.0

| Version | Association | Median runtime | GFLOP/s | Speedup vs v15 |
|---|---|---:|---:|---:|
| v15 | `(AX)W` | 31.861 ms | 2.571 | 1.000x |
| v16 | `(AX)W` | 6.891 ms | 11.888 | 4.624x |
| v17 | `A(XW)` | 5.492 ms | 13.424 | 5.801x |
| v18 | `(AX)W` + BLAS | 3.512 ms | 23.325 | 9.072x |
| v19 | `A(XW)` + BLAS | 2.899 ms | 25.430 | 10.990x |
| v20 | `A(XW)` + BLAS + noinline | **2.314 ms** | **31.855** | **13.766x** |

v20 reduced runtime relative to v15 by approximately:

    92.74%

v20 was also:

    2.977x faster than v16
    2.373x faster than v17
    1.517x faster than v18
    1.253x faster than v19

Compared with v19, v20 reduced runtime by approximately:

    20.17%

---

## Final Cross-Platform Result

The final sequential implementation combines:

- algebraic reassociation
- optimized sparse and dense kernels
- serial OpenBLAS
- prevention of detrimental CSR-kernel inlining

Relative to the portable v15 baseline, v20 achieved:

    cs-ssh: 16.486x speedup
    Expanse: 13.766x speedup

v20 was the fastest implementation in the final six-version comparison
on both architectures.

---

# Correctness

For the primary `N=4000, F=128, H=64` workload, all final versions passed
the numerical tolerance checks.

Tolerance:

    absolute = 1e-4
    relative = 1e-4

Representative maximum absolute differences relative to v15 were:

    v16: exact match
    v17: 6.11e-05
    v18: 2.29e-05 on cs-ssh
    v19/v20: 5.34e-05 on cs-ssh

All comparisons produced:

    tolerance failures = 0

Small differences are expected when BLAS or algebraic reassociation
changes the order of floating-point operations.

---

# Compiler Inlining Investigation

During the BLAS experiments, v19 initially appeared to have an unexpected
sparse-stage slowdown when the sparse multiply followed SGEMM.

This was investigated rather than being attributed directly to BLAS.

The investigation included:

- equal-work matrix shapes
- different OpenBLAS core types
- delay after SGEMM
- pre-touching the intermediate matrix
- hardware performance counters
- compiler vectorization reports
- source-level comparison of the CSR kernel
- `noinline`
- `noclone`
- combined `noinline,noclone`

The CSR kernel source itself was identical between the compared cases.

Hardware counters showed approximately 1.91x as many retired
instructions in the slower call-site context even though the same CSR
work was being performed.

Forcing the CSR function to remain out-of-line made the retired
instruction counts converge and substantially improved runtime.

## Attribute Isolation on cs-ssh

Approximate medians:

    original v19 = 3.344 ms
    noinline     = 2.350 ms
    noclone      = 3.340 ms
    both         = 2.354 ms

Results:

    noinline speedup vs original = 1.423x
    runtime reduction            = 29.73%

`noclone` alone had essentially no effect.

The combined `noinline,noclone` version was essentially the same as
`noinline` alone.

## Attribute Isolation on Expanse

Approximate medians:

    original = 2.787 ms
    noinline = 2.240 ms
    both     = 2.259 ms

Results:

    noinline speedup vs original = 1.244x
    runtime reduction            = 19.63%

The effect therefore transferred from Intel Xeon to AMD EPYC.

## Interpretation

The earlier apparent "stage-order penalty" should not be described as an
SGEMM state or cache penalty.

The strongest evidence indicates that GCC inlining and call-site code
generation were the primary reason for the observed difference.

The exact lower-level compiler mechanism beyond the inlining decision
was not isolated, so the safe conclusion is:

> Preventing GCC from inlining the CSR-dense kernel materially improved
> generated execution behavior for this workload on both tested
> architectures.

---

# Dimension Sensitivity

A second experiment reversed the dense dimensions:

    original:
        F = 128
        H = 64

    reverse shape:
        F = 64
        H = 128

For the reverse shape:

    v18 (AX)W FLOPs = 73,728,000
    v20 A(XW) FLOPs = 81,920,000

The dense SGEMM work remains the same, but the sparse stage changes
width.

## Reverse Shape on cs-ssh

    v18 (AX)W = 2.815 ms
    v20 A(XW) = 3.715 ms

Result:

    v18 was 24.25% faster
    v20/v18 runtime ratio = 1.320x

Correctness:

    max absolute difference = 1.91e-05
    tolerance failures = 0

## Reverse Shape on Expanse

    v18 (AX)W = 2.926 ms
    v20 A(XW) = 3.501 ms

Result:

    v18 was 16.42% faster
    v20/v18 runtime ratio = 1.196x

Correctness:

    max absolute difference = 1.91e-05
    tolerance failures = 0

## Dimension-Sensitivity Conclusion

The preferred evaluation order reversed on both tested architectures.

For the tested workloads:

> Performance improved when the sparse multiplication was applied to the
> narrower dense intermediate dimension.

For:

    F=128, H=64

`A(XW)` applies the sparse operation to 64 columns and is preferred.

For:

    F=64, H=128

`(AX)W` applies the sparse operation to 64 columns and is preferred.

This is an experimentally supported result for the tested workloads and
should not be presented as a universal theorem for every sparse matrix
or architecture.

---

# Synthetic Irregular Sparse Structure

The original benchmark is completely regular:

    mean row degree   = 16
    median row degree = 16
    minimum           = 16
    maximum           = 16
    standard deviation = 0

Two deterministic synthetic irregular matrices were also tested.

## A060

    rows = 4000
    columns = 4000
    nnz = 64000
    mean row degree = 16
    median = 10
    minimum = 7
    maximum = 955
    standard deviation = 28.304

## A075

    rows = 4000
    columns = 4000
    nnz = 64000
    mean row degree = 16
    median = 8
    minimum = 4
    maximum = 2256
    standard deviation = 55.019

These synthetic matrices change both the row-degree distribution and the
column-index structure. Therefore, the results should be described as
sensitivity to synthetic sparse-structure changes, not as an experiment
that isolates row irregularity alone.

---

## Irregular-Structure Results on cs-ssh

| Matrix | v15 | v18 | v20 | v20 speedup vs v15 |
|---|---:|---:|---:|---:|
| Regular | 32.403 ms | 2.879 ms | **2.176 ms** | 14.889x |
| A060 | 32.017 ms | 3.447 ms | **2.110 ms** | 15.172x |
| A075 | 32.177 ms | 3.387 ms | **2.083 ms** | 15.445x |

Effect relative to the regular matrix:

    v15 A060: -1.19%
    v15 A075: -0.70%

    v18 A060: +19.73%
    v18 A075: +17.67%

    v20 A060: -3.04%
    v20 A075: -4.28%

v20 compared with v18:

    Regular: 1.323x faster
    A060:    1.633x faster
    A075:    1.626x faster

---

## Irregular-Structure Results on Expanse

| Matrix | v15 | v18 | v20 | v20 speedup vs v15 |
|---|---:|---:|---:|---:|
| Regular | 31.824 ms | 3.499 ms | **2.156 ms** | 14.760x |
| A060 | 32.227 ms | 3.777 ms | **2.616 ms** | 12.319x |
| A075 | 32.234 ms | 3.758 ms | **2.600 ms** | 12.397x |

Effect relative to the regular matrix:

    v15 A060: +1.27%
    v15 A075: +1.29%

    v18 A060: +7.92%
    v18 A075: +7.39%

    v20 A060: +21.33%
    v20 A075: +20.60%

v20 compared with v18:

    Regular: 1.623x faster
    A060:    1.444x faster
    A075:    1.445x faster

---

## Sparse-Structure Conclusion

v20 remained the fastest tested sequential implementation on every
regular and synthetic irregular matrix on both architectures.

However, the response to sparse-structure changes was
architecture-dependent.

On cs-ssh:

- v18 slowed by approximately 18-20%
- v20 remained approximately unchanged and was slightly faster in these
  particular measurements

On Expanse:

- v18 slowed by approximately 7-8%
- v20 slowed by approximately 21%

Therefore, it would be incorrect to claim that v20 is universally less
sensitive to sparse irregularity.

The supported conclusion is:

> v20 retained the best overall runtime across all tested sparse
> structures, but the magnitude and direction of the performance effect
> from sparse-structure changes depended on the architecture.

---

# Main Sequential Findings

The final sequential experiments support the following conclusions.

1. Dense loop ordering was one of the largest early portable
   optimizations.

   Changing the row-major dense kernel from `i-j-k` to `i-k-j` improved
   contiguous memory traversal and allowed GCC to vectorize the inner
   loop more effectively.

2. Not every low-level optimization generalized.

   `restrict`, software prefetching, and sparse blocking choices produced
   mixed or architecture-dependent results.

3. Serial BLAS significantly improved the dense stage.

   OpenBLAS produced a major improvement over the hand-written dense
   multiplication, although correct architecture selection mattered on
   cs-ssh.

4. Algebraically equivalent evaluation orders are not performance
   equivalent.

   The preferred order depends on the dense dimensions because the
   association determines the width of the sparse stage.

5. Compiler decisions can materially affect performance.

   Preventing detrimental CSR-kernel inlining improved the final BLAS
   implementation on both Intel and AMD systems.

6. Sparse-structure sensitivity is architecture-dependent.

   The same synthetic sparse-structure changes affected Intel and AMD
   differently, even though v20 remained the fastest tested
   implementation in every case.

---

# Final Sequential Result

For the primary `N=4000, F=128, H=64` workload, the final sequential
version is:

    project1_fileio_blas_reassociated_noinline_v20.c

It achieved:

    Intel Xeon cs-ssh:
        1.961 ms
        37.597 GFLOP/s
        16.486x speedup over v15

    AMD EPYC Expanse:
        2.314 ms
        31.855 GFLOP/s
        13.766x speedup over v15

All tested correctness comparisons passed.

The strongest overall technical insight is:

> Algebraic equivalence does not imply performance equivalence.
> Evaluation order determines the width of the sparse stage, and compiler
> inlining can obscure that advantage. For the tested matrix shapes,
> applying the sparse multiplication to the narrower intermediate
> dimension produced the better evaluation order after controlling
> detrimental inlining.

---

# Reproducibility Artifacts

## Sources

    Roly/src/project1_fileio_portable_v15.c
    Roly/src/project1_fileio_optimized_v16.c
    Roly/src/project1_fileio_reassociated_v17.c
    Roly/src/project1_fileio_blas_v18.c
    Roly/src/project1_fileio_blas_reassociated_v19.c
    Roly/src/project1_fileio_blas_reassociated_noinline_v20.c

## cs-ssh Benchmark Script

    Roly/benchmarks/benchmark-irregular-cs-20261121.sh

## Expanse Evidence

    Roly/evidence/expanse-20261121/expanse-reverse-shape-55935085.sbatch
    Roly/evidence/expanse-20261121/expanse-reverse-55935085.out
    Roly/evidence/expanse-20261121/expanse-reverse-55935085.err

    Roly/evidence/expanse-20261121/expanse-irregular-structure-55935145.sbatch
    Roly/evidence/expanse-20261121/expanse-irregular-55935145.out
    Roly/evidence/expanse-20261121/expanse-irregular-55935145.err

    Roly/evidence/expanse-20261121/expanse-sequential-evidence-20261121.sha256

## Source Checksums

    Roly/checksums/project1_fileio_blas_v18.sha256
    Roly/checksums/project1_fileio_blas_reassociated_v19.sha256
    Roly/checksums/project1_fileio_blas_reassociated_noinline_v20.sha256
    Roly/checksums/benchmark-irregular-cs-20261121.sha256

Frozen source SHA-256 values:

    v18:
    46ba9344cd026ec3b10bfb767461cbff96fc7550b152bfa96e907e2217e27361

    v19:
    6c2d9a6a50c024a682cb92642534558183de654b2cb73b311c00b6a5a941dbdc

    v20:
    33d4dc6cab032946b9626c3971d1e57d2f331c4b53568745c2e30959ad038b7d

    cs-ssh irregular benchmark script:
    26ff612b53b1858eaede50d9b3ea1fdafd628ae9340d6443dd9ceefca86d186d

    Expanse evidence manifest:
    a146754018526d8498cb72057c62aadadc4b4200ff826f30678b3f4b0a392438
