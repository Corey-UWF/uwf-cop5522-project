# Sequential v21 Adaptive Evaluation-Order Results

## Purpose

This record documents the final sequential optimization stage after the
v20 checkpoint.

The previous consolidated v15-v20 results remain preserved in:

    results/sequential-final-20261121.md

That earlier record is not replaced by this file. It remains the frozen
checkpoint for the experiments that led to v20.

This document records the additional work that produced and validated
v21.

---

## Final Portable Sequential Source

Source:

    src/project1_fileio_blas_adaptive_v21.c

SHA-256:

    d06dfafef6c9465cdfb86d0f75b7434b05614209c94ccdc6d67d27d79514769a

v21 preserves the optimized CSR-dense kernel, serial OpenBLAS dense
multiplication, and the `noinline` CSR-kernel decision established during
the v19/v20 compiler investigation.

The new contribution in v21 is adaptive evaluation-order selection.

---

## Adaptive Evaluation Order

The computation is:

    Y = A X W

Two mathematically equivalent evaluation orders are available:

    (AX)W

and:

    A(XW)

The dense multiplication performs the same nominal dense arithmetic in
either order:

    2 * N * F * H

The difference is the width used by the sparse matrix multiplication.

For `(AX)W`, the sparse stage has width:

    F

For `A(XW)`, the sparse stage has width:

    H

v21 therefore uses the following rule:

    if H <= F:
        use A(XW)
    else:
        use (AX)W

This applies the sparse multiplication to the narrower dense width for
the two tested dimension regimes.

This is an experimentally supported optimization rule for the tested
workloads. It is not claimed as a universal performance theorem for all
matrices, architectures, compilers, or sparse structures.

---

## Tested Shapes

### Primary shape

    N = 4000
    nnz(A) = 64000
    F = 128
    H = 64

v21 selects:

    A(XW)

### Reverse shape

    N = 4000
    nnz(A) = 64000
    F = 64
    H = 128

v21 selects:

    (AX)W

The reverse-shape inputs were derived deterministically and frozen before
the v21 experiment.

---

## Correctness

### UWF cs-ssh

For the primary shape, v21 selected the same `A(XW)` path as v20.

The v21 output was byte-for-byte identical to the v20 output.

For the reverse shape, v21 selected the same `(AX)W` path as v18.

The v21 output was byte-for-byte identical to the v18 output.

Comparisons against the portable v15 baseline used:

    absolute tolerance = 1e-4
    relative tolerance = 1e-4

All tested comparisons reported:

    failures = 0

### SDSC Expanse

The final Expanse job also verified both adaptive paths.

For the primary shape:

    v21-base   == v20 output
    v21-native == v20 output

For the reverse shape:

    v21-base   == v18 output
    v21-native == v18 output

All numerical comparisons against v15 reported:

    failures = 0

Therefore, the adaptive branch selection and the native compiler build
preserved the expected numerical results on both Intel and AMD systems.

---

## UWF cs-ssh Validation

System:

    Intel Xeon Gold 6338N
    GCC 8.5.0

Serial OpenBLAS was used with:

    OPENBLAS_NUM_THREADS=1
    OPENBLAS_CORETYPE=SkylakeX

CPU execution was pinned to one CPU during the controlled measurements.

### Adaptive v21 Follow-Up

A 21-round paired test was used after a shorter seven-round experiment
showed normal run-to-run variability.

#### Primary shape

Comparison:

    v20 fixed A(XW)
    vs
    v21 adaptive

Median runtimes:

    v20 = 2082.700 us
    v21 = 2007.910 us

Observed median-runtime change:

    v21 = 3.59% lower

Paired result:

    v21 wins  = 12
    v21 losses = 9

Median paired speedup:

    1.0187x

The longer paired experiment did not reproduce the small apparent v21
regression seen in the earlier seven-round run.

#### Reverse shape

Comparison:

    v18 fixed (AX)W
    vs
    v21 adaptive

Median runtimes:

    v18 = 2358.830 us
    v21 = 2289.680 us

Observed median-runtime change:

    v21 = 2.93% lower

Paired result:

    v21 wins  = 17
    v21 losses = 4

Median paired speedup:

    1.0432x

This confirmed that the adaptive implementation could select the
appropriate association without introducing a meaningful performance
penalty.

---

## cs-ssh Native Compiler Experiment

The exact v21 source was compiled two ways:

Portable build:

    -O3

Native build:

    -O3 -march=native -mtune=native

GCC selected:

    -march=icelake-server
    -mtune=icelake-server

The native target enabled the applicable AVX, AVX2, AVX-512, and FMA
features for the cs-ssh processor.

The source file itself was not changed.

### Primary shape

Median runtimes:

    portable v21 = 2097.180 us
    native v21   = 1889.250 us

Native speedup:

    1.1101x

Runtime reduction:

    9.91%

Paired result:

    native wins   = 18
    native losses = 3

### Reverse shape

Median runtimes:

    portable v21 = 2270.740 us
    native v21   = 2150.420 us

Native speedup:

    1.0560x

Runtime reduction:

    5.30%

Paired result:

    native wins   = 16
    native losses = 5

The native compiler flags were therefore beneficial on the tested Intel
system.

---

## SDSC Expanse Final Validation

Final job:

    55942327

Compute node:

    exp-13-09

Processor:

    AMD EPYC 7742 64-Core Processor

Compiler:

    GCC 8.5.0

OpenBLAS:

    /lib64/libopenblas.so.0
    OpenBLAS 0.3.15

OpenBLAS was kept single-threaded.

The final job used:

    one task
    one CPU per task
    explicit CPU binding

The experiment used three warmups per version and shape followed by 21
measured rounds with rotated execution order.

---

## Expanse Median Results

### Primary shape: F=128, H=64

| Version | Median runtime |
|---|---:|
| v15 portable baseline | 31669.700 us |
| v18 BLAS `(AX)W` | 3337.010 us |
| v20 BLAS `A(XW)` + noinline | 1946.710 us |
| v21 adaptive portable | 1968.070 us |
| v21 adaptive native | **1898.400 us** |

### Adaptive comparison

Comparison:

    v20 -> v21-base

Median-runtime ratio:

    0.9891x

Median runtime difference:

    v21-base = 1.10% higher

Paired median:

    1.0021x

Paired result:

    v21-base wins   = 11
    v21-base losses = 10

The paired result is effectively neutral. For the primary shape, portable
v21 and v20 performed approximately the same while executing the same
`A(XW)` association.

### Native compiler comparison

Comparison:

    v21-base -> v21-native

Native speedup:

    1.0367x

Runtime reduction:

    3.54%

Paired result:

    native wins   = 14
    native losses = 7

### Final speedup over v15

    16.6823x

Runtime reduction:

    94.01%

Paired result:

    v21-native wins = 21 of 21

---

## Expanse Reverse Shape: F=64, H=128

| Version | Median runtime |
|---|---:|
| v15 portable baseline | 29922.400 us |
| v18 BLAS `(AX)W` | 2659.690 us |
| v20 BLAS `A(XW)` + noinline | 3044.070 us |
| v21 adaptive portable | 2125.770 us |
| v21 adaptive native | **2067.470 us** |

### Adaptive comparison

The cleanest evaluation-order comparison uses v20 because both v20 and
v21 retain the previously established `noinline` CSR-kernel decision.

Comparison:

    v20 -> v21-base

Adaptive speedup:

    1.4320x

Runtime reduction:

    30.17%

Paired median:

    1.4256x

Paired result:

    v21-base wins   = 21
    v21-base losses = 0

This is the strongest direct evidence for the adaptive evaluation-order
decision. When F and H were reversed, v21 changed from the v20
`A(XW)` association to `(AX)W` while retaining the same `noinline`
CSR-kernel policy.

For additional context, v21-base also outperformed v18:

    v18 -> v21-base speedup = 1.2512x
    runtime reduction       = 20.07%
    paired median speedup   = 1.2557x
    v21-base wins/losses    = 21/0

### Native compiler comparison

Comparison:

    v21-base -> v21-native

Native speedup:

    1.0282x

Runtime reduction:

    2.74%

Paired result:

    native wins   = 17
    native losses = 4

### Final speedup over v15

    14.4730x

Runtime reduction:

    93.09%

Paired result:

    v21-native wins = 21 of 21

---

## Expanse Native Compiler Target

For the AMD EPYC 7742 compute node, GCC selected:

    -march=znver1
    -mtune=znver1

The resulting target enabled AVX, AVX2, and FMA while leaving AVX-512
disabled, as expected for this processor generation.

Native compilation improved both tested shapes on Expanse.

This independently supports using:

    -O3 -march=native -mtune=native

as an optional target-specific build optimization when the program is
compiled directly for the execution system.

The portable source itself does not depend on those flags.

---

## Final Interpretation

The v21 experiments strengthen the main conclusion from the sequential
work:

> Algebraic equivalence does not imply performance equivalence.

The order of evaluation changes the width processed by the sparse stage.
For the tested dimension regimes, selecting the association that gives
the sparse multiplication the narrower dense width produced the better
or equivalent sequential result after controlling detrimental CSR-kernel
inlining.

The experiments also show that compiler behavior remains important.

The earlier v19/v20 investigation showed that preventing detrimental CSR
inlining materially improved performance on both Intel and AMD.

The v21 compiler experiment then showed that target-specific native
compilation produced additional measurable improvements on both
architectures.

These optimizations address different levels of the implementation:

    evaluation order:
        chooses which mathematically equivalent computation to execute

    noinline:
        controls an important compiler code-generation decision for the
        sparse kernel

    OpenBLAS:
        provides optimized dense multiplication

    -march=native -mtune=native:
        permits target-specific compiler optimization for the surrounding
        sequential code

---

## Recommended Build Strategy

### Portable build

Use:

    -O3

This preserves portability across compatible systems.

### Target-specific performance build

When compiling directly on the execution system, use:

    -O3 -march=native -mtune=native

This should be treated as a platform-specific build optimization rather
than a requirement of the portable v21 source.

OpenBLAS should remain single-threaded when measuring the sequential
implementation.

---

## Preserved Expanse Evidence

The final Expanse evidence is stored under:

    evidence/expanse-v21-20261122/

Files:

    expanse-v21-final-validation.sbatch
    expanse-v21-final-55942327.out
    expanse-v21-final-55942327.err
    expanse-v21-final-55942327.csv
    expanse-v21-final-55942327.sha256

The `.err` file is empty.

The SHA-256 manifest verifies the preserved job script, output, error
file, and raw CSV.

---

## Final Sequential Status

The sequential experimentation phase is complete.

The final portable sequential implementation is:

    src/project1_fileio_blas_adaptive_v21.c

SHA-256:

    d06dfafef6c9465cdfb86d0f75b7434b05614209c94ccdc6d67d27d79514769a

v20 remains preserved as the prior optimization checkpoint and as the
source of the earlier compiler-inlining and fixed-association evidence.

No additional sequential optimization experiments are required for the
current project scope.
