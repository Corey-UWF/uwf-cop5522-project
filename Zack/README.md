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

## Adaptive MPI implementation

`mpi_adaptive.c` is a separate equal-row optimized version; `mpi_baseline.c` is
unchanged. It adapts Rolando's v21 sparse kernel (128-column feature blocks,
no software prefetching, and a guarded `noinline` attribute) and uses OpenBLAS
`cblas_sgemm` for dense multiplication.

- When `H <= F`, rank 0 computes `XW`, broadcasts that `N x H` intermediate,
  and each rank computes its local sparse multiplication.
- When `H > F`, all ranks receive `X` and `W`, compute local `AX`, and use
  SGEMM for their local output rows.
- Both paths gather the output on rank 0 and use the same four-file interface.
- OpenBLAS is explicitly restricted to one thread per rank in the program.
  Also export `OPENBLAS_NUM_THREADS=1` for the sequential comparison.

The dimension rule matches v21's arithmetic heuristic. It does not guarantee
the fastest distributed execution: the root-only `XW` computation is serial
across MPI ranks and may limit scaling.

### Build and run on ACCESS

Upload `mpi_adaptive.c` and the updated Makefile into your remote Zack directory.
Load the course's matching compiler/MPI modules and an OpenBLAS development
module. Discover the OpenBLAS module with `module avail openblas`; its exact
name and include/library paths depend on the machine.

```sh
make mpi-adaptive
export OPENBLAS_NUM_THREADS=1
mpirun -np 4 ./mpi-adaptive A.csr X.dense W.dense Y-adaptive.dense
```

Run MPI inside a compute allocation. `make` still builds the original baseline
and generator without requiring OpenBLAS; the adaptive target is explicit.
If the loaded OpenBLAS module does not automatically expose its paths, use:

```sh
make mpi-adaptive BLAS_CPPFLAGS="-I/actual/openblas/include" BLAS_LDFLAGS="-L/actual/openblas/lib"
```

Substitute the paths supplied by that module, not the literal example paths.
This version requires `cblas.h` and standard LP64 OpenBLAS (32-bit BLAS
dimensions), not an ILP64 library. Dimensions and MPI message counts are
limited to `INT_MAX`.

### Timing and validation

The output reports the chosen order, OpenBLAS thread count, and seconds for:

1. CSR distribution, including local CSR allocation and reconstruction.
2. Root-only `XW` SGEMM (zero for the `(AX)W` path).
3. Dense broadcasts.
4. Local sparse and, where applicable, dense kernels.
5. Output gathering.
6. Total distributed elapsed time.

Each reported time is the maximum across ranks. A barrier before dense
broadcast excludes waiting for root SGEMM from that communication phase.
A barrier before gathering excludes waiting for other ranks' kernels from
the gathering phase. Total time includes allocation and those barriers;
it starts before CSR distribution and ends after gathering, excluding file
I/O, initial input/dimension exchange, partition setup, and timing reductions.
Phase maxima need not sum to total time. These are instrumented measurements,
not a claim of pure communication cost or an exact compute-only comparison
with sequential v21.

Local WSL validation with MPICH 4.0 and OpenBLAS 0.3.20 passed comparisons
against v21 and an independently calculated result at absolute and relative
tolerances of `1e-4`. Cases covered `H < F`, `H > F`, `H = F`, a partial
128-column block, empty rows, and an all-zero CSR matrix at 1, 2, and 8 ranks,
including ranks with zero rows. Malformed CSR pointers were also rejected.
This is correctness evidence, not ACCESS performance or multi-node validation.
Before benchmarking, repeat numerical comparisons on your ACCESS datasets;
BLAS and reassociation can change floating-point rounding.

## Nonzero-balanced MPI implementation

`mpi_balanced.c` is a separate version derived from `mpi_adaptive.c`. Neither
the adaptive nor baseline source is modified. The sparse kernel, OpenBLAS
SGEMM, single-thread configuration, `H <= F` decision, dense communication,
matrix output format, and existing phase timing labels are preserved.

### Row partitioning

Rank 0 computes contiguous boundaries for cumulative targets
`p * total_nnz / ranks`, for `p=1,...,ranks-1`. A binary search finds the
first CSR pointer at or above each target. The nearer of that boundary and
the preceding boundary is selected; equal distances select the preceding
boundary. Targets are compared as scaled 64-bit integers, so fractional
targets are retained without floating-point or integer-division rounding.

Duplicate pointers and repeated boundaries are valid: they can assign empty
rows or no rows to a rank. The first boundary is zero, the final boundary is
`N`, and boundaries are nondecreasing. Entire CSR rows are preserved. When
all nonzero counts are zero, the algorithm instead distributes rows evenly,
including the case with more ranks than rows.

Rank 0 derives each rank's row count/offset, nonzero count/offset, and output
count/offset from those boundaries, then broadcasts all six metadata arrays
before distribution. Local pointers are reconstructed from row lengths and
checked to start at zero and end at the local nonzero count. Input and output
products are constrained to `INT_MAX` so MPI counts and displacements fit
32-bit integers. Partitioning costs `O(P log N)` and uses `O(P)` metadata.

### Build on cs-ssh or Expanse

The balanced version uses explicit LP64 CBLAS/OpenBLAS declarations adapted
from sequential v21; it does **not** need `cblas.h`. It requires a standard
32-bit-dimension OpenBLAS library, not ILP64. Load the matching MPI/compiler
environment and make OpenBLAS available. For the stated cs-ssh environment:

```sh
module load mpi/openmpi-x86_64
mpicc -O3 -Wall -Wextra -std=c11 -o mpi-balanced mpi_balanced.c -lopenblas -lm
```

Alternatively, use `make mpi-balanced`. On Expanse use the course's appropriate
compiler, MPI, and OpenBLAS modules; if needed pass
`BLAS_LDFLAGS="-L/actual/openblas/lib"` to make. The original default `make`
targets are unchanged.

Inside your allocated execution environment:

```sh
export OPENBLAS_NUM_THREADS=1
mpirun -np 4 ./mpi-balanced A.csr X.dense W.dense Y-balanced.dense
```

### Diagnostics and controlled comparison

Rank 0 prints total rows/nonzeros/ranks, minimum/maximum assigned rows,
minimum/maximum/average assigned nonzeros, and
`max_nnz / average_nnz`. The ratio is explicitly `N/A` for a zero-nonzero
matrix because division by zero is undefined, not evidence of imbalance.
The summary also reports minimum, maximum, and mean rank-local kernel times
using reductions; its maximum equals the original `Local computation` time.
These times exclude root-only `XW`, which retains its own phase.

Diagnostics go to stdout and never into the matrix file. Partition setup,
metadata broadcasts, diagnostic reductions, and summary printing are outside
the existing total timer. CSR reconstruction checks are inside distribution.
Phase maxima need not add up to total time; allocation and instrumentation
barriers remain included as described for the adaptive implementation.

Use identical files, rank placement, compiler flags, BLAS environment,
warm-ups, and repetitions when comparing adaptive and balanced versions.
Preserve input checksums and report partitioning/setup exclusions. This code
does not establish a performance improvement.

### Correctness tests

Build the focused partition test and run it as a single MPI process:

```sh
make test-balanced-partition
mpirun -np 1 ./test-balanced-partition
```

It checks contiguous coverage, counts/displacements, deterministic tie handling,
duplicate pointers, a dominant row, zero nonzeros, excess ranks, one rank,
and cumulative targets near `INT_MAX` without allocating enormous matrices.
Do not compile this assertion-based test with `-DNDEBUG`.

For full numerical testing, build `mpi-adaptive` in its working environment
and Rolando's sequential v21 executable. Then run:

```sh
python3 test_balanced_mpi.py --reference /actual/path/to/sequential-v21
```

The script creates temporary deterministic inputs, runs both MPI versions and
v21, and removes those inputs after completion. It defaults to 1, 2, and 4 ranks.
Use an allocation with enough slots, or explicitly change `--ranks 1 2`;
`--mpiexec`, `--balanced`, and `--adaptive` can select alternate executable paths.
The test-generated data is not claimed to reproduce Rolando's frozen inputs.

Cases include both `N=4000, nnz=64000` dimension configurations, empty rows,
a dominant high-degree row, more ranks than rows, zero nonzeros, a partial
128-column block, and strongly irregular power-law-style row counts for both
orders. It checks file shapes, finite results, partition summaries, timing
fields, and numerical agreement using
`abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`.

Local WSL verification with MPICH 4.0 and OpenBLAS 0.3.20 passed the partition
test and all 27 numerical configurations. Primary outputs matched v21 exactly;
the largest reverse-case absolute difference was `6.1e-05`, within tolerance.
Repeat validation against your actual cs-ssh/Expanse inputs and record errors;
bitwise agreement can vary with BLAS kernel selection and local GEMM shapes.

### Limitations

- An indivisible high-degree row can prevent good balance and leave ranks idle.
  Its nonzeros must stay on one rank; the reported ratio reveals this limit.
- Empty-row placement is deterministic but does not balance row counts. For
  `(AX)W`, dense work depends on local row count as well as sparse nonzeros;
  nonzero balance therefore does not guarantee computation-time balance.
- `A(XW)` still computes `XW` only on rank 0. Replicated dense data, broadcasts,
  gathering, synchronization, and root memory requirements can limit scaling.
- Matrices/messages beyond the current `INT_MAX` limits are rejected. There is
  no row splitting, asynchronous scheduling, or alternate sparse format.
- The local tests are correctness evidence only. ACCESS performance and
  multi-node scalability require separate controlled experiments.
