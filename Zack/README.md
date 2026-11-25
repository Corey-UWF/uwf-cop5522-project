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

## Node-shared MPI implementation (`src/mpi_nodeshared.c`)

A single-file rewrite aimed at the bottlenecks in your Expanse phase data. For the
regular `N=40000` workload at 16 ranks on one node, the earlier versions spent
about 14 ms on a root-only `XW`, about 17 ms broadcasting the 10 MB intermediate
to every rank, and about 7 ms scattering/gathering, but only about 1 ms on local
sparse compute. The sequential OpenBLAS version (about 21-25 ms) beat the 16-rank runs.

### How it works

1. **One shared copy per node.** Ranks on a node share a single MPI-3
   shared-memory window holding `A`, `X`, `W`, the dense intermediate and `Y`.
   Nothing is copied per rank, so the memory-bandwidth-bound broadcast disappears.
2. **Rank 0 reads the input straight into that window**, so on one node there is
   no scatter. Each rank writes its rows of `Y` in place, so there is no gather.
3. **`XW` is computed in parallel.** Every rank on the root node multiplies its
   slice of the rows of `X` by `W` (OpenBLAS SGEMM, one thread per rank).
4. **Rows of `A` are split by estimated work** (nonzeros plus a per-row cost, in
   the spirit of the hybrid version), one contiguous block per rank.
5. **Only one "leader" rank per node uses the network.** With several nodes the
   leaders broadcast `A` and the dense input, and gather `Y` to rank 0.

The order is `A(XW)` when `H <= F`, otherwise `(AX)W`, as in sequential v21.

### Build and run

```sh
mpicc -O3 -Wall -Wextra -std=c11 -o mpi-nodeshared src/mpi_nodeshared.c -lopenblas -lm
export OPENBLAS_NUM_THREADS=1
mpirun -np 16 ./mpi-nodeshared A.csr X.dense W.dense Y.dense
```

`make mpi-nodeshared` does the same. It needs an MPI-3 library and standard
(32-bit) OpenBLAS, but not `cblas.h`. The report keeps the labels your existing
drivers parse (`CSR distribution`, `Root XW`, `Dense broadcast`,
`Local computation`, `Output gathering`, `MPI elapsed time`). Here `Root XW` means
"`XW` computed by the ranks of the root node", and the dense-broadcast and gather
phases are zero on one node. `MPI setup-inclusive time` adds the time spent before
the timer on creating the node communicators and the shared window (the maximum
over ranks), so it is the figure to use when comparing against versions that do
their setup differently. It also prints `Input reading` and `Output writing`.

### What the timer covers

The timer starts after rank 0 has read the files (as before) and ends after `Y` is
complete on rank 0. The earlier versions read into rank 0's private memory and then
distributed it inside the timer. This version reads directly into shared memory,
so that cost is not in `MPI elapsed time`. To avoid overstating the gain, the table
below also shows the hybrid time with its distribution and gather phases removed.
Input parsing and text output (about 4-8 s per run) are outside the timer and
dominate wall-clock time for every version, including this one.

### Tests

```sh
python3 tests/test_nodeshared.py --executable ./mpi-nodeshared \
    --reference ./sequential-v21 --mpiexec "mpirun --oversubscribe"
```

It compares against sequential v21 for both orders, empty rows, a dominant row,
more ranks than rows, zero nonzeros, a one-row matrix, a partial 128-column
block, and power-law-like rows, at 1, 2, 3, 4, 7 and 16 ranks. It also checks the
reported partition against an independent calculation and that bad input stops
every rank. Setting `NODESHARED_TEST_NODES=k` makes the program pretend there are
`k` nodes (ranks are dealt round-robin), which exercises the leader broadcast and
gather on one machine; the test runs 1 (real), 2 and 3 pretend nodes.

All 198 runs passed against both MPICH 4.0 and Open MPI 4.1.2 (Expanse uses Open
MPI 4.1.1). The largest absolute difference from v21 was `1.2e-4` (regular
`A(XW)` case), within the `1e-4` absolute plus relative tolerance. **Not yet
verified:** a real multi-node run over the network, and any run on Expanse.

### Local measurements (not Expanse)

Open MPI 4.1.2, OpenBLAS 0.3.20, a Ryzen 7 7435HS laptop (8 cores, 16 threads)
under WSL2, `N=40000`, `nnz=640000`, one warm-up and five repetitions, with
`scripts/compare_variants.py` checking every variant against sequential v21 first.
`regular` and `clustered` are the datasets from `generate_large_inputs.py`
(`clustered-strong`); `A(XW)` uses `F=128, H=64`, `(AX)W` uses `F=64, H=128`.
Medians of `MPI elapsed time` in ms at 8 ranks (one per physical core):

| Workload | Sequential v21 | Hybrid | Hybrid without distribute + gather | Node-shared | vs hybrid | vs stricter hybrid | vs sequential |
|---|---:|---:|---:|---:|---:|---:|---:|
| regular, A(XW) | 48.4 | 93.8 | 59.9 | 15.1 | 6.2x | 4.0x | 3.2x |
| clustered, A(XW) | 89.6 | 99.5 | 74.9 | 22.9 | 4.3x | 3.3x | 3.9x |
| regular, (AX)W | 66.6 | 97.0 | 45.5 | 15.9 | 6.1x | 2.9x | 4.2x |
| clustered, (AX)W | 84.2 | 128.2 | 77.9 | 20.6 | 6.2x | 3.8x | 4.1x |

Sequential times are compute only. At 1 rank the new version is about equal to
sequential (0.87-1.17x), as it should be, and the hybrid is 1.3-2.2x slower
than the new version. The 16-rank runs are oversubscribed (16 ranks on 8 physical cores) and
noisy, so they are not shown. This laptop has a different CPU, memory system and
BLAS from Expanse, so absolute times and scaling will differ. Re-measure on
Expanse before quoting any number in the report.

### Running the comparison on Expanse

`slurm/compare_nodeshared.sbatch` builds the hybrid and node-shared versions and
runs `scripts/compare_variants.py` (validation, warm-ups, 10 repetitions) on
1, 2 or 4 nodes: `sbatch --nodes=2 compare_nodeshared.sbatch`. It follows your
existing job scripts (flat upload directory, same modules) but has **not been
run**; check it before submitting.

### Limitations

- Every node holds the whole of `A` and the dense input, so memory per node does
  not shrink as nodes are added. `XW` is done only by the root node's ranks, and
  with several nodes the dense input must cross the network once per node.
- The ranks on a node must be able to share memory (MPI-3 windows); the
  `Y` rows of a node must fit one `MPI_Gatherv` (`N*H <= INT_MAX`).
- Reading and writing text files is serial on rank 0 and now dominates total run
  time. Parsing the files in parallel is the next place to look for savings.
