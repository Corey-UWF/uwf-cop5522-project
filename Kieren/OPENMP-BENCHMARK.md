# Kieren's OpenMP schedule experiment

`project_openmp_v21.c` is a **prototype**, derived from Rolando Veliz's
frozen v21 sequential implementation (source SHA-256 embedded at the top of
the file). It changes the CSR output-row loop to
`#pragma omp parallel for schedule(runtime)` and reports dense and sparse
stage times. It preserves v21's adaptive evaluation order and serial BLAS
SGEMM. This contribution should be reviewed with Corey before it becomes the
team's final OpenMP implementation.

`benchmark_openmp.py` generates matched regular and skewed CSR inputs with
the same total nonzeros and shared dense X and W. It compiles the unmodified
v21 sequential reference and this OpenMP prototype against the **same BLAS
library**. It checks every output using
`abs(openmp-reference) <= 1e-4 + 1e-4*abs(reference)`, then performs one warm-up
and five measured runs per thread count and schedule. It rotates execution
order and saves raw timings, stage times, input/source checksums, medians,
speedup versus the same schedule at one thread, and efficiency.

## Local pilot

The pilot ran on a shared virtual AMD EPYC 9V74 machine with GCC 14.2.0 and
the available system `libblas.so.3`, **not the OpenBLAS used by Rolando**.
Raw data is in `results/local-openmp-4000.json` and
`results/local-openmp-12000.json`.

| N | Pattern | Schedule | 1 thread | 2 threads | 4 threads | Sparse stage, 1 to 4 threads |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 12000 | Regular | Static | 37.904 ms | 37.680 ms | 36.818 ms | 3.753 to 2.155 ms |
| 12000 | Irregular | Static | 39.761 ms | 38.514 ms | 36.891 ms | 3.596 to 1.969 ms |
| 12000 | Irregular | Dynamic | 38.461 ms | 38.864 ms | 36.035 ms | 3.866 to 1.951 ms |
| 12000 | Irregular | Guided | 37.721 ms | 37.438 ms | 36.073 ms | 3.682 to 1.851 ms |

All 18 correctness configurations at N=12000 passed. The four-thread sparse
stage improved, but serial dense multiplication accounted for most total
runtime on this local BLAS library. These total-time differences are small
and noisy; do not claim that dynamic or guided is categorically best. The
4,000-row pilot also showed little end-to-end scaling.

To reproduce locally where BLAS is installed:

```sh
python3 Kieren/benchmark_openmp.py \
  --out /tmp/axw-openmp \
  --blas-library /absolute/path/to/libblas.so \
  --n 12000 --threads 1 2 4 --runs 5
```

To collect reportable course-machine evidence on Expanse, submit from the
repository root, replacing `YOUR_ACCOUNT` with your allocation:

```sh
sbatch --account=YOUR_ACCOUNT Kieren/expanse-openmp.sbatch
```

The job keeps OpenBLAS single-threaded and binds the OpenMP threads to cores.
It writes `Kieren/results/openmp-JOB_ID.json` and `Kieren/openmp-JOB_ID.out`.
Check all correctness failures and review each stage time before interpreting
the scaling result. The generated skew is synthetic and differs from
Rolando's frozen A060/A075 matrices.
