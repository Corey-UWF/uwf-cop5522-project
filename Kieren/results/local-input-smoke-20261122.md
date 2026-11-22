# Local input and sequential correctness smoke check

The benchmark generator produced two reproducible 4000 x 4000 CSR matrices
with 64,000 nonzeros each, and shared dense matrices X (4000 x 128) and W
(128 x 64). Two runs of `--generate-only` produced identical SHA-256 values:

| File | SHA-256 |
| --- | --- |
| regular-A.csr | c35536648d8c8c703f7a9194b44aa0197ba13f31d913712ce9be338d7ff16b4a |
| irregular-A.csr | 36858b28a3d9b3667aa920767f57ccb5eb57becdd9ff214a693560d56c8fe83d |
| X.dense | ada695a169178aa9edc438ed68060426b062f2393bbd6caf5b2533b1b896c8a6 |
| W.dense | 26fbceb587cdd034270b45974a0b79824f5b7a61250ea60da42800bd516f5d86 |

The irregular matrix has 80 empty rows, median degree 5, 90th percentile
degree 21, 99th percentile 180, and maximum degree 4000. The regular matrix
has 16 nonzeros in every row.

On the local virtual AMD EPYC 9V74 machine (GCC 14.2.0), the portable v15
program and adaptive v21 were compiled with `-O3`; v21 linked against the
available system `libblas.so.3` rather than the OpenBLAS used by Rolando.
Both programs processed both matrices. All 256,000 output values per matrix
passed `abs(v21-v15) <= 1e-4 + 1e-4*abs(v15)`. Maximum absolute difference
was 6.55e-7 on regular and 9.84e-7 on irregular input.

These were single smoke runs, **not benchmark timing results**. The local
environment has `mpirun` and `mpicc` commands but lacks the `mpi.h` development
header, so no MPI implementation was compiled or timed here. Use the Expanse
job scripts to collect the required MPI evidence.
