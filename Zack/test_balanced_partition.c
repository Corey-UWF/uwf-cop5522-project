#define main balanced_program_main
#include "mpi_balanced.c"
#undef main

#include <assert.h>

static void check_partition(int n, int *pointers, int ranks, const int *expected)
{
    CsrMatrix a = {n, pointers[n], pointers, NULL, NULL};
    int *rows = allocate_array((size_t)ranks, sizeof(*rows));
    int *offsets = allocate_array((size_t)ranks, sizeof(*offsets));
    int *counts = allocate_array((size_t)ranks, sizeof(*counts));
    int *displacements = allocate_array((size_t)ranks, sizeof(*displacements));
    int *output_counts = allocate_array((size_t)ranks, sizeof(*output_counts));
    int *output_offsets = allocate_array((size_t)ranks, sizeof(*output_offsets));
    partition_rows(&a, ranks, 1, rows, offsets, counts, displacements,
                   output_counts, output_offsets);
    int total_rows = 0;
    int total_nnz = 0;
    for (int p = 0; p < ranks; ++p) {
        int end = p + 1 < ranks ? offsets[p + 1] : n;
        assert(offsets[p] == total_rows);
        assert(offsets[p] == expected[p]);
        assert(end >= offsets[p] && end <= n);
        assert(rows[p] == end - offsets[p]);
        assert(counts[p] == pointers[end] - pointers[offsets[p]]);
        assert(displacements[p] == pointers[offsets[p]]);
        assert(output_counts[p] == rows[p]);
        assert(output_offsets[p] == offsets[p]);
        total_rows += rows[p];
        total_nnz += counts[p];
    }
    assert(total_rows == n && total_nnz == a.nnz);
    free(rows);
    free(offsets);
    free(counts);
    free(displacements);
    free(output_counts);
    free(output_offsets);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int regular[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    int regular_expected[] = {0, 2, 4, 6};
    check_partition(8, regular, 4, regular_expected);

    int clustered[] = {0, 8, 16, 24, 32, 32, 32, 32, 32};
    int clustered_expected[] = {0, 1, 2, 3};
    check_partition(8, clustered, 4, clustered_expected);

    int empty_rows[] = {0, 0, 0, 4, 4, 8, 8};
    int empty_expected[] = {0, 2, 3, 4};
    check_partition(6, empty_rows, 4, empty_expected);

    int dominant[] = {0, 1, 101, 102, 103};
    int dominant_expected[] = {0, 1, 2, 2};
    check_partition(4, dominant, 4, dominant_expected);

    int zero[] = {0, 0, 0, 0, 0, 0};
    int zero_expected[] = {0, 1, 2, 3, 4, 5, 5, 5};
    check_partition(5, zero, 8, zero_expected);

    int excess_expected[] = {0, 0, 1, 1, 1, 2, 2, 3};
    int short_matrix[] = {0, 1, 2, 3};
    check_partition(3, short_matrix, 8, excess_expected);

    int large[] = {0, 1, INT_MAX - 1, INT_MAX};
    int large_expected[] = {0, 1, 1, 2};
    check_partition(3, large, 4, large_expected);
    CsrMatrix large_matrix = {3, INT_MAX, large, NULL, NULL};
    assert(nonzero_boundary(&large_matrix, INT_MAX - 1, INT_MAX) == 2);

    int one_rank[] = {0};
    check_partition(4, dominant, 1, one_rank);
    puts("PASS: regular, clustered, empty-row, dominant-row, zero, excess-rank, integer-limit and one-rank partitions");
    MPI_Finalize();
    return EXIT_SUCCESS;
}
