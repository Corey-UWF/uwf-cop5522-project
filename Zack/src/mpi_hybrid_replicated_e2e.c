#include <mpi.h>
#include <time.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LP64 declarations, as in Rolando's v21, for systems without cblas.h. */
enum { CblasRowMajor = 101, CblasNoTrans = 111 };
extern void cblas_sgemm(const int order, const int trans_a, const int trans_b,
                        const int m, const int n, const int k,
                        const float alpha, const float *a, const int lda,
                        const float *b, const int ldb, const float beta,
                        float *c, const int ldc);
extern void openblas_set_num_threads(int threads);
extern int openblas_get_num_threads(void);

enum { FEATURE_BLOCK = 128 };

typedef struct {
    int n;
    int nnz;
    int *row_ptr;
    int *col_idx;
    float *values;
} CsrMatrix;

typedef struct {
    int rows;
    int cols;
    float *values;
} DenseMatrix;

static void fail(const char *message)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static void *allocate_array(size_t count, size_t size)
{
    if (count > SIZE_MAX / size) {
        fail("Allocation size overflow");
    }
    void *memory = calloc(count == 0 ? 1 : count, size);
    if (memory == NULL) {
        fail("Insufficient memory");
    }
    return memory;
}

static FILE *open_file(const char *path, const char *mode)
{
    FILE *file = fopen(path, mode);
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s: ", path);
        perror(NULL);
        fail("File open failed");
    }
    return file;
}

static CsrMatrix load_csr(const char *path)
{
    FILE *file = open_file(path, "r");
    CsrMatrix a = {0};
    int cols;
    if (fscanf(file, "%d %d %d", &a.n, &cols, &a.nnz) != 3 ||
        a.n <= 0 || cols != a.n || a.nnz < 0) {
        fail("Invalid CSR header (expected rows columns nnz; A must be square)");
    }
    a.row_ptr = allocate_array((size_t)a.n + 1, sizeof(*a.row_ptr));
    a.col_idx = allocate_array((size_t)a.nnz, sizeof(*a.col_idx));
    a.values = allocate_array((size_t)a.nnz, sizeof(*a.values));
    for (size_t i = 0; i <= (size_t)a.n; ++i) {
        if (fscanf(file, "%d", &a.row_ptr[i]) != 1 ||
            a.row_ptr[i] < 0 || a.row_ptr[i] > a.nnz ||
            (i > 0 && a.row_ptr[i] < a.row_ptr[i - 1])) {
            fail("Invalid CSR row pointers");
        }
    }
    if (a.row_ptr[0] != 0 || a.row_ptr[a.n] != a.nnz) {
        fail("CSR row pointers must start at zero and end at nnz");
    }
    for (int i = 0; i < a.nnz; ++i) {
        if (fscanf(file, "%d", &a.col_idx[i]) != 1 ||
            a.col_idx[i] < 0 || a.col_idx[i] >= a.n) {
            fail("Invalid CSR column index");
        }
    }
    for (int i = 0; i < a.nnz; ++i) {
        if (fscanf(file, "%f", &a.values[i]) != 1) {
            fail("Invalid CSR value");
        }
    }
    fclose(file);
    return a;
}

static DenseMatrix load_dense(const char *path)
{
    FILE *file = open_file(path, "r");
    DenseMatrix matrix = {0};
    if (fscanf(file, "%d %d", &matrix.rows, &matrix.cols) != 2 ||
        matrix.rows <= 0 || matrix.cols <= 0 ||
        (size_t)matrix.rows > (size_t)INT_MAX / (size_t)matrix.cols) {
        fail("Invalid dense dimensions or matrix exceeds MPI count limit");
    }
    size_t count = (size_t)matrix.rows * (size_t)matrix.cols;
    matrix.values = allocate_array(count, sizeof(*matrix.values));
    for (size_t i = 0; i < count; ++i) {
        if (fscanf(file, "%f", &matrix.values[i]) != 1) {
            fail("Invalid dense value");
        }
    }
    fclose(file);
    return matrix;
}

static void write_dense(const char *path, const DenseMatrix *matrix)
{
    FILE *file = open_file(path, "w");
    fprintf(file, "%d %d\n", matrix->rows, matrix->cols);
    for (int row = 0; row < matrix->rows; ++row) {
        for (int col = 0; col < matrix->cols; ++col) {
            fprintf(file, "%.9g%c",
                    matrix->values[(size_t)row * (size_t)matrix->cols + (size_t)col],
                    col + 1 == matrix->cols ? '\n' : ' ');
        }
    }
    int failed = ferror(file);
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        fail("Output write failed");
    }
}

/* Adapted from Rolando's v21: 128-wide blocks, no prefetching. */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void csr_dense_multiply(const CsrMatrix *a, int local_rows,
                               const DenseMatrix *x, DenseMatrix *output)
{
    memset(output->values, 0,
           (size_t)local_rows * (size_t)x->cols * sizeof(float));
    for (int row = 0; row < local_rows; ++row) {
        for (int block = 0; block < x->cols;) {
            int end = block + (x->cols - block < FEATURE_BLOCK ?
                               x->cols - block : FEATURE_BLOCK);
            for (int p = a->row_ptr[row]; p < a->row_ptr[row + 1]; ++p) {
                float *out_row = output->values + (size_t)row * (size_t)x->cols;
                const float *x_row = x->values +
                    (size_t)a->col_idx[p] * (size_t)x->cols;
                const float value = a->values[p];
                for (int j = block; j < end; ++j) {
                    out_row[j] += value * x_row[j];
                }
            }
            block = end;
        }
    }
}

static void dense_multiply(const DenseMatrix *a, const DenseMatrix *b,
                           DenseMatrix *output)
{
    if (a->cols != b->rows || output->rows != a->rows ||
        output->cols != b->cols) {
        fail("Dense multiplication dimension mismatch");
    }
    if (a->rows == 0) {
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                a->rows, b->cols, a->cols, 1.0f,
                a->values, a->cols, b->values, b->cols,
                0.0f, output->values, output->cols);
}

static void free_csr(CsrMatrix *matrix)
{
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
}

/* Hybrid cost uses one unit per CSR nonzero plus row_weight units per row.
 * For (AX)W, default row_weight=H/2 (a tunable heuristic, not a fitted
 * performance model). For A(XW), default row_weight=0 to preserve
 * nonzero balancing. A command-line-independent override can be supplied
 * using HYBRID_ROW_WEIGHT (nonnegative integer).
 */
static int hybrid_row_weight(int hidden, int use_xw_first)
{
    const char *override = getenv("HYBRID_ROW_WEIGHT");
    if (override != NULL && *override != '\0') {
        char *endptr = NULL;
        long long value = strtoll(override, &endptr, 10);
        if (endptr == override || *endptr != '\0' || value < 0 || value > INT_MAX) {
            fail("HYBRID_ROW_WEIGHT must be a nonnegative integer <= INT_MAX");
        }
        return (int)value;
    }
    return use_xw_first ? 0 : hidden / 2;
}

static int64_t cumulative_work(const CsrMatrix *a, int boundary, int row_weight)
{
    return (int64_t)a->row_ptr[boundary] + (int64_t)row_weight * boundary;
}

static int hybrid_boundary(const CsrMatrix *a, int partition, int ranks,
                           int row_weight)
{
    const int64_t total = cumulative_work(a, a->n, row_weight);
    /* Multiplication before division uses long double to avoid int64 overflow. */
    const long double target = (long double)total * partition / ranks;
    int low = 0;
    int high = a->n;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if ((long double)cumulative_work(a, mid, row_weight) < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low > 0 &&
        target - (long double)cumulative_work(a, low - 1, row_weight) <=
        (long double)cumulative_work(a, low, row_weight) - target) {
        --low;
    }
    return low;
}

static void partition_rows(const CsrMatrix *a, int ranks, int hidden,
                           int row_weight, int *rows, int *offsets,
                           int *nnz_counts, int *nnz_offsets,
                           int *output_counts, int *output_offsets)
{
    if (ranks <= 0 || hidden <= 0 ||
        (size_t)a->n > (size_t)INT_MAX / (size_t)hidden) {
        fail("Invalid partition dimensions or output exceeds MPI count limit");
    }
    int start = 0;
    for (int p = 0; p < ranks; ++p) {
        int end = p == ranks - 1 ? a->n :
            hybrid_boundary(a, p + 1, ranks, row_weight);
        if (end < start || end > a->n) {
            fail("Invalid hybrid partition row boundaries");
        }
        rows[p] = end - start;
        offsets[p] = start;
        nnz_offsets[p] = a->row_ptr[start];
        nnz_counts[p] = a->row_ptr[end] - nnz_offsets[p];
        output_counts[p] = rows[p] * hidden;
        output_offsets[p] = start * hidden;
        start = end;
    }
}

static void print_partition_summary(const CsrMatrix *a, int ranks, int row_weight,
                                    const int *rows, const int *nnz_counts)
{
    int min_rows = rows[0];
    int max_rows = rows[0];
    int min_nnz = nnz_counts[0];
    int max_nnz = nnz_counts[0];
    for (int p = 1; p < ranks; ++p) {
        if (rows[p] < min_rows) min_rows = rows[p];
        if (rows[p] > max_rows) max_rows = rows[p];
        if (nnz_counts[p] < min_nnz) min_nnz = nnz_counts[p];
        if (nnz_counts[p] > max_nnz) max_nnz = nnz_counts[p];
    }
    const double average = (double)a->nnz / ranks;
    printf("Partition=hybrid-cost Rows=%d Nonzeros=%d Ranks=%d\n",
           a->n, a->nnz, ranks);
    printf("Hybrid row weight = %d nonzero-equivalent units per row\n", row_weight);
    printf("Rows per rank: min=%d max=%d\n", min_rows, max_rows);
    printf("Nonzeros per rank: min=%d max=%d average=%.6f\n",
           min_nnz, max_nnz, average);
    if (a->nnz == 0) {
        printf("Nonzero imbalance ratio = N/A (zero nonzeros)\n");
    } else {
        printf("Nonzero imbalance ratio = %.6f\n", max_nnz / average);
    }
}

static double e2e_clock_seconds(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    /* Identical process-lifetime measurement in all variants. This includes
       MPI_Init, file I/O, pre-timing communicator work, computation, and MPI_Finalize.
       Measured for rank zero; Python also times complete mpirun execution. */
    const double e2e_start = e2e_clock_seconds();
    MPI_Init(&argc, &argv);
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (argc != 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s A.csr X.dense W.dense Y.dense\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    openblas_set_num_threads(1);

    CsrMatrix a = {0};
    DenseMatrix x = {0};
    DenseMatrix w = {0};
    int dimensions[3];
    if (rank == 0) {
        a = load_csr(argv[1]);
        x = load_dense(argv[2]);
        w = load_dense(argv[3]);
        if (x.rows != a.n || x.cols != w.rows) {
            fail("Expected A=NxN, X=NxF, W=FxH");
        }
        if ((size_t)a.n > (size_t)INT_MAX / (size_t)w.cols) {
            fail("Output exceeds MPI count limit");
        }
        dimensions[0] = a.n;
        dimensions[1] = x.cols;
        dimensions[2] = w.cols;
    }
    MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);
    const int n = dimensions[0];
    const int features = dimensions[1];
    const int hidden = dimensions[2];
    const int use_xw_first = hidden <= features;

    /* New timing boundary: inputs and dimensions are loaded; partition setup starts. */
    MPI_Barrier(MPI_COMM_WORLD);
    const double partition_start = MPI_Wtime();

    const int row_weight = hybrid_row_weight(hidden, use_xw_first);
    int *rows = allocate_array((size_t)ranks, sizeof(*rows));
    int *offsets = allocate_array((size_t)ranks, sizeof(*offsets));
    int *nnz_counts = allocate_array((size_t)ranks, sizeof(*nnz_counts));
    int *nnz_offsets = allocate_array((size_t)ranks, sizeof(*nnz_offsets));
    int *output_counts = allocate_array((size_t)ranks, sizeof(*output_counts));
    int *output_offsets = allocate_array((size_t)ranks, sizeof(*output_offsets));
    if (rank == 0) {
        partition_rows(&a, ranks, hidden, row_weight, rows, offsets, nnz_counts, nnz_offsets,
                       output_counts, output_offsets);
    }
    MPI_Bcast(rows, ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(offsets, ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz_counts, ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(nnz_offsets, ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_counts, ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output_offsets, ranks, MPI_INT, 0, MPI_COMM_WORLD);

    /* Include rank-local partition setup plus metadata communication in the metric. */
    const double partition_elapsed = MPI_Wtime() - partition_start;
    double max_partition_elapsed = 0.0;
    MPI_Reduce(&partition_elapsed, &max_partition_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    double phases[5] = {0}; /* Distribution, root GEMM, dense broadcast, local kernels, gather. */
    MPI_Barrier(MPI_COMM_WORLD);
    const double total_start = MPI_Wtime();
    double start = MPI_Wtime();
    /* Experimental optimization: broadcast the full CSR once instead of
     * performing 4 root-managed Scatter/Scatterv collectives.  Each rank
     * keeps a replicated CSR matrix and creates a non-owning local view.
     * This is practical for the current N=40000 / nnz=640000 workload,
     * but memory usage grows with rank count. */
    MPI_Bcast(&a.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        a.n = n;
        a.row_ptr = allocate_array((size_t)n + 1, sizeof(*a.row_ptr));
        a.col_idx = allocate_array((size_t)a.nnz, sizeof(*a.col_idx));
        a.values = allocate_array((size_t)a.nnz, sizeof(*a.values));
    }
    MPI_Bcast(a.row_ptr, n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(a.col_idx, a.nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(a.values, a.nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CsrMatrix local = {0};
    local.n = n;
    local.nnz = nnz_counts[rank];
    local.row_ptr = allocate_array((size_t)rows[rank] + 1,
                                   sizeof(*local.row_ptr));
    for (int i = 0; i <= rows[rank]; ++i) {
        local.row_ptr[i] = a.row_ptr[offsets[rank] + i] - nnz_offsets[rank];
    }
    if (local.row_ptr[rows[rank]] != local.nnz) {
        fail("Replicated CSR local view has incorrect nnz");
    }
    /* The following pointers are non-owning slices of replicated A. */
    local.col_idx = a.col_idx + nnz_offsets[rank];
    local.values = a.values + nnz_offsets[rank];
    int *lengths = NULL;
    int *send_lengths = NULL;
    phases[0] = MPI_Wtime() - start;

    DenseMatrix dense_input = {n, use_xw_first ? hidden : features, NULL};
    dense_input.values = allocate_array((size_t)n * (size_t)dense_input.cols,
                                        sizeof(*dense_input.values));
    DenseMatrix local_y = {rows[rank], hidden, NULL};
    local_y.values = allocate_array((size_t)rows[rank] * (size_t)hidden,
                                    sizeof(*local_y.values));
    DenseMatrix local_ax = {rows[rank], features, NULL};
    if (!use_xw_first) {
        local_ax.values = allocate_array((size_t)rows[rank] * (size_t)features,
                                         sizeof(*local_ax.values));
        if (rank != 0) {
            w.rows = features;
            w.cols = hidden;
            w.values = allocate_array((size_t)features * (size_t)hidden,
                                       sizeof(*w.values));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        if (use_xw_first) {
            start = MPI_Wtime();
            dense_multiply(&x, &w, &dense_input);
            phases[1] = MPI_Wtime() - start;
        } else {
            memcpy(dense_input.values, x.values,
                   (size_t)n * (size_t)features * sizeof(float));
        }
    }
    /* Exclude waiting for the root GEMM from the dense-broadcast phase. */
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    MPI_Bcast(dense_input.values, n * dense_input.cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (!use_xw_first) {
        MPI_Bcast(w.values, features * hidden, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    phases[2] = MPI_Wtime() - start;

    start = MPI_Wtime();
    if (use_xw_first) {
        csr_dense_multiply(&local, rows[rank], &dense_input, &local_y);
    } else {
        csr_dense_multiply(&local, rows[rank], &dense_input, &local_ax);
        dense_multiply(&local_ax, &w, &local_y);
    }
    phases[3] = MPI_Wtime() - start;

    DenseMatrix y = {n, hidden, NULL};
    if (rank == 0) {
        y.values = allocate_array((size_t)n * (size_t)hidden, sizeof(*y.values));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    MPI_Gatherv(local_y.values, rows[rank] * hidden, MPI_FLOAT,
                y.values, output_counts, output_offsets, MPI_FLOAT, 0, MPI_COMM_WORLD);
    phases[4] = MPI_Wtime() - start;
    double total = MPI_Wtime() - total_start;
    double inclusive_total = MPI_Wtime() - partition_start;
    double max_inclusive_total = 0.0;
    double max_phases[5];
    double max_total;
    double min_compute;
    double sum_compute;
    MPI_Reduce(phases, max_phases, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&inclusive_total, &max_inclusive_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&phases[3], &min_compute, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&phases[3], &sum_compute, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        write_dense(argv[4], &y);
        printf("Order=%s Ranks=%d OpenBLAS_threads=%d\n",
               use_xw_first ? "A(XW)" : "(AX)W", ranks, openblas_get_num_threads());
        printf("CSR distribution = %.9f seconds\nRoot XW = %.9f seconds\n"
               "Dense broadcast = %.9f seconds\nLocal computation = %.9f seconds\n"
               "Output gathering = %.9f seconds\nMPI elapsed time = %.9f seconds\n",
               max_phases[0], max_phases[1], max_phases[2],
               max_phases[3], max_phases[4], max_total);
        printf("Partition setup = %.9f seconds\n"
               "MPI setup-inclusive time = %.9f seconds\n",
               max_partition_elapsed, max_inclusive_total);
        print_partition_summary(&a, ranks, row_weight, rows, nnz_counts);
        printf("Rank-local computation: min=%.9f max=%.9f average=%.9f seconds\n",
               min_compute, max_phases[3], sum_compute / ranks);
    }
    free_csr(&a);
    free(local.row_ptr); /* other local arrays alias replicated A */
    free(x.values);
    free(w.values);
    free(dense_input.values);
    free(local_ax.values);
    free(local_y.values);
    free(y.values);
    free(rows);
    free(offsets);
    free(nnz_counts);
    free(nnz_offsets);
    free(output_counts);
    free(output_offsets);
    free(lengths);
    free(send_lengths);
    MPI_Finalize();
    if (rank == 0) {
        printf("Rank0 process lifetime = %.9f seconds\n",
               e2e_clock_seconds() - e2e_start);
        fflush(stdout);
    }
    return EXIT_SUCCESS;
}
