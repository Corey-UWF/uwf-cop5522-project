#include <mpi.h>
/* LP64 OpenBLAS declarations: Expanse does not provide cblas.h in this module. */
enum { CblasRowMajor = 101, CblasNoTrans = 111 };
extern void cblas_sgemm(const int order, const int trans_a, const int trans_b,
                        const int m, const int n, const int k,
                        const float alpha, const float *a, const int lda,
                        const float *b, const int ldb, const float beta,
                        float *c, const int ldc);
extern void openblas_set_num_threads(int threads);
extern int openblas_get_num_threads(void);

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char **argv)
{
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

    int *rows = allocate_array((size_t)ranks, sizeof(*rows));
    int *offsets = allocate_array((size_t)ranks, sizeof(*offsets));
    int *nnz_counts = allocate_array((size_t)ranks, sizeof(*nnz_counts));
    int *nnz_offsets = allocate_array((size_t)ranks, sizeof(*nnz_offsets));
    int *output_counts = allocate_array((size_t)ranks, sizeof(*output_counts));
    int *output_offsets = allocate_array((size_t)ranks, sizeof(*output_offsets));
    int next = 0;
    for (int p = 0; p < ranks; ++p) {
        rows[p] = n / ranks + (p < n % ranks);
        offsets[p] = next;
        next += rows[p];
        output_counts[p] = rows[p] * hidden;
        output_offsets[p] = offsets[p] * hidden;
        if (rank == 0) {
            nnz_offsets[p] = a.row_ptr[offsets[p]];
            nnz_counts[p] = a.row_ptr[next] - nnz_offsets[p];
        }
    }

    /* Include rank-local partition setup plus metadata communication in the metric. */
    const double partition_elapsed = MPI_Wtime() - partition_start;
    double max_partition_elapsed = 0.0;
    MPI_Reduce(&partition_elapsed, &max_partition_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    double phases[5] = {0}; /* Distribution, root GEMM, dense broadcast, local kernels, gather. */
    MPI_Barrier(MPI_COMM_WORLD);
    const double total_start = MPI_Wtime();
    double start = MPI_Wtime();
    CsrMatrix local = {0};
    local.n = n;
    MPI_Scatter(nnz_counts, 1, MPI_INT, &local.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int *lengths = allocate_array((size_t)rows[rank], sizeof(*lengths));
    int *send_lengths = NULL;
    if (rank == 0) {
        send_lengths = allocate_array((size_t)n, sizeof(*send_lengths));
        for (int i = 0; i < n; ++i) {
            send_lengths[i] = a.row_ptr[i + 1] - a.row_ptr[i];
        }
    }
    local.row_ptr = allocate_array((size_t)rows[rank] + 1, sizeof(*local.row_ptr));
    local.col_idx = allocate_array((size_t)local.nnz, sizeof(*local.col_idx));
    local.values = allocate_array((size_t)local.nnz, sizeof(*local.values));
    MPI_Scatterv(send_lengths, rows, offsets, MPI_INT,
                 lengths, rows[rank], MPI_INT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < rows[rank]; ++i) {
        local.row_ptr[i + 1] = local.row_ptr[i] + lengths[i];
    }
    MPI_Scatterv(a.col_idx, nnz_counts, nnz_offsets, MPI_INT,
                 local.col_idx, local.nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a.values, nnz_counts, nnz_offsets, MPI_FLOAT,
                 local.values, local.nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    MPI_Reduce(phases, max_phases, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&inclusive_total, &max_inclusive_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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
    }
    free_csr(&a);
    free_csr(&local);
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
    return EXIT_SUCCESS;
}
