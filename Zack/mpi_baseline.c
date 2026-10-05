#include <mpi.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static void *allocate_array(size_t count, size_t element_size)
{
    if (count > SIZE_MAX / element_size) {
        return NULL;
    }
    return calloc(count == 0 ? 1 : count, element_size);
}

static void free_csr(CsrMatrix *matrix)
{
    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    matrix->row_ptr = NULL;
    matrix->col_idx = NULL;
    matrix->values = NULL;
}

static void free_dense(DenseMatrix *matrix)
{
    free(matrix->values);
    matrix->values = NULL;
}

static int load_csr(const char *path, CsrMatrix *matrix)
{
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open sparse matrix file: %s\n", path);
        return 0;
    }

    int rows;
    int cols;
    int ok = fscanf(file, "%d %d %d", &rows, &cols, &matrix->nnz) == 3 &&
             rows > 0 && cols > 0 && rows == cols && matrix->nnz >= 0;
    if (!ok) {
        fprintf(stderr, "Invalid CSR header in %s (expected: rows columns nnz; A must be square)\n",
                path);
        fclose(file);
        return 0;
    }
    matrix->n = rows;

    matrix->row_ptr = allocate_array((size_t)matrix->n + 1, sizeof(*matrix->row_ptr));
    matrix->col_idx = allocate_array((size_t)matrix->nnz, sizeof(*matrix->col_idx));
    matrix->values = allocate_array((size_t)matrix->nnz, sizeof(*matrix->values));
    if (matrix->row_ptr == NULL || matrix->col_idx == NULL || matrix->values == NULL) {
        fprintf(stderr, "Insufficient memory while reading %s\n", path);
        fclose(file);
        return 0;
    }

    for (int i = 0; i <= matrix->n && ok; ++i) {
        ok = fscanf(file, "%d", &matrix->row_ptr[i]) == 1;
    }
    ok = ok && matrix->row_ptr[0] == 0 &&
         matrix->row_ptr[matrix->n] == matrix->nnz;
    for (int i = 0; i < matrix->n && ok; ++i) {
        ok = matrix->row_ptr[i] <= matrix->row_ptr[i + 1];
    }
    for (int i = 0; i < matrix->nnz && ok; ++i) {
        ok = fscanf(file, "%d", &matrix->col_idx[i]) == 1 &&
             matrix->col_idx[i] >= 0 && matrix->col_idx[i] < matrix->n;
    }
    for (int i = 0; i < matrix->nnz && ok; ++i) {
        ok = fscanf(file, "%f", &matrix->values[i]) == 1;
    }

    if (!ok) {
        fprintf(stderr, "Invalid CSR data in %s\n", path);
    }
    fclose(file);
    return ok;
}

static int load_dense(const char *path, DenseMatrix *matrix)
{
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open dense matrix file: %s\n", path);
        return 0;
    }

    int ok = fscanf(file, "%d %d", &matrix->rows, &matrix->cols) == 2 &&
             matrix->rows > 0 && matrix->cols > 0;
    if (!ok) {
        fprintf(stderr, "Invalid dense matrix header in %s (expected: rows columns)\n",
                path);
        fclose(file);
        return 0;
    }

    size_t elements = (size_t)matrix->rows * (size_t)matrix->cols;
    if (elements > INT_MAX) {
        fprintf(stderr, "Dense matrix in %s exceeds MPI count limits\n", path);
        fclose(file);
        return 0;
    }

    matrix->values = allocate_array(elements, sizeof(*matrix->values));
    if (matrix->values == NULL) {
        fprintf(stderr, "Insufficient memory while reading %s\n", path);
        fclose(file);
        return 0;
    }

    for (size_t i = 0; i < elements && ok; ++i) {
        ok = fscanf(file, "%f", &matrix->values[i]) == 1;
    }
    if (!ok) {
        fprintf(stderr, "Invalid dense matrix data in %s\n", path);
    }
    fclose(file);
    return ok;
}

static void write_dense(const char *path, const float *values, int rows, int cols)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "Cannot open output file %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(file, "%d %d\n", rows, cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col > 0) {
                fputc(' ', file);
            }
            fprintf(file, "%.9g", values[(size_t)row * (size_t)cols +
                                         (size_t)col]);
        }
        fputc('\n', file);
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Failed to finish writing output file %s\n", path);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

static void abort_on_allocation_failure(int rank, const char *what)
{
    fprintf(stderr, "Rank %d: insufficient memory for %s\n", rank, what);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    if (argc != 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s A.csr X.dense W.dense Y.dense\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    CsrMatrix a = {0};
    DenseMatrix x = {0};
    DenseMatrix w = {0};
    int dimensions[3] = {0, 0, 0};
    int input_ok = 1;

    if (rank == 0) {
        input_ok = load_csr(argv[1], &a) &&
                   load_dense(argv[2], &x) &&
                   load_dense(argv[3], &w);
        if (input_ok && (x.rows != a.n || w.rows != x.cols)) {
            fprintf(stderr,
                    "Dimension mismatch: expected A=NxN, X=NxF, and W=FxH\n");
            input_ok = 0;
        }
        if (input_ok && (a.nnz > INT_MAX ||
                         (size_t)a.n * (size_t)x.cols > INT_MAX ||
                         (size_t)x.cols * (size_t)w.cols > INT_MAX ||
                         (size_t)a.n * (size_t)w.cols > INT_MAX)) {
            fprintf(stderr, "Input exceeds MPI count limits\n");
            input_ok = 0;
        }
        if (input_ok) {
            dimensions[0] = a.n;
            dimensions[1] = x.cols;
            dimensions[2] = w.cols;
        }
    }

    MPI_Bcast(&input_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!input_ok) {
        free_csr(&a);
        free_dense(&x);
        free_dense(&w);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    MPI_Bcast(dimensions, 3, MPI_INT, 0, MPI_COMM_WORLD);

    const int n = dimensions[0];
    const int features = dimensions[1];
    const int hidden = dimensions[2];

    int *row_counts = allocate_array((size_t)process_count, sizeof(*row_counts));
    int *row_displacements =
        allocate_array((size_t)process_count, sizeof(*row_displacements));
    int *nnz_counts = allocate_array((size_t)process_count, sizeof(*nnz_counts));
    int *nnz_displacements =
        allocate_array((size_t)process_count, sizeof(*nnz_displacements));
    if (row_counts == NULL || row_displacements == NULL ||
        nnz_counts == NULL || nnz_displacements == NULL) {
        abort_on_allocation_failure(rank, "partition metadata");
    }

    int next_row = 0;
    for (int p = 0; p < process_count; ++p) {
        row_counts[p] = n / process_count + (p < n % process_count);
        row_displacements[p] = next_row;
        next_row += row_counts[p];
        if (rank == 0) {
            int first_row = row_displacements[p];
            int last_row = first_row + row_counts[p];
            nnz_displacements[p] = a.row_ptr[first_row];
            nnz_counts[p] = a.row_ptr[last_row] - a.row_ptr[first_row];
        }
    }

    int local_rows = row_counts[rank];
    int local_nnz = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    MPI_Scatter(nnz_counts, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_row_lengths = allocate_array((size_t)local_rows, sizeof(*local_row_lengths));
    int *local_row_ptr = allocate_array((size_t)local_rows + 1, sizeof(*local_row_ptr));
    int *local_col_idx = allocate_array((size_t)local_nnz, sizeof(*local_col_idx));
    float *local_values = allocate_array((size_t)local_nnz, sizeof(*local_values));
    float *local_ax =
        allocate_array((size_t)local_rows * (size_t)features, sizeof(*local_ax));
    float *local_y =
        allocate_array((size_t)local_rows * (size_t)hidden, sizeof(*local_y));
    if (local_row_lengths == NULL || local_row_ptr == NULL ||
        local_col_idx == NULL || local_values == NULL ||
        local_ax == NULL || local_y == NULL) {
        abort_on_allocation_failure(rank, "local matrices");
    }

    int *send_row_lengths = NULL;
    if (rank == 0) {
        send_row_lengths =
            allocate_array((size_t)n, sizeof(*send_row_lengths));
        if (send_row_lengths == NULL) {
            abort_on_allocation_failure(rank, "CSR row lengths");
        }
        for (int i = 0; i < n; ++i) {
            send_row_lengths[i] = a.row_ptr[i + 1] - a.row_ptr[i];
        }
    }

    MPI_Scatterv(send_row_lengths, row_counts, row_displacements, MPI_INT,
                 local_row_lengths, local_rows, MPI_INT, 0, MPI_COMM_WORLD);

    local_row_ptr[0] = 0;
    for (int i = 0; i < local_rows; ++i) {
        local_row_ptr[i + 1] = local_row_ptr[i] + local_row_lengths[i];
    }
    if (local_row_ptr[local_rows] != local_nnz) {
        fprintf(stderr, "Rank %d: inconsistent local CSR partition\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Scatterv(a.col_idx, nnz_counts, nnz_displacements, MPI_INT,
                 local_col_idx, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a.values, nnz_counts, nnz_displacements, MPI_FLOAT,
                 local_values, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    size_t x_elements = (size_t)n * (size_t)features;
    size_t w_elements = (size_t)features * (size_t)hidden;
    float *x_values = allocate_array(x_elements, sizeof(*x_values));
    float *w_values = allocate_array(w_elements, sizeof(*w_values));
    if (x_values == NULL || w_values == NULL) {
        abort_on_allocation_failure(rank, "replicated dense inputs");
    }
    if (rank == 0) {
        for (size_t i = 0; i < x_elements; ++i) {
            x_values[i] = x.values[i];
        }
        for (size_t i = 0; i < w_elements; ++i) {
            w_values[i] = w.values[i];
        }
    }
    MPI_Bcast(x_values, (int)x_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(w_values, (int)w_elements, MPI_FLOAT, 0, MPI_COMM_WORLD);

    for (int local_row = 0; local_row < local_rows; ++local_row) {
        float *ax_row = &local_ax[(size_t)local_row * (size_t)features];
        for (int entry = local_row_ptr[local_row];
             entry < local_row_ptr[local_row + 1]; ++entry) {
            const float value = local_values[entry];
            const float *x_row =
                &x_values[(size_t)local_col_idx[entry] * (size_t)features];
            for (int feature = 0; feature < features; ++feature) {
                ax_row[feature] += value * x_row[feature];
            }
        }

        float *y_row = &local_y[(size_t)local_row * (size_t)hidden];
        for (int feature = 0; feature < features; ++feature) {
            const float value = ax_row[feature];
            const float *w_row =
                &w_values[(size_t)feature * (size_t)hidden];
            for (int column = 0; column < hidden; ++column) {
                y_row[column] += value * w_row[column];
            }
        }
    }

    int *gather_counts = NULL;
    int *gather_displacements = NULL;
    float *global_output = NULL;
    if (rank == 0) {
        gather_counts = allocate_array((size_t)process_count, sizeof(*gather_counts));
        gather_displacements =
            allocate_array((size_t)process_count, sizeof(*gather_displacements));
        global_output =
            allocate_array((size_t)n * (size_t)hidden, sizeof(*global_output));
        if (gather_counts == NULL || gather_displacements == NULL ||
            global_output == NULL) {
            abort_on_allocation_failure(rank, "output partition metadata");
        }
        for (int p = 0; p < process_count; ++p) {
            gather_counts[p] = row_counts[p] * hidden;
            gather_displacements[p] = row_displacements[p] * hidden;
        }
    }
    MPI_Gatherv(local_y, local_rows * hidden, MPI_FLOAT,
                global_output, gather_counts, gather_displacements, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        write_dense(argv[4], global_output, n, hidden);
        printf("MPI elapsed time = %.6f seconds\n", elapsed);
    }

    free_csr(&a);
    free_dense(&x);
    free_dense(&w);
    free(row_counts);
    free(row_displacements);
    free(nnz_counts);
    free(nnz_displacements);
    free(local_row_lengths);
    free(local_row_ptr);
    free(local_col_idx);
    free(local_values);
    free(local_ax);
    free(local_y);
    free(send_row_lengths);
    free(x_values);
    free(w_values);
    free(gather_counts);
    free(gather_displacements);
    free(global_output);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
