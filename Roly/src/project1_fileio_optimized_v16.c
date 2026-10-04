/* Minor modifications to code produced by ChatGPT */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include <time.h>


/*
 * Frozen portable sequential sparse-kernel configuration.
 *
 * These settings were selected after testing across the development VM,
 * cs-ssh, Bridges-2, and Expanse.  The 128-wide feature block with no
 * software prefetch was the most portable configuration tested.
 */
#ifndef FEATURE_BLOCK
#define FEATURE_BLOCK 128
#endif

#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 0
#endif

#if FEATURE_BLOCK <= 0
#error "FEATURE_BLOCK must be greater than zero"
#endif

#if PREFETCH_DISTANCE < 0
#error "PREFETCH_DISTANCE must be zero or greater"
#endif

typedef struct {
    int rows;
    int cols;
    int nnz;
    int *rowptr;
    int *colidx;
    float *values;
} CSRMatrix;

typedef struct {
    int rows;
    int cols;
    float *data;   /* row-major */
} DenseMatrix;


/*
 * Portable replacement for the professor's microtime.h timer.
 *
 * CLOCK_MONOTONIC is used because elapsed performance measurements should
 * not be affected if the system wall clock is adjusted while the program
 * is running.
 *
 * These functions return microseconds so that the timing units remain the
 * same as the professor's reference implementation.
 */
static double microtime(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        fprintf(stderr, "clock_gettime failed\n");
        exit(EXIT_FAILURE);
    }

    return (double)ts.tv_sec * 1.0e6 +
           (double)ts.tv_nsec * 1.0e-3;
}

static double get_microtime_resolution(void)
{
    struct timespec res;

    if (clock_getres(CLOCK_MONOTONIC, &res) != 0) {
        fprintf(stderr, "clock_getres failed\n");
        exit(EXIT_FAILURE);
    }

    return (double)res.tv_sec * 1.0e6 +
           (double)res.tv_nsec * 1.0e-3;
}

/* ---------- Utility functions ---------- */

static void die(const char *msg)
{
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t nbytes)
{
    void *p = malloc(nbytes);
    if (p == NULL) {
        fprintf(stderr, "Memory allocation of %zu bytes failed\n", nbytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

/*
 * Read the entire file into memory.
 * The returned buffer is NUL-terminated so that strtol/strtof can parse it.
 */
static char *read_file(const char *filename, size_t *size_out)
{
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Could not open %s: %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    struct stat st;
    if (fstat(fileno(fp), &st) != 0) {
        fprintf(stderr, "fstat failed on %s: %s\n", filename, strerror(errno));
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    if (st.st_size < 0) {
        fprintf(stderr, "Invalid file size for %s\n", filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    size_t size = (size_t) st.st_size;
    char *buffer = (char *) xmalloc(size + 1);

    size_t nread = fread(buffer, 1, size, fp);
    fclose(fp);

    if (nread != size) {
        fprintf(stderr, "Short read on %s: expected %zu bytes, got %zu\n",
                filename, size, nread);
        free(buffer);
        exit(EXIT_FAILURE);
    }

    buffer[size] = '\0';
    *size_out = size;
    return buffer;
}

static void skip_space(char **p)
{
    while (**p == ' ' || **p == '\t' || **p == '\n' ||
           **p == '\r' || **p == '\f' || **p == '\v') {
        ++(*p);
    }
}

static long next_long(char **p, const char *what)
{
    skip_space(p);

    if (**p == '\0') {
        fprintf(stderr, "Unexpected end of file while reading %s\n", what);
        exit(EXIT_FAILURE);
    }

    char *end;
    errno = 0;
    long value = strtol(*p, &end, 10);

    if (end == *p || errno != 0) {
        fprintf(stderr, "Could not parse integer while reading %s\n", what);
        exit(EXIT_FAILURE);
    }

    *p = end;
    return value;
}

static float next_float(char **p, const char *what)
{
    skip_space(p);

    if (**p == '\0') {
        fprintf(stderr, "Unexpected end of file while reading %s\n", what);
        exit(EXIT_FAILURE);
    }

    char *end;
    errno = 0;
    float value = strtof(*p, &end);

    if (end == *p || errno != 0) {
        fprintf(stderr, "Could not parse floating-point value while reading %s\n", what);
        exit(EXIT_FAILURE);
    }

    *p = end;
    return value;
}

/* ---------- Matrix input/output ---------- */

static CSRMatrix read_csr(const char *filename)
{
    size_t filesize;
    char *buffer = read_file(filename, &filesize);
    (void) filesize;

    char *p = buffer;

    long rows_l = next_long(&p, "CSR row count");
    long cols_l = next_long(&p, "CSR column count");
    long nnz_l  = next_long(&p, "CSR nonzero count");

    if (rows_l <= 0 || cols_l <= 0 || nnz_l < 0) {
        die("Invalid CSR matrix dimensions");
    }

    if (rows_l > 2147483647L || cols_l > 2147483647L || nnz_l > 2147483647L) {
        die("CSR dimensions exceed 32-bit integer range");
    }

    CSRMatrix A;
    A.rows = (int) rows_l;
    A.cols = (int) cols_l;
    A.nnz  = (int) nnz_l;

    A.rowptr = (int *) xmalloc((size_t)(A.rows + 1) * sizeof(int));
    A.colidx = (int *) xmalloc((size_t)A.nnz * sizeof(int));
    A.values = (float *) xmalloc((size_t)A.nnz * sizeof(float));

    for (int i = 0; i <= A.rows; ++i) {
        long v = next_long(&p, "CSR row pointer");
        if (v < 0 || v > A.nnz) {
            die("CSR row pointer out of range");
        }
        A.rowptr[i] = (int) v;
    }

    if (A.rowptr[0] != 0 || A.rowptr[A.rows] != A.nnz) {
        die("CSR row pointers must start at 0 and end at nnz");
    }

    for (int i = 0; i < A.rows; ++i) {
        if (A.rowptr[i] > A.rowptr[i + 1]) {
            die("CSR row pointers must be nondecreasing");
        }
    }

    for (int k = 0; k < A.nnz; ++k) {
        long c = next_long(&p, "CSR column index");
        if (c < 0 || c >= A.cols) {
            die("CSR column index out of range");
        }
        A.colidx[k] = (int) c;
    }

    for (int k = 0; k < A.nnz; ++k) {
        A.values[k] = next_float(&p, "CSR value");
    }

    free(buffer);
    return A;
}

static DenseMatrix read_dense(const char *filename)
{
    size_t filesize;
    char *buffer = read_file(filename, &filesize);
    (void) filesize;

    char *p = buffer;

    long rows_l = next_long(&p, "dense row count");
    long cols_l = next_long(&p, "dense column count");

    if (rows_l <= 0 || cols_l <= 0) {
        die("Invalid dense matrix dimensions");
    }

    if (rows_l > 2147483647L || cols_l > 2147483647L) {
        die("Dense matrix dimensions exceed 32-bit integer range");
    }

    DenseMatrix A;
    A.rows = (int) rows_l;
    A.cols = (int) cols_l;

    size_t nelem = (size_t)A.rows * (size_t)A.cols;
    A.data = (float *) xmalloc(nelem * sizeof(float));

    for (size_t i = 0; i < nelem; ++i) {
        A.data[i] = next_float(&p, "dense matrix value");
    }

    free(buffer);
    return A;
}

static void write_dense(const char *filename, const DenseMatrix *A)
{
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not open %s for writing: %s\n",
                filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "%d %d\n", A->rows, A->cols);

    for (int i = 0; i < A->rows; ++i) {
        for (int j = 0; j < A->cols; ++j) {
            if (j > 0)
                fputc(' ', fp);
            fprintf(fp, "%.9g", A->data[(size_t)i * A->cols + j]);
        }
        fputc('\n', fp);
    }

    fclose(fp);
}

/* ---------- Reference computation ---------- */

/*
 * T = A * X
 *
 * A is CSR, X and T are dense row-major matrices.
 * This is intentionally straightforward reference code, not optimized code.
 */
static void csr_dense_multiply(const CSRMatrix *A,
                               const DenseMatrix *X,
                               DenseMatrix *T)
{
    if (A->cols != X->rows)
        die("Dimension mismatch in A * X");

    if (T->rows != A->rows || T->cols != X->cols)
        die("Incorrect output dimensions for A * X");

    /*
     * Preserve the professor reference behavior: every call starts
     * with a zeroed output matrix.
     */
    memset(T->data, 0,
           (size_t)T->rows * (size_t)T->cols * sizeof(float));

    /*
     * Process one CSR row at a time.
     *
     * The dense feature dimension is divided into blocks.  Our
     * cross-platform sequential experiments selected a block size
     * of 128 with software prefetch disabled as the portable
     * configuration.
     */
    for (int row = 0; row < A->rows; ++row) {
        const int row_begin = A->rowptr[row];
        const int row_end   = A->rowptr[row + 1];

        for (int block_start = 0;
             block_start < X->cols;
             block_start += FEATURE_BLOCK) {

            int block_end = block_start + FEATURE_BLOCK;

            if (block_end > X->cols)
                block_end = X->cols;

            for (int p = row_begin; p < row_end; ++p) {

#if PREFETCH_DISTANCE > 0
                /*
                 * Optional software prefetch retained for controlled
                 * experiments.  The frozen portable setting is zero.
                 */
                const int prefetch_p = p + PREFETCH_DISTANCE;

                if (prefetch_p < row_end) {
                    const int prefetch_col =
                        A->colidx[prefetch_p];

                    __builtin_prefetch(
                        &X->data[
                            (size_t)prefetch_col * X->cols +
                            block_start],
                        0,
                        1);
                }
#endif

                const float a = A->values[p];
                const int col = A->colidx[p];

                float *t_row =
                    &T->data[(size_t)row * T->cols];

                const float *x_row =
                    &X->data[(size_t)col * X->cols];

                /*
                 * Contiguous inner loop over the current feature block.
                 */
                for (int j = block_start; j < block_end; ++j) {
                    t_row[j] += a * x_row[j];
                }
            }
        }
    }
}

/*
 * C = A * B
 *
 * All matrices are dense and stored row-major.
 * This is deliberately the same simple i-j-k organization used in HW1.
 */
static void dense_multiply(const DenseMatrix *A,
                           const DenseMatrix *B,
                           DenseMatrix *C)
{
    if (A->cols != B->rows)
        die("Dimension mismatch in dense matrix multiplication");

    if (C->rows != A->rows || C->cols != B->cols)
        die("Incorrect output dimensions for dense matrix multiplication");

    /*
     * Preserve the professor reference behavior.
     */
    memset(C->data, 0,
           (size_t)C->rows * (size_t)C->cols * sizeof(float));

    /*
     * Optimized loop order: i -> k -> j.
     *
     * For each A[i][k], the innermost j loop walks contiguous rows
     * of B and C.  This improves spatial locality and gives GCC a
     * much simpler loop to vectorize than the reference i-j-k order.
     */
    for (int i = 0; i < A->rows; ++i) {

        float *c_row =
            &C->data[(size_t)i * C->cols];

        for (int k = 0; k < A->cols; ++k) {

            const float a =
                A->data[(size_t)i * A->cols + k];

            const float *b_row =
                &B->data[(size_t)k * B->cols];

            for (int j = 0; j < B->cols; ++j) {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

static void free_csr(CSRMatrix *A)
{
    free(A->rowptr);
    free(A->colidx);
    free(A->values);

    A->rowptr = NULL;
    A->colidx = NULL;
    A->values = NULL;
}

static void free_dense(DenseMatrix *A)
{
    free(A->data);
    A->data = NULL;
}

/* ---------- Main ---------- */

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr,
                "USAGE: %s A.csr X.dense W.dense Y.dense\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    CSRMatrix A = read_csr(argv[1]);
    DenseMatrix X = read_dense(argv[2]);
    DenseMatrix W = read_dense(argv[3]);

    /*
     * Project problem:
     *     Y = A X W
     *
     * Required dimensions:
     *     A : N x N
     *     X : N x F
     *     W : F x H
     *     Y : N x H
     */
    if (A.rows != A.cols)
        die("A must be square");

    if (A.cols != X.rows)
        die("A and X dimensions are incompatible");

    if (X.cols != W.rows)
        die("X and W dimensions are incompatible");

    DenseMatrix T;
    T.rows = A.rows;
    T.cols = X.cols;
    T.data = (float *) xmalloc(
        (size_t)T.rows * (size_t)T.cols * sizeof(float));

    DenseMatrix Y;
    Y.rows = A.rows;
    Y.cols = W.cols;
    Y.data = (float *) xmalloc(
        (size_t)Y.rows * (size_t)Y.cols * sizeof(float));

    double time1, time2;

    time1 = microtime();

    /*
     * Reference evaluation order:
     *
     *     T = A X
     *     Y = T W
     *
     * Students are free to evaluate A(XW) or use any mathematically
     * equivalent approach in their project.
     */
    csr_dense_multiply(&A, &X, &T);
    dense_multiply(&T, &W, &Y);

    time2 = microtime();

    /*
     * Nominal floating-point operation count for (A X) W:
     *
     *     A X : 2 * nnz(A) * F
     *     T W : 2 * N * F * H
     */
    double flops =
        2.0 * (double)A.nnz * (double)X.cols +
        2.0 * (double)A.rows * (double)X.cols * (double)W.cols;

    double elapsed = time2 - time1;

    printf("Time = %g us\tTimer Resolution = %g us\tPerformance = %g Gflop/s\n",
           elapsed,
           get_microtime_resolution(),
           flops * 1e-3 / elapsed);

    write_dense(argv[4], &Y);

    free_csr(&A);
    free_dense(&X);
    free_dense(&W);
    free_dense(&T);
    free_dense(&Y);

    return 0;
}
