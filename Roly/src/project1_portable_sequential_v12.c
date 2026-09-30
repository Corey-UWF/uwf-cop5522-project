#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_RUNS 5

/*
 * Current best sparse settings from the previous experiments.
 */
#ifndef FEATURE_BLOCK
#define FEATURE_BLOCK 128
#endif

#ifndef PREFETCH_DISTANCE
#define PREFETCH_DISTANCE 0
#endif

/*
 * Dense-loop experiment:
 *
 * DENSE_IKJ = 0
 *     Original dense loop order:
 *
 *         i -> j -> k
 *
 * DENSE_IKJ = 1
 *     Candidate optimized loop order:
 *
 *         i -> k -> j
 *
 * Keeping both versions in one source file gives us a cleaner
 * controlled comparison.
 */
#ifndef DENSE_IKJ
#define DENSE_IKJ 1
#endif

#if FEATURE_BLOCK <= 0
#error "FEATURE_BLOCK must be greater than zero"
#endif

#if PREFETCH_DISTANCE < 0
#error "PREFETCH_DISTANCE must be zero or greater"
#endif

#if (DENSE_IKJ != 0) && (DENSE_IKJ != 1)
#error "DENSE_IKJ must be 0 or 1"
#endif


/*
 * Convert a row/column pair into the corresponding position in a
 * one-dimensional row-major dense matrix.
 */
#define INDEX(row, col, cols) \
    ((size_t)(row) * (size_t)(cols) + (size_t)(col))


/*
 * ================================================================
 * wall_seconds()
 * ================================================================
 *
 * High-resolution monotonic elapsed-time routine.
 */
static double wall_seconds(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }

    return (double)ts.tv_sec +
           (double)ts.tv_nsec * 1.0e-9;
}


/*
 * ================================================================
 * sparse_dense_multiply()
 * ================================================================
 *
 * Computes:
 *
 *      C = A * B
 *
 * where A is stored in CSR format.
 *
 * This routine keeps the current best sparse optimizations:
 *
 *      FEATURE_BLOCK = 64
 *      PREFETCH_DISTANCE = 2
 *
 * We deliberately keep this routine identical between the two dense
 * loop-order binaries so that the dense loop order is the variable
 * being tested.
 */
static void sparse_dense_multiply(
    int N,
    int K,
    const int row_ptr[],
    const int col_idx[],
    const float values[],
    const float B[],
    float C[])
{
    /*
     * Process one CSR row at a time.
     */
    for (int row = 0; row < N; row++)
    {
        const int row_begin =
            row_ptr[row];

        const int row_end =
            row_ptr[row + 1];

        /*
         * Divide the dense feature dimension into blocks.
         */
        for (int block_start = 0;
             block_start < K;
             block_start += FEATURE_BLOCK)
        {
            int block_end =
                block_start + FEATURE_BLOCK;

            if (block_end > K)
            {
                block_end = K;
            }

            /*
             * Process each sparse nonzero in the current row.
             */
            for (int p = row_begin;
                 p < row_end;
                 p++)
            {
#if PREFETCH_DISTANCE > 0
                /*
                 * Prefetch the dense row associated with a future
                 * sparse nonzero.
                 */
                const int prefetch_p =
                    p + PREFETCH_DISTANCE;

                if (prefetch_p < row_end)
                {
                    const int prefetch_col =
                        col_idx[prefetch_p];

                    __builtin_prefetch(
                        &B[
                            INDEX(
                                prefetch_col,
                                block_start,
                                K)],
                        0,
                        1);
                }
#endif

                const float a =
                    values[p];

                const int col =
                    col_idx[p];

                /*
                 * Process the current dense feature block.
                 */
                for (int j = block_start;
                     j < block_end;
                     j++)
                {
                    C[INDEX(row, j, K)]
                        += a * B[INDEX(col, j, K)];
                }
            }
        }
    }
}


/*
 * ================================================================
 * dense_dense_multiply()
 * ================================================================
 *
 * Computes:
 *
 *      C = A * B
 *
 * Dimensions:
 *
 *      A = rows x shared
 *      B = shared x cols
 *      C = rows x cols
 *
 *
 * CONTROL VERSION: i -> j -> k
 * --------------------------------
 *
 * For each output element C[i][j], the k loop walks:
 *
 *      A[i][k]
 *      B[k][j]
 *
 * A[i][k] is contiguous, but B[k][j] moves down a column.
 *
 * Because B is stored row-major, moving down a column means jumping
 * through memory by an entire row at a time.
 *
 *
 * CANDIDATE VERSION: i -> k -> j
 * --------------------------------
 *
 * The j loop becomes innermost.
 *
 * For each A[i][k], we walk:
 *
 *      B[k][0], B[k][1], B[k][2], ...
 *
 * and:
 *
 *      C[i][0], C[i][1], C[i][2], ...
 *
 * Both are contiguous row-major accesses.
 *
 * This should improve spatial locality and should also give GCC a
 * much simpler loop to vectorize.
 */
static void dense_dense_multiply(
    int rows,
    int shared,
    int cols,
    const float A[],
    const float B[],
    float C[])
{
#if DENSE_IKJ == 0

    /*
     * ============================================================
     * ORIGINAL CONTROL: i -> j -> k
     * ============================================================
     */
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            for (int k = 0; k < shared; k++)
            {
                C[INDEX(i, j, cols)]
                    += A[INDEX(i, k, shared)]
                     * B[INDEX(k, j, cols)];
            }
        }
    }

#else

    /*
     * ============================================================
     * OPTIMIZED CANDIDATE: i -> k -> j
     * ============================================================
     */
    for (int i = 0; i < rows; i++)
    {
        /*
         * Starting address of output row i.
         *
         * Keeping this pointer outside the k and j loops avoids
         * repeatedly recalculating the row base.
         */
        float *c_row =
            &C[INDEX(i, 0, cols)];

        for (int k = 0; k < shared; k++)
        {
            /*
             * A[i][k] is reused across the entire j loop.
             */
            const float a =
                A[INDEX(i, k, shared)];

            /*
             * Starting address of row k in B.
             *
             * The inner j loop will walk through this row
             * contiguously.
             */
            const float *b_row =
                &B[INDEX(k, 0, cols)];

            /*
             * Contiguous inner loop.
             *
             * Both:
             *
             *      b_row[j]
             *
             * and:
             *
             *      c_row[j]
             *
             * advance one float at a time.
             */
            for (int j = 0; j < cols; j++)
            {
                c_row[j] +=
                    a * b_row[j];
            }
        }
    }

#endif
}


/*
 * ================================================================
 * initialize_synthetic_csr()
 * ================================================================
 *
 * Generates the same deterministic sparse matrix used throughout
 * the controlled sequential experiments.
 */
static void initialize_synthetic_csr(
    int N,
    int nnz_per_row,
    int row_ptr[],
    int col_idx[],
    float values[])
{
    row_ptr[0] =
        0;

    for (int row = 0;
         row < N;
         row++)
    {
        const int start =
            row * nnz_per_row;

        for (int p = 0;
             p < nnz_per_row;
             p++)
        {
            const int index =
                start + p;

            col_idx[index] =
                (row + p * 37) % N;

            values[index] =
                0.10f +
                0.01f *
                (float)((p % 9) + 1);
        }

        row_ptr[row + 1] =
            start + nnz_per_row;
    }
}


/*
 * ================================================================
 * initialize_dense_matrices()
 * ================================================================
 *
 * Initializes X and W deterministically so every implementation
 * receives identical inputs.
 */
static void initialize_dense_matrices(
    int N,
    int F,
    int H,
    float X[],
    float W[])
{
    /*
     * X dimensions:
     *
     *      N x F
     */
    for (int i = 0;
         i < N;
         i++)
    {
        for (int j = 0;
             j < F;
             j++)
        {
            X[INDEX(i, j, F)] =
                (float)(
                    ((i + 3 * j) % 17) + 1)
                / 17.0f;
        }
    }

    /*
     * W dimensions:
     *
     *      F x H
     */
    for (int i = 0;
         i < F;
         i++)
    {
        for (int j = 0;
             j < H;
             j++)
        {
            W[INDEX(i, j, H)] =
                (float)(
                    ((2 * i + 5 * j) % 19) + 1)
                / 19.0f;
        }
    }
}


/*
 * ================================================================
 * matrices_match()
 * ================================================================
 *
 * Compare two floating-point matrices using absolute and relative
 * tolerances.
 */
static int matrices_match(
    int rows,
    int cols,
    const float A[],
    const float B[],
    float absolute_tolerance,
    float relative_tolerance)
{
    for (int i = 0;
         i < rows;
         i++)
    {
        for (int j = 0;
             j < cols;
             j++)
        {
            const size_t index =
                INDEX(i, j, cols);

            const float a =
                A[index];

            const float b =
                B[index];

            const float difference =
                fabsf(a - b);

            const float scale =
                fmaxf(
                    fabsf(a),
                    fabsf(b));

            const float allowed_difference =
                absolute_tolerance +
                relative_tolerance * scale;

            if (difference >
                allowed_difference)
            {
                fprintf(
                    stderr,
                    "Mismatch at row %d, column %d: "
                    "%.8f versus %.8f\n",
                    i,
                    j,
                    (double)a,
                    (double)b);

                return 0;
            }
        }
    }

    return 1;
}


/*
 * ================================================================
 * calculate_gflops()
 * ================================================================
 */
static double calculate_gflops(
    double operations,
    double seconds)
{
    if (seconds <= 0.0)
    {
        return 0.0;
    }

    return operations /
           seconds /
           1.0e9;
}


/*
 * ================================================================
 * compare_doubles()
 * ================================================================
 *
 * qsort() comparison helper.
 */
static int compare_doubles(
    const void *left,
    const void *right)
{
    const double a =
        *(const double *)left;

    const double b =
        *(const double *)right;

    if (a < b)
    {
        return -1;
    }

    if (a > b)
    {
        return 1;
    }

    return 0;
}


/*
 * ================================================================
 * median_time()
 * ================================================================
 */
static double median_time(
    const double samples[NUM_RUNS])
{
    double sorted[NUM_RUNS];

    memcpy(
        sorted,
        samples,
        sizeof(sorted));

    qsort(
        sorted,
        NUM_RUNS,
        sizeof(sorted[0]),
        compare_doubles);

    return sorted[NUM_RUNS / 2];
}


/*
 * ================================================================
 * measure_method1()
 * ================================================================
 *
 * Measures:
 *
 *      Y1 = (AX)W
 */
static void measure_method1(
    int N,
    int F,
    int H,
    const int row_ptr[],
    const int col_idx[],
    const float values[],
    const float X[],
    const float W[],
    float AX[],
    float Y1[],
    double *time_sparse,
    double *time_dense)
{
    /*
     * Clear intermediate and output matrices outside the timed
     * multiplication regions.
     */
    memset(
        AX,
        0,
        (size_t)N *
        (size_t)F *
        sizeof(*AX));

    memset(
        Y1,
        0,
        (size_t)N *
        (size_t)H *
        sizeof(*Y1));

    /*
     * ------------------------------------------------------------
     * Sparse stage:
     *
     *      AX = A * X
     * ------------------------------------------------------------
     */
    double start =
        wall_seconds();

    sparse_dense_multiply(
        N,
        F,
        row_ptr,
        col_idx,
        values,
        X,
        AX);

    *time_sparse =
        wall_seconds() - start;

    /*
     * ------------------------------------------------------------
     * Dense stage:
     *
     *      Y1 = AX * W
     * ------------------------------------------------------------
     */
    start =
        wall_seconds();

    dense_dense_multiply(
        N,
        F,
        H,
        AX,
        W,
        Y1);

    *time_dense =
        wall_seconds() - start;
}


/*
 * ================================================================
 * measure_method2()
 * ================================================================
 *
 * Measures:
 *
 *      Y2 = A(XW)
 */
static void measure_method2(
    int N,
    int F,
    int H,
    const int row_ptr[],
    const int col_idx[],
    const float values[],
    const float X[],
    const float W[],
    float XW[],
    float Y2[],
    double *time_dense,
    double *time_sparse)
{
    /*
     * Clear intermediate and output matrices outside the timed
     * multiplication regions.
     */
    memset(
        XW,
        0,
        (size_t)N *
        (size_t)H *
        sizeof(*XW));

    memset(
        Y2,
        0,
        (size_t)N *
        (size_t)H *
        sizeof(*Y2));

    /*
     * ------------------------------------------------------------
     * Dense stage:
     *
     *      XW = X * W
     * ------------------------------------------------------------
     */
    double start =
        wall_seconds();

    dense_dense_multiply(
        N,
        F,
        H,
        X,
        W,
        XW);

    *time_dense =
        wall_seconds() - start;

    /*
     * ------------------------------------------------------------
     * Sparse stage:
     *
     *      Y2 = A * XW
     * ------------------------------------------------------------
     */
    start =
        wall_seconds();

    sparse_dense_multiply(
        N,
        H,
        row_ptr,
        col_idx,
        values,
        XW,
        Y2);

    *time_sparse =
        wall_seconds() - start;
}


/*
 * ================================================================
 * main()
 * ================================================================
 */
int main(void)
{
    /*
     * Controlled benchmark dimensions.
     */
    const int N =
        4000;

    const int F =
        128;

    const int H =
        64;

    const int nnz_per_row =
        16;

    const int nnz =
        N * nnz_per_row;


    /*
     * ============================================================
     * ALLOCATE CSR ARRAYS
     * ============================================================
     */
    float *values =
        malloc(
            (size_t)nnz *
            sizeof(*values));

    int *col_idx =
        malloc(
            (size_t)nnz *
            sizeof(*col_idx));

    int *row_ptr =
        malloc(
            (size_t)(N + 1) *
            sizeof(*row_ptr));


    /*
     * ============================================================
     * ALLOCATE DENSE MATRICES
     * ============================================================
     */
    float *X =
        malloc(
            (size_t)N *
            (size_t)F *
            sizeof(*X));

    float *W =
        malloc(
            (size_t)F *
            (size_t)H *
            sizeof(*W));

    float *AX =
        calloc(
            (size_t)N *
            (size_t)F,
            sizeof(*AX));

    float *XW =
        calloc(
            (size_t)N *
            (size_t)H,
            sizeof(*XW));

    float *Y1 =
        calloc(
            (size_t)N *
            (size_t)H,
            sizeof(*Y1));

    float *Y2 =
        calloc(
            (size_t)N *
            (size_t)H,
            sizeof(*Y2));


    /*
     * ============================================================
     * VERIFY ALLOCATIONS
     * ============================================================
     */
    if (values == NULL ||
        col_idx == NULL ||
        row_ptr == NULL ||
        X == NULL ||
        W == NULL ||
        AX == NULL ||
        XW == NULL ||
        Y1 == NULL ||
        Y2 == NULL)
    {
        fprintf(
            stderr,
            "Error: unable to allocate benchmark memory.\n");

        free(values);
        free(col_idx);
        free(row_ptr);
        free(X);
        free(W);
        free(AX);
        free(XW);
        free(Y1);
        free(Y2);

        return EXIT_FAILURE;
    }


    /*
     * ============================================================
     * INITIALIZE DATA
     * ============================================================
     */
    initialize_synthetic_csr(
        N,
        nnz_per_row,
        row_ptr,
        col_idx,
        values);

    initialize_dense_matrices(
        N,
        F,
        H,
        X,
        W);


    /*
     * ============================================================
     * WARM-UP
     * ============================================================
     */
    double warm_sparse =
        0.0;

    double warm_dense =
        0.0;

    measure_method1(
        N,
        F,
        H,
        row_ptr,
        col_idx,
        values,
        X,
        W,
        AX,
        Y1,
        &warm_sparse,
        &warm_dense);

    measure_method2(
        N,
        F,
        H,
        row_ptr,
        col_idx,
        values,
        X,
        W,
        XW,
        Y2,
        &warm_dense,
        &warm_sparse);


    /*
     * ============================================================
     * TIMING ARRAYS
     * ============================================================
     */
    double times_AX[NUM_RUNS] =
        {0.0};

    double times_AXW_dense[NUM_RUNS] =
        {0.0};

    double times_XW[NUM_RUNS] =
        {0.0};

    double times_AXW_sparse[NUM_RUNS] =
        {0.0};

    double totals_method1[NUM_RUNS] =
        {0.0};

    double totals_method2[NUM_RUNS] =
        {0.0};


    /*
     * ============================================================
     * MEASURED RUNS
     * ============================================================
     *
     * Alternate method execution order to reduce ordering bias.
     */
    for (int run = 0;
         run < NUM_RUNS;
         run++)
    {
        if ((run % 2) == 0)
        {
            measure_method1(
                N,
                F,
                H,
                row_ptr,
                col_idx,
                values,
                X,
                W,
                AX,
                Y1,
                &times_AX[run],
                &times_AXW_dense[run]);

            measure_method2(
                N,
                F,
                H,
                row_ptr,
                col_idx,
                values,
                X,
                W,
                XW,
                Y2,
                &times_XW[run],
                &times_AXW_sparse[run]);
        }
        else
        {
            measure_method2(
                N,
                F,
                H,
                row_ptr,
                col_idx,
                values,
                X,
                W,
                XW,
                Y2,
                &times_XW[run],
                &times_AXW_sparse[run]);

            measure_method1(
                N,
                F,
                H,
                row_ptr,
                col_idx,
                values,
                X,
                W,
                AX,
                Y1,
                &times_AX[run],
                &times_AXW_dense[run]);
        }

        totals_method1[run] =
            times_AX[run] +
            times_AXW_dense[run];

        totals_method2[run] =
            times_XW[run] +
            times_AXW_sparse[run];
    }


    /*
     * ============================================================
     * MEDIANS
     * ============================================================
     */
    const double median_AX =
        median_time(
            times_AX);

    const double median_AXW_dense =
        median_time(
            times_AXW_dense);

    const double median_XW =
        median_time(
            times_XW);

    const double median_AXW_sparse =
        median_time(
            times_AXW_sparse);

    const double median_method1 =
        median_time(
            totals_method1);

    const double median_method2 =
        median_time(
            totals_method2);


    /*
     * ============================================================
     * APPROXIMATE FLOP COUNTS
     * ============================================================
     */
    const double ops_sparse_AX =
        2.0 *
        (double)nnz *
        (double)F;

    const double ops_dense =
        2.0 *
        (double)N *
        (double)F *
        (double)H;

    const double ops_sparse_AXW =
        2.0 *
        (double)nnz *
        (double)H;


    /*
     * ============================================================
     * PRINT CONFIGURATION
     * ============================================================
     */
    printf(
        "Synthetic sequential timing test\n");

    printf(
        "--------------------------------\n");

    printf(
        "N = %d, F = %d, H = %d\n",
        N,
        F,
        H);

    printf(
        "Nonzeros per row = %d\n",
        nnz_per_row);

    printf(
        "Total nnz(A) = %d\n",
        nnz);

    printf(
        "Feature block size = %d\n",
        FEATURE_BLOCK);

    printf(
        "Prefetch distance = %d\n",
        PREFETCH_DISTANCE);

#if DENSE_IKJ == 0

    printf(
        "Dense loop order = i-j-k (control)\n");

#else

    printf(
        "Dense loop order = i-k-j (candidate)\n");

#endif

    printf(
        "Measured runs = %d "
        "(plus 1 warm-up)\n\n",
        NUM_RUNS);


    /*
     * ============================================================
     * INDIVIDUAL RUNS
     * ============================================================
     */
    printf(
        "Individual runs\n");

    printf(
        "---------------\n");

    for (int run = 0;
         run < NUM_RUNS;
         run++)
    {
        printf(
            "Run %d:\n",
            run + 1);

        printf(
            "  Method 1 (AX)W : %.6f s\n",
            totals_method1[run]);

        printf(
            "  Method 2 A(XW) : %.6f s\n",
            totals_method2[run]);
    }


    /*
     * ============================================================
     * MEDIAN RESULTS
     * ============================================================
     */
    printf(
        "\nMedian timing results\n");

    printf(
        "---------------------\n");

    printf(
        "Method 1: (AX)W\n");

    printf(
        "  A * X       : %.6f s  "
        "(%.3f GFLOP/s)\n",
        median_AX,
        calculate_gflops(
            ops_sparse_AX,
            median_AX));

    printf(
        "  (AX) * W    : %.6f s  "
        "(%.3f GFLOP/s)\n",
        median_AXW_dense,
        calculate_gflops(
            ops_dense,
            median_AXW_dense));

    printf(
        "  Median total: %.6f s\n\n",
        median_method1);


    printf(
        "Method 2: A(XW)\n");

    printf(
        "  X * W       : %.6f s  "
        "(%.3f GFLOP/s)\n",
        median_XW,
        calculate_gflops(
            ops_dense,
            median_XW));

    printf(
        "  A * (XW)    : %.6f s  "
        "(%.3f GFLOP/s)\n",
        median_AXW_sparse,
        calculate_gflops(
            ops_sparse_AXW,
            median_AXW_sparse));

    printf(
        "  Median total: %.6f s\n\n",
        median_method2);


    /*
     * ============================================================
     * COMPARE ASSOCIATIVE ORDERS
     * ============================================================
     */
    if (median_method1 >
        median_method2)
    {
        const double speedup =
            median_method1 /
            median_method2;

        const double percent_faster =
            (median_method1 -
             median_method2) /
            median_method1 *
            100.0;

        printf(
            "Comparison: A(XW) is %.3fx faster "
            "(%.2f%% lower runtime).\n",
            speedup,
            percent_faster);
    }
    else if (median_method2 >
             median_method1)
    {
        const double speedup =
            median_method2 /
            median_method1;

        const double percent_faster =
            (median_method2 -
             median_method1) /
            median_method2 *
            100.0;

        printf(
            "Comparison: (AX)W is %.3fx faster "
            "(%.2f%% lower runtime).\n",
            speedup,
            percent_faster);
    }
    else
    {
        printf(
            "Comparison: both methods have the same "
            "median runtime.\n");
    }


    /*
     * ============================================================
     * CORRECTNESS CHECK
     * ============================================================
     */
    if (matrices_match(
            N,
            H,
            Y1,
            Y2,
            1.0e-4f,
            1.0e-4f))
    {
        printf(
            "Verification: BOTH METHODS MATCH "
            "within tolerance.\n");
    }
    else
    {
        printf(
            "Verification: RESULTS DO NOT MATCH.\n");
    }


    /*
     * ============================================================
     * CLEANUP
     * ============================================================
     */
    free(values);
    free(col_idx);
    free(row_ptr);

    free(X);
    free(W);

    free(AX);
    free(XW);

    free(Y1);
    free(Y2);

    return EXIT_SUCCESS;
}
