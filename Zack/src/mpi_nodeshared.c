/*
 * mpi_nodeshared.c - Y = A X W with one shared copy of the data per node.
 *
 * Why it is faster than the earlier MPI versions:
 *   1. All ranks on a node share ONE copy of A, X, W, the dense intermediate
 *      and Y (an MPI-3 shared-memory window). The old versions copied the dense
 *      intermediate to every rank, which is limited by memory bandwidth.
 *   2. Every rank on the root node computes a slice of X*W. The old versions
 *      computed all of X*W on rank 0 while the other ranks waited.
 *   3. Rows of A are split by estimated work (nonzeros plus a per-row cost).
 *   4. Only one rank per node (the "leader") uses the network.
 *
 * Usage: mpirun -np P ./mpi-nodeshared A.csr X.dense W.dense Y.dense
 * Order: A(XW) if H <= F, otherwise (AX)W, as in Rolando's sequential v21.
 */
#include <mpi.h>

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* OpenBLAS declarations, because some systems have no cblas.h. */
enum { CblasRowMajor = 101, CblasNoTrans = 111 };
extern void cblas_sgemm(const int order, const int trans_a, const int trans_b,
                        const int m, const int n, const int k, const float alpha,
                        const float *a, const int lda, const float *b,
                        const int ldb, const float beta, float *c, const int ldc);
extern void openblas_set_num_threads(int threads);
extern int openblas_get_num_threads(void);

typedef struct {
    int n;        /* A is n x n */
    int nnz;      /* nonzeros in A */
    int features; /* X is n x features */
    int hidden;   /* W is features x hidden, Y is n x hidden */
} Dims;

/* Pointers into the node's shared window. */
typedef struct {
    int *row_ptr;
    int *col_idx;
    float *values;
    float *x;
    float *w;
    float *xw; /* n x hidden, only used for A(XW) */
    float *y;
} Data;

typedef struct {
    int world_rank, world_size;
    MPI_Comm node;          /* the ranks on this node */
    int node_rank;
    int node_size;
    MPI_Comm leaders;       /* one rank per node; MPI_COMM_NULL on the others */
    int node_id, nodes;     /* number of this node, and how many nodes there are */
    MPI_Win win;            /* shared window owned by `node` */
} Node;

/* Text of the three input files, read by rank 0 only. */
typedef struct {
    char *a_text, *x_text, *w_text; /* owned buffers */
    char *a, *x, *w;                /* current parse positions */
} Inputs;

static void fail(const char *message)
{
    fprintf(stderr, "Error: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
}

static void *allocate(size_t count, size_t size)
{
    void *memory = calloc(count == 0 ? 1 : count, size);
    if (memory == NULL) {
        fail("Out of memory");
    }
    return memory;
}

/* ---------- Reading and writing files ---------- */

static char *read_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s\n", path);
        fail("Input file missing");
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    char *text = allocate((size_t)size + 1, 1);
    if (fread(text, 1, (size_t)size, file) != (size_t)size) {
        fail("Could not read input file");
    }
    fclose(file);
    return text;
}

static int next_int(char **position)
{
    char *end;
    long value = strtol(*position, &end, 10);
    if (end == *position || value < INT_MIN || value > INT_MAX) {
        fail("Bad integer in input file");
    }
    *position = end;
    return (int)value;
}

static float next_float(char **position)
{
    char *end;
    float value = strtof(*position, &end);
    if (end == *position) {
        fail("Bad number in input file");
    }
    *position = end;
    return value;
}

/* Reads the file headers (sizes) and leaves the parse positions at the data. */
static Dims read_headers(Inputs *in, char **argv)
{
    in->a = in->a_text = read_file(argv[1]);
    in->x = in->x_text = read_file(argv[2]);
    in->w = in->w_text = read_file(argv[3]);

    Dims d;
    const int rows = next_int(&in->a);
    const int columns = next_int(&in->a);
    d.n = rows;
    d.nnz = next_int(&in->a);
    const int x_rows = next_int(&in->x);
    d.features = next_int(&in->x);
    const int w_rows = next_int(&in->w);
    d.hidden = next_int(&in->w);

    if (d.n <= 0 || columns != d.n || d.nnz < 0 || d.features <= 0 ||
        d.hidden <= 0 || x_rows != d.n || w_rows != d.features) {
        fail("Expected A = NxN, X = NxF and W = FxH with positive sizes");
    }
    /* Keeps every MPI count and BLAS dimension inside a 32-bit int. */
    if ((long long)d.n * d.features > INT_MAX || (long long)d.n * d.hidden > INT_MAX ||
        (long long)d.features * d.hidden > INT_MAX) {
        fail("Matrices are too large for this program");
    }
    return d;
}

static void read_bodies(Inputs *in, const Dims *d, const Data *s)
{
    s->row_ptr[0] = next_int(&in->a);
    for (int i = 1; i <= d->n; i++) {
        s->row_ptr[i] = next_int(&in->a);
        if (s->row_ptr[i] < s->row_ptr[i - 1] || s->row_ptr[i] > d->nnz) {
            fail("CSR row pointers must not decrease and must end at nnz");
        }
    }
    if (s->row_ptr[0] != 0 || s->row_ptr[d->n] != d->nnz) {
        fail("CSR row pointers must start at 0 and end at nnz");
    }
    for (int p = 0; p < d->nnz; p++) {
        s->col_idx[p] = next_int(&in->a);
        if (s->col_idx[p] < 0 || s->col_idx[p] >= d->n) {
            fail("CSR column index out of range");
        }
    }
    for (int p = 0; p < d->nnz; p++) {
        s->values[p] = next_float(&in->a);
    }
    for (size_t i = 0; i < (size_t)d->n * d->features; i++) {
        s->x[i] = next_float(&in->x);
    }
    for (size_t i = 0; i < (size_t)d->features * d->hidden; i++) {
        s->w[i] = next_float(&in->w);
    }
    free(in->a_text);
    free(in->x_text);
    free(in->w_text);
}

static void write_dense(const char *path, const float *values, int rows, int columns)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fail("Cannot open the output file");
    }
    fprintf(file, "%d %d\n", rows, columns);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < columns; col++) {
            fprintf(file, "%.9g%c", values[(size_t)row * columns + col],
                    col + 1 == columns ? '\n' : ' ');
        }
    }
    if (ferror(file) || fclose(file) != 0) {
        fail("Could not finish writing the output file");
    }
}

/* ---------- Nodes and shared memory ---------- */

static Node split_nodes(void)
{
    Node nd = {0};
    MPI_Comm_rank(MPI_COMM_WORLD, &nd.world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nd.world_size);

    /* NODESHARED_TEST_NODES=k pretends there are k nodes (for testing on one machine). */
    const char *pretend = getenv("NODESHARED_TEST_NODES");
    if (pretend != NULL && atoi(pretend) > 0) {
        MPI_Comm_split(MPI_COMM_WORLD, nd.world_rank % atoi(pretend), nd.world_rank, &nd.node);
    } else {
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, nd.world_rank,
                            MPI_INFO_NULL, &nd.node);
    }
    MPI_Comm_rank(nd.node, &nd.node_rank);
    MPI_Comm_size(nd.node, &nd.node_size);

    /* The leader is the lowest rank on each node, so rank 0 leads the root node. */
    MPI_Comm_split(MPI_COMM_WORLD, nd.node_rank == 0 ? 0 : MPI_UNDEFINED, nd.world_rank,
                   &nd.leaders);
    int ids[2] = {0, 0};
    if (nd.node_rank == 0) {
        MPI_Comm_rank(nd.leaders, &ids[0]);
        MPI_Comm_size(nd.leaders, &ids[1]);
    }
    MPI_Bcast(ids, 2, MPI_INT, 0, nd.node);
    nd.node_id = ids[0];
    nd.nodes = ids[1];
    return nd;
}

/* Allocates one window per node and carves it into the arrays we need. */
static Data make_shared_data(const Dims *d, Node *nd)
{
    const size_t n = (size_t)d->n, nnz = (size_t)d->nnz;
    const size_t f = (size_t)d->features, h = (size_t)d->hidden;
    const size_t bytes[7] = {(n + 1) * sizeof(int), nnz * sizeof(int),
                             nnz * sizeof(float), n * f * sizeof(float),
                             f * h * sizeof(float), n * h * sizeof(float),
                             n * h * sizeof(float)};
    size_t offset[7];
    size_t total = 0;
    for (int i = 0; i < 7; i++) {
        offset[i] = total;
        total += (bytes[i] + 63) / 64 * 64;
    }

    char *base;
    MPI_Win_allocate_shared(nd->node_rank == 0 ? (MPI_Aint)total : 0, 1,
                            MPI_INFO_NULL, nd->node, &base, &nd->win);
    if (nd->node_rank != 0) {
        MPI_Aint size;
        int unit;
        MPI_Win_shared_query(nd->win, 0, &size, &unit, &base);
    }
    MPI_Win_lock_all(MPI_MODE_NOCHECK, nd->win);

    Data s;
    s.row_ptr = (int *)(base + offset[0]);
    s.col_idx = (int *)(base + offset[1]);
    s.values = (float *)(base + offset[2]);
    s.x = (float *)(base + offset[3]);
    s.w = (float *)(base + offset[4]);
    s.xw = (float *)(base + offset[5]);
    s.y = (float *)(base + offset[6]);
    return s;
}

/* Makes writes by one rank on the node visible to the others. */
static void node_sync(const Node *nd)
{
    MPI_Win_sync(nd->win);
    MPI_Barrier(nd->node);
    MPI_Win_sync(nd->win);
}

/* Broadcast from the root node's leader to the other leaders, in 1 GiB pieces. */
static void broadcast_between_nodes(const Node *nd, void *buffer, size_t bytes)
{
    if (nd->node_rank != 0) {
        return;
    }
    char *position = buffer;
    while (bytes > 0) {
        const int piece = bytes > ((size_t)1 << 30) ? (1 << 30) : (int)bytes;
        MPI_Bcast(position, piece, MPI_BYTE, 0, nd->leaders);
        position += piece;
        bytes -= (size_t)piece;
    }
}

/* ---------- Work distribution ---------- */

/* Ranks are numbered node by node, so each node owns one block of rows. */
static int *first_worker_of_each_node(const Node *nd)
{
    int *node_of_rank = allocate((size_t)nd->world_size, sizeof(int));
    MPI_Allgather(&nd->node_id, 1, MPI_INT, node_of_rank, 1, MPI_INT, MPI_COMM_WORLD);
    int *first = allocate((size_t)nd->nodes + 1, sizeof(int));
    for (int r = 0; r < nd->world_size; r++) {
        first[node_of_rank[r] + 1]++;
    }
    for (int k = 0; k < nd->nodes; k++) {
        first[k + 1] += first[k];
    }
    free(node_of_rank);
    return first;
}

/*
 * Splits the rows into `workers` contiguous blocks with about equal work.
 * Row cost = its nonzeros + row_cost. For (AX)W every row also pays for a
 * dense multiply with W (about hidden/2 nonzero-equivalents, the same rule the
 * earlier hybrid version used). Block w is rows bounds[w] .. bounds[w+1]-1.
 */
static void partition_rows(const Dims *d, const int *row_ptr, int use_xw_first,
                           int workers, int *bounds)
{
    const long long row_cost = 1 + (use_xw_first ? 0 : d->hidden / 2);
    const long long total = d->nnz + row_cost * d->n;
    int row = 0;
    bounds[0] = 0;
    for (int w = 1; w <= workers; w++) {
        while (row < d->n && (row_ptr[row] + row_cost * row) * workers < total * w) {
            row++;
        }
        bounds[w] = row;
    }
}

/* ---------- Arithmetic ---------- */

/* out = (rows row_begin..row_end-1 of A) * b, where b has `width` columns. */
static void sparse_times_dense(const Data *s, int row_begin, int row_end,
                               const float *b, int width, float *out)
{
    for (int row = row_begin; row < row_end; row++) {
        float *restrict y = out + (size_t)(row - row_begin) * width;
        memset(y, 0, (size_t)width * sizeof(float));
        for (int p = s->row_ptr[row]; p < s->row_ptr[row + 1]; p++) {
            const float *restrict x = b + (size_t)s->col_idx[p] * width;
            const float value = s->values[p];
            for (int j = 0; j < width; j++) {
                y[j] += value * x[j];
            }
        }
    }
}

/* c (m x n) = a (m x k) * b (k x n), all row-major. */
static void dense_multiply(int m, int k, int n, const float *a, const float *b, float *c)
{
    if (m > 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                    a, k, b, n, 0.0f, c, n);
    }
}

/* ---------- Report ---------- */

static void print_report(const Dims *d, const Data *s, const Node *nd, int use_xw_first,
                         const int *bounds, const double *phase_max, double compute_min,
                         double compute_sum, double total, double inclusive,
                         double read_time, double write_time)
{
    int rows_min = d->n, rows_max = 0, nnz_min = d->nnz, nnz_max = 0;
    for (int w = 0; w < nd->world_size; w++) {
        const int rows = bounds[w + 1] - bounds[w];
        const int nnz = s->row_ptr[bounds[w + 1]] - s->row_ptr[bounds[w]];
        rows_min = rows < rows_min ? rows : rows_min;
        rows_max = rows > rows_max ? rows : rows_max;
        nnz_min = nnz < nnz_min ? nnz : nnz_min;
        nnz_max = nnz > nnz_max ? nnz : nnz_max;
    }
    const double nnz_average = (double)d->nnz / nd->world_size;

    printf("Order=%s Ranks=%d Nodes=%d OpenBLAS_threads=%d\n",
           use_xw_first ? "A(XW)" : "(AX)W", nd->world_size, nd->nodes,
           openblas_get_num_threads());
    printf("CSR distribution = %.9f seconds\nRoot XW = %.9f seconds\n"
           "Dense broadcast = %.9f seconds\nLocal computation = %.9f seconds\n"
           "Output gathering = %.9f seconds\nMPI elapsed time = %.9f seconds\n"
           "MPI setup-inclusive time = %.9f seconds\n",
           phase_max[0], phase_max[1], phase_max[2], phase_max[3], phase_max[4],
           total, inclusive);
    printf("Rows per rank: min=%d max=%d\n", rows_min, rows_max);
    printf("Nonzeros per rank: min=%d max=%d average=%.6f\n", nnz_min, nnz_max,
           nnz_average);
    if (d->nnz > 0) {
        printf("Nonzero imbalance ratio = %.6f\n", nnz_max / nnz_average);
    } else {
        printf("Nonzero imbalance ratio = N/A (zero nonzeros)\n");
    }
    printf("Rank-local computation: min=%.9f max=%.9f average=%.9f seconds\n",
           compute_min, phase_max[3], compute_sum / nd->world_size);
    printf("Input reading = %.9f seconds (not in MPI elapsed time)\n", read_time);
    printf("Output writing = %.9f seconds (not in MPI elapsed time)\n", write_time);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    if (argc != 5) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            fprintf(stderr, "Usage: %s A.csr X.dense W.dense Y.dense\n", argv[0]);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    openblas_set_num_threads(1);

    /* Setup (communicators, shared window) is timed separately from the timed section. */
    double setup_time = 0.0;
    double t = MPI_Wtime();
    Node node = split_nodes();
    setup_time += MPI_Wtime() - t;
    const int is_root = node.world_rank == 0;

    /* Rank 0 reads the files straight into the shared window (not timed). */
    Inputs in = {0};
    Dims d = {0};
    double read_time = 0.0;
    if (is_root) {
        t = MPI_Wtime();
        d = read_headers(&in, argv);
        read_time += MPI_Wtime() - t;
    }
    MPI_Bcast(&d, (int)sizeof d, MPI_BYTE, 0, MPI_COMM_WORLD);
    const int use_xw_first = d.hidden <= d.features;
    const int width = use_xw_first ? d.hidden : d.features; /* columns of the dense input */

    t = MPI_Wtime();
    Data s = make_shared_data(&d, &node);
    int *first_worker = first_worker_of_each_node(&node);
    const int worker = first_worker[node.node_id] + node.node_rank;
    int *bounds = allocate((size_t)node.world_size + 1, sizeof(int));
    setup_time += MPI_Wtime() - t;

    if (is_root) {
        t = MPI_Wtime();
        read_bodies(&in, &d, &s);
        read_time += MPI_Wtime() - t;
    }
    node_sync(&node);
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Timed section ---- */
    double phase[5] = {0};
    const double start = MPI_Wtime();
    t = start;

    /* 1. Other nodes receive A; every rank learns its rows. */
    if (node.nodes > 1) {
        broadcast_between_nodes(&node, s.row_ptr, ((size_t)d.n + 1) * sizeof(int));
        broadcast_between_nodes(&node, s.col_idx, (size_t)d.nnz * sizeof(int));
        broadcast_between_nodes(&node, s.values, (size_t)d.nnz * sizeof(float));
        node_sync(&node);
    }
    partition_rows(&d, s.row_ptr, use_xw_first, node.world_size, bounds);
    phase[0] = MPI_Wtime() - t;

    /* 2. A(XW) only: the root node computes XW, each rank doing a slice of rows. */
    t = MPI_Wtime();
    if (use_xw_first) {
        if (node.node_id == 0) {
            const int first = (int)((long long)d.n * node.node_rank / node.node_size);
            const int last = (int)((long long)d.n * (node.node_rank + 1) / node.node_size);
            dense_multiply(last - first, d.features, d.hidden,
                           s.x + (size_t)first * d.features, s.w,
                           s.xw + (size_t)first * d.hidden);
        }
        node_sync(&node);
    }
    phase[1] = MPI_Wtime() - t;

    /* 3. Other nodes receive the dense input (XW, or X and W for (AX)W). */
    t = MPI_Wtime();
    if (node.nodes > 1) {
        broadcast_between_nodes(&node, use_xw_first ? s.xw : s.x,
                                (size_t)d.n * width * sizeof(float));
        if (!use_xw_first) {
            broadcast_between_nodes(&node, s.w, (size_t)d.features * d.hidden * sizeof(float));
        }
        node_sync(&node);
    }
    phase[2] = MPI_Wtime() - t;

    /* 4. Each rank computes its own rows of Y. */
    t = MPI_Wtime();
    const int row_begin = bounds[worker], row_end = bounds[worker + 1];
    float *y_rows = s.y + (size_t)row_begin * d.hidden;
    if (use_xw_first) {
        sparse_times_dense(&s, row_begin, row_end, s.xw, d.hidden, y_rows);
    } else {
        float *ax = allocate((size_t)(row_end - row_begin) * d.features, sizeof(float));
        sparse_times_dense(&s, row_begin, row_end, s.x, d.features, ax);
        dense_multiply(row_end - row_begin, d.features, d.hidden, ax, s.w, y_rows);
        free(ax);
    }
    phase[3] = MPI_Wtime() - t;
    node_sync(&node);

    /* 5. Leaders send their node's rows of Y to rank 0 (nothing to do on one node). */
    t = MPI_Wtime();
    if (node.nodes > 1 && node.node_rank == 0) {
        int *counts = allocate((size_t)node.nodes, sizeof(int));
        int *displs = allocate((size_t)node.nodes, sizeof(int));
        for (int k = 0; k < node.nodes; k++) {
            const int first_row = bounds[first_worker[k]];
            displs[k] = first_row * d.hidden;
            counts[k] = (bounds[first_worker[k + 1]] - first_row) * d.hidden;
        }
        MPI_Gatherv(node.node_id == 0 ? MPI_IN_PLACE : s.y + displs[node.node_id],
                    counts[node.node_id], MPI_FLOAT, s.y, counts, displs, MPI_FLOAT, 0,
                    node.leaders);
        free(counts);
        free(displs);
    }
    phase[4] = MPI_Wtime() - t;
    const double elapsed = MPI_Wtime() - start;
    /* ---- End of timed section ---- */

    double phase_max[5], total, inclusive, compute_min, compute_sum;
    const double setup_and_elapsed = setup_time + elapsed;
    MPI_Reduce(phase, phase_max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed, &total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&setup_and_elapsed, &inclusive, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&phase[3], &compute_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&phase[3], &compute_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (is_root) {
        const double write_start = MPI_Wtime();
        write_dense(argv[4], s.y, d.n, d.hidden);
        print_report(&d, &s, &node, use_xw_first, bounds, phase_max, compute_min,
                     compute_sum, total, inclusive, read_time, MPI_Wtime() - write_start);
    }

    MPI_Win_unlock_all(node.win);
    MPI_Win_free(&node.win);
    MPI_Comm_free(&node.node);
    if (node.leaders != MPI_COMM_NULL) {
        MPI_Comm_free(&node.leaders);
    }
    free(first_worker);
    free(bounds);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
