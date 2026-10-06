#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 4000, F = 128, H = 64, NONZEROS_PER_ROW = 16 };

static FILE *open_output(const char *path)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "Cannot create %s: ", path);
        perror(NULL);
        exit(EXIT_FAILURE);
    }
    return file;
}

static void finish_output(FILE *file, const char *path)
{
    int failed = ferror(file);
    if (fclose(file) != 0) {
        failed = 1;
    }
    if (failed) {
        fprintf(stderr, "Failed to write %s; discard the incomplete output\n", path);
        exit(EXIT_FAILURE);
    }
}

static void write_sparse(const char *path)
{
    FILE *file = open_output(path);
    fprintf(file, "%d %d %d\n", N, N, N * NONZEROS_PER_ROW);
    for (int row = 0; row <= N; ++row) {
        fprintf(file, "%d%c", row * NONZEROS_PER_ROW, row == N ? '\n' : ' ');
    }
    for (int row = 0; row < N; ++row) {
        for (int p = 0; p < NONZEROS_PER_ROW; ++p) {
            fprintf(file, "%d%c", (row + p * 37) % N,
                    p + 1 == NONZEROS_PER_ROW ? '\n' : ' ');
        }
    }
    for (int row = 0; row < N; ++row) {
        for (int p = 0; p < NONZEROS_PER_ROW; ++p) {
            float value = 0.10f + 0.01f * (float)((p % 9) + 1);
            fprintf(file, "%.9g%c", value,
                    p + 1 == NONZEROS_PER_ROW ? '\n' : ' ');
        }
    }
    finish_output(file, path);
}

static void write_x(const char *path)
{
    FILE *file = open_output(path);
    fprintf(file, "%d %d\n", N, F);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < F; ++j) {
            float value = (float)(((i + 3 * j) % 17) + 1) / 17.0f;
            fprintf(file, "%.9g%c", value, j + 1 == F ? '\n' : ' ');
        }
    }
    finish_output(file, path);
}

static void write_w(const char *path)
{
    FILE *file = open_output(path);
    fprintf(file, "%d %d\n", F, H);
    for (int i = 0; i < F; ++i) {
        for (int j = 0; j < H; ++j) {
            float value = (float)(((2 * i + 5 * j) % 19) + 1) / 19.0f;
            fprintf(file, "%.9g%c", value, j + 1 == H ? '\n' : ' ');
        }
    }
    finish_output(file, path);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s A.csr X.dense W.dense\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], argv[2]) == 0 || strcmp(argv[1], argv[3]) == 0 ||
        strcmp(argv[2], argv[3]) == 0) {
        fprintf(stderr, "Use three distinct output paths\n");
        return EXIT_FAILURE;
    }

    write_sparse(argv[1]);
    write_x(argv[2]);
    write_w(argv[3]);
    printf("Generated v12-derived regular inputs: N=%d F=%d H=%d nnz=%d\n",
           N, F, H, N * NONZEROS_PER_ROW);
    return EXIT_SUCCESS;
}
