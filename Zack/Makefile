MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
BLAS_CPPFLAGS ?=
BLAS_LDFLAGS ?=
BLAS_LIBS ?= -lopenblas

.PHONY: all clean

all: mpi-baseline generate-regular-inputs

mpi-baseline: mpi_baseline.c
	$(MPICC) $(CFLAGS) -o $@ $<

mpi-adaptive: mpi_adaptive.c
	$(MPICC) $(CPPFLAGS) $(BLAS_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_LDFLAGS) $(BLAS_LIBS)

mpi-balanced: mpi_balanced.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_LDFLAGS) $(BLAS_LIBS) -lm

test-balanced-partition: test_balanced_partition.c mpi_balanced.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_LDFLAGS) $(BLAS_LIBS) -lm

mpi-nodeshared: src/mpi_nodeshared.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_LDFLAGS) $(BLAS_LIBS) -lm

generate-regular-inputs: generate_regular_inputs.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f mpi-baseline mpi-adaptive mpi-balanced mpi-nodeshared test-balanced-partition generate-regular-inputs
