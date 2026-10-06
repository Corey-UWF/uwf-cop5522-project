MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic

.PHONY: all clean

all: mpi-baseline generate-regular-inputs

mpi-baseline: mpi_baseline.c
	$(MPICC) $(CFLAGS) -o $@ $<

generate-regular-inputs: generate_regular_inputs.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f mpi-baseline generate-regular-inputs
