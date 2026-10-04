MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic

.PHONY: all clean

all: mpi-baseline

mpi-baseline: mpi_baseline.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f mpi-baseline
