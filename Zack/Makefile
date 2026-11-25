# COP5522 submission build (draft: requires basic_sequential.c from teammate)
# Load compiler/OpenMPI/OpenBLAS modules before building on Expanse.
CC ?= gcc
MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
CPPFLAGS ?=
LDFLAGS ?=
BLAS_LIBS ?= -lopenblas -lm
# Example: make BLAS_DIR=/path/to/openblas/lib
BLAS_DIR ?=
ifneq ($(strip $(BLAS_DIR)),)
  BLAS_FLAGS := -L$(BLAS_DIR) -Wl,-rpath,$(BLAS_DIR)
endif

# Change this variable to the actual filename supplied by your teammate.
PROJECT1_SRC ?= basic_sequential.c

.PHONY: all clean check-sources
all: project-1 project-2 project-3 project-4

check-sources:
	@for f in "$(PROJECT1_SRC)" project1_fileio_blas_adaptive_v21.c mpi_baseline.c project4_nodeshared.c; do \
	  test -f "$$f" || { echo "Missing source: $$f" >&2; exit 1; }; \
	done

project-1: $(PROJECT1_SRC)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_FLAGS) $(BLAS_LIBS)

project-2: project1_fileio_blas_adaptive_v21.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_FLAGS) $(BLAS_LIBS)

project-3: mpi_baseline.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_FLAGS) -lm

project-4: project4_nodeshared.c
	$(MPICC) $(CPPFLAGS) $(CFLAGS) -o $@ $< $(LDFLAGS) $(BLAS_FLAGS) $(BLAS_LIBS)

clean:
	rm -f project-1 project-2 project-3 project-4
