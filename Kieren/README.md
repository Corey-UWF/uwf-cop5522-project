# Kieren's Additions
# Performance Benchmarking and Experimental Analysis

**Contributor: Kieren Gregory**

My contribution will focus on benchmarking and analyzing the performance of the sequential, OpenMP, and MPI implementations. Since power-law graphs contain a small number of highly connected vertices and many vertices with relatively few connections, the amount of work between sparse matrix rows can vary significantly. This can create challenges involving load imbalance, irregular memory access, cache locality, and parallel scalability.

## Benchmarking Methodology

To provide a fair comparison, each implementation will be tested using consistent matrix sizes, input data, compiler settings, and optimization flags. Important configurations will be executed multiple times to reduce the effect of measurement variation.

The main metrics collected will include **execution time, speedup, parallel efficiency, and scalability**.

Speedup will be calculated as:

$$
S_p = \frac{T_1}{T_p}
$$

Parallel efficiency will be calculated as:

$$
E_p = \frac{S_p}{p}
$$

where $T_1$ is the sequential execution time and $T_p$ is the execution time using $p$ threads or processes.

## OpenMP Evaluation

The OpenMP implementation will be tested using different thread counts and scheduling strategies, particularly `static`, `dynamic`, and `guided`. Because power-law matrices can have very different numbers of nonzeros per row, dynamic scheduling may improve load balancing while introducing additional scheduling overhead. The experiments will determine which strategy provides the best performance for our workload.

## MPI Evaluation

The MPI implementation will be tested using different process counts and, when possible, multiple nodes. The analysis will compare **row-based partitioning** with **nonzero-balanced partitioning**. Since equal numbers of rows may contain very different numbers of nonzeros, balancing work according to nonzeros may improve performance. Communication and synchronization overhead will also be considered when evaluating MPI scalability.

## Results and Analysis

Results will be organized into tables and graphs comparing execution time, speedup, and efficiency. Important comparisons will include OpenMP scheduling strategies, MPI partitioning strategies, and overall sequential vs. OpenMP vs. MPI performance.

The final analysis will identify where additional threads or processes stop producing useful speedup and whether the primary limitation is load imbalance, memory behavior, scheduling overhead, or MPI communication.

## Planned Deliverables

- Benchmark methodology
- Sequential baseline measurements
- OpenMP scaling results
- MPI scaling results
- Timing data
- Performance graphs
- Analysis and conclusions
