#!/usr/bin/env python3
"""Print multi-node medians by dataset, order, ranks, nodes, and method."""
import csv
import statistics
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.exit('Usage: python3 summarize_multinode.py expanse-benchmarks/mpi-multinode-*.csv')
values = defaultdict(list)
for path in sys.argv[1:]:
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row['dataset'], row['order'], int(row['ranks']), int(row['nodes']), row['method'])
            values[key].append(row)
print('dataset,order,ranks,nodes,method,n,partition_ms,mpi_old_ms,setup_inclusive_ms,external_wall_ms')
for key, rows in sorted(values.items()):
    medians = [1000 * statistics.median(float(r[name]) for r in rows)
               for name in ('partition_setup_s', 'mpi_elapsed_s', 'mpi_setup_inclusive_s', 'wall_s')]
    print('{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f}'.format(
        *key, len(rows), *medians))
