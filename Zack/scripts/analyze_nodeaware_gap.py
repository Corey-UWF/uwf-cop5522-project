#!/usr/bin/env python3
"""Compare paired node-aware MPI benchmark medians and isolate timing differences.
Usage: python3 analyze_nodeaware_gap.py expanse-benchmarks/nodeaware-*.csv
"""
import argparse
import csv
import statistics
from collections import defaultdict

PHS = ('setup_inclusive_ms','mpi_elapsed_ms','distribution_ms','local_compute_ms','external_wall_ms')

def med(rows, name):
    return statistics.median(float(row[name]) for row in rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', nargs='+', help='Node-aware benchmark CSVs')
    args=p.parse_args()
    grouped=defaultdict(lambda: defaultdict(list))
    for path in args.csv:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                key=(int(row['nodes']), row['dataset'], row['order'], int(row['ranks']))
                grouped[key][row['variant']].append(row)
    for key in sorted(grouped):
        groups=grouped[key]
        print('\n{} nodes | {} | {} | {} ranks'.format(*key))
        for variant in ('reference','nodeaware'):
            print('  {}: {} measurements'.format(variant, len(groups[variant])))
        if not groups['reference'] or not groups['nodeaware']:
            print('  INCOMPLETE: one variant missing')
            continue
        print('  {:24s} {:>12s} {:>12s} {:>12s}'.format('Metric','Reference','Node-aware','Difference'))
        for name in PHS:
            a=med(groups['reference'],name)
            b=med(groups['nodeaware'],name)
            print('  {:24s} {:12.3f} {:12.3f} {:12.3f}'.format(name,a,b,b-a))
        # setup outside main timed MPI interval (includes rank partitioning and barriers)
        a=statistics.median(float(r['setup_inclusive_ms'])-float(r['mpi_elapsed_ms']) for r in groups['reference'])
        b=statistics.median(float(r['setup_inclusive_ms'])-float(r['mpi_elapsed_ms']) for r in groups['nodeaware'])
        print('  {:24s} {:12.3f} {:12.3f} {:12.3f}'.format('setup_minus_mpi_ms',a,b,b-a))
        # Paired repetition deltas account for the alternating order experiment
        by_rep={v:{int(r['repetition']):r for r in rows} for v,rows in groups.items()}
        common=sorted(set(by_rep['reference']) & set(by_rep['nodeaware']))
        if common:
            d=[float(by_rep['nodeaware'][rep]['setup_inclusive_ms'])-float(by_rep['reference'][rep]['setup_inclusive_ms']) for rep in common]
            print('  paired setup-inclusive delta (nodeaware-reference): median {:.3f} ms; {} pairs'.format(statistics.median(d),len(d)))
        print('  NOTE: MPI phase maxima can overlap; do not sum them to infer wall runtime.')

if __name__=='__main__':
    main()
