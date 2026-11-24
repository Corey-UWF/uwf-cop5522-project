#!/usr/bin/env python3
"""Compare two N x H text dense matrices without loading them into memory."""
import argparse
import itertools
import math


def main():
    p = argparse.ArgumentParser()
    p.add_argument('reference')
    p.add_argument('candidate')
    p.add_argument('--atol', type=float, default=0.001)
    p.add_argument('--rtol', type=float, default=0.0001)
    args = p.parse_args()
    count = 0
    max_abs = 0.0
    with open(args.reference) as left, open(args.candidate) as right:
        a_head = left.readline().split()
        b_head = right.readline().split()
        if a_head != b_head or len(a_head) != 2:
            raise SystemExit('FAIL: dense matrix headers differ: {} vs {}'.format(a_head, b_head))
        n, h = map(int, a_head)
        for rowidx, (arow, brow) in enumerate(itertools.zip_longest(left, right), 1):
            if arow is None or brow is None:
                raise SystemExit('FAIL: different number of rows')
            avals = arow.split()
            bvals = brow.split()
            if len(avals) != h or len(bvals) != h:
                raise SystemExit('FAIL: wrong number of columns at row {}'.format(rowidx))
            for colidx, (av, bv) in enumerate(zip(avals, bvals), 1):
                a, b = float(av), float(bv)
                err = abs(a - b)
                if not math.isfinite(a) or not math.isfinite(b) or err > args.atol + args.rtol * abs(a):
                    raise SystemExit('FAIL: row {} col {} reference={} actual={} abs_err={}'.format(
                        rowidx, colidx, a, b, err))
                max_abs = max(max_abs, err)
                count += 1
        if rowidx != n:
            raise SystemExit('FAIL: expected {} rows, read {}'.format(n, rowidx))
    print('PASS: {} values; max abs error {:.7g}'.format(count, max_abs))


if __name__ == '__main__':
    main()
