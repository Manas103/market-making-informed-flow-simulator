"""Aggregates per-session CSVs from run_sessions into the three measured
claims: the adverse-selection give-back ratio, the inventory-variance cut
from skewing, and the gross-capture cost of skewing.

Usage:
    python aggregate.py fixed_020.csv skew_020.csv
"""
import csv
import statistics
import sys


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_int(rows, key):
    return [int(r[key]) for r in rows]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    fixed = load(sys.argv[1])
    skew = load(sys.argv[2])

    gross_fixed = to_int(fixed, "gross_capture")
    adverse_fixed = to_int(fixed, "adverse_cost")
    inv_fixed = to_int(fixed, "final_inventory")

    gross_skew = to_int(skew, "gross_capture")
    inv_skew = to_int(skew, "final_inventory")

    giveback = sum(adverse_fixed) / sum(gross_fixed)
    stdev_fixed = statistics.pstdev(inv_fixed)
    stdev_skew = statistics.pstdev(inv_skew)
    variance_cut = stdev_fixed / stdev_skew if stdev_skew else float("inf")
    capture_cost = (sum(gross_fixed) - sum(gross_skew)) / sum(gross_fixed)

    print(f"sessions: fixed={len(fixed)} skew={len(skew)}")
    print(f"gross_capture (fixed): sum={sum(gross_fixed)} mean={statistics.mean(gross_fixed):.1f}")
    print(f"gross_capture (skew):  sum={sum(gross_skew)} mean={statistics.mean(gross_skew):.1f}")
    print(f"adverse_cost (fixed):  sum={sum(adverse_fixed)}")
    print(f"adverse-selection give-back (fixed, informed_frac from input file): {giveback:.4f}")
    print(f"final_inventory stdev: fixed={stdev_fixed:.3f} skew={stdev_skew:.3f}")
    print(f"inventory stdev cut (fixed / skew): {variance_cut:.3f}x")
    print(f"gross capture cost of skewing: {capture_cost:.4f}")


if __name__ == "__main__":
    main()
