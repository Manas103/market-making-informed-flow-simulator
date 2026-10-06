"""Aggregates per-session CSVs from run_options_sessions into the three
measured claims for the options-chain extension: the hedge-band stdev cut
and its cost, and the vega-scaled-widening adverse-selection-share cut.

Usage:
    python aggregate_options.py hedged_flat.csv unhedged_flat.csv h2_flat.csv h2_vega.csv

hedged_flat / unhedged_flat: same base half spread (2 ticks), hedge_band
0.10 vs effectively infinite, isolates the hedge-band effect.

h2_flat / h2_vega: same wider base half spread (10 ticks) and hedge band,
flat vs vega-scaled widening, isolates the widening effect.
"""
import csv
import statistics
import sys


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return [float(r[key]) for r in rows]


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(2)
    hedged = load(sys.argv[1])
    unhedged = load(sys.argv[2])
    flat = load(sys.argv[3])
    vega = load(sys.argv[4])

    pnl_hedged = col(hedged, "pnl_oracle")
    pnl_unhedged = col(unhedged, "pnl_oracle")
    stdev_hedged = statistics.pstdev(pnl_hedged)
    stdev_unhedged = statistics.pstdev(pnl_unhedged)
    stdev_ratio = stdev_unhedged / stdev_hedged if stdev_hedged else float("inf")

    edge_hedged = sum(col(hedged, "edge_captured"))
    slippage_hedged = sum(col(hedged, "hedge_slippage"))
    hedge_cost_pct = 100.0 * slippage_hedged / edge_hedged

    edge_flat = sum(col(flat, "edge_captured"))
    adv_flat = sum(col(flat, "adverse_selection"))
    adv_pct_flat = 100.0 * adv_flat / edge_flat

    edge_vega = sum(col(vega, "edge_captured"))
    adv_vega = sum(col(vega, "adverse_selection"))
    adv_pct_vega = 100.0 * adv_vega / edge_vega

    print(f"sessions: hedged={len(hedged)} unhedged={len(unhedged)} flat={len(flat)} vega={len(vega)}")
    print()
    print("-- hedge band --")
    print(f"end-of-session pnl_oracle stdev: unhedged={stdev_unhedged:.4f} hedged={stdev_hedged:.4f}")
    print(f"stdev cut (unhedged / hedged): {stdev_ratio:.3f}x")
    print(f"gross edge_captured (hedged): {edge_hedged:.2f}")
    print(f"hedge_slippage (hedged): {slippage_hedged:.2f}")
    print(f"cost of hedging as % of gross edge: {hedge_cost_pct:.2f}%")
    print()
    print("-- vega-scaled widening --")
    print(f"gross edge_captured: flat={edge_flat:.2f} vega={edge_vega:.2f}")
    print(f"adverse_selection: flat={adv_flat:.2f} vega={adv_vega:.2f}")
    print(f"adverse selection as % of gross edge: flat={adv_pct_flat:.2f}% vega={adv_pct_vega:.2f}%")


if __name__ == "__main__":
    main()
