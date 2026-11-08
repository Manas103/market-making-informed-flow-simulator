"""Runs the 500-session comparison the README's "Measured results
(risk-budgeted optimization extension)" section reports: the uncoordinated
contract-by-contract baseline against the book-level convex-optimized
policy, over the same 120-contract chain and the same flow model.

Usage:
    python -m risk_budget.run_benchmark [--sessions N]
"""
import argparse
import time

import numpy as np

from . import simulate as sim
from .optimizer import BookOptimizer


def summarize(name, results, elapsed):
    total_ticks = sum(r.total_ticks for r in results)
    in_vega = sum(r.ticks_in_vega_budget for r in results)
    in_gamma = sum(r.ticks_in_gamma_budget for r in results)
    in_delta = sum(r.ticks_in_delta_budget for r in results)
    in_all = sum(r.ticks_in_all_budgets for r in results)
    gross_edge = sum(r.edge_captured for r in results)
    fills = sum(r.fills for r in results)
    declined = sum(r.declined_fills for r in results)
    max_vega = max(r.max_abs_net_vega for r in results)

    print(f"--- {name} ---")
    print(f"sessions: {len(results)}  total_ticks: {total_ticks}  elapsed: {elapsed:.2f}s")
    print(f"gross edge captured: {gross_edge:.4f}")
    print(f"fills: {fills}  declined (no quoted size on the needed side): {declined}")
    print(f"ticks with net vega inside budget: {in_vega} / {total_ticks} ({100*in_vega/total_ticks:.2f}%)")
    print(f"ticks with net gamma inside budget: {in_gamma} / {total_ticks} ({100*in_gamma/total_ticks:.2f}%)")
    print(f"ticks with net delta inside budget: {in_delta} / {total_ticks} ({100*in_delta/total_ticks:.2f}%)")
    print(f"ticks with all three inside budget: {in_all} / {total_ticks} ({100*in_all/total_ticks:.2f}%)")
    print(f"max |net vega| observed: {max_vega:.2f}")
    print()
    return gross_edge, in_vega / total_ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=500)
    args = parser.parse_args()

    cfg = sim.SimConfig()
    print(f"config: ticks={cfg.ticks} requote_interval={cfg.requote_interval} "
          f"informed_frac={cfg.informed_frac} sigma={cfg.sigma} max_size={cfg.max_size} "
          f"edge_per_lot={cfg.edge_per_lot}")
    print(f"budgets: vega={cfg.budgets.vega} gamma={cfg.budgets.gamma} delta={cfg.budgets.delta}")
    print(f"sessions: {args.sessions}  base_seed: {cfg.base_seed}")
    print()

    t0 = time.time()
    baseline = sim.run_benchmark(cfg, args.sessions, "contract_by_contract")
    t1 = time.time()
    optimized = sim.run_benchmark(cfg, args.sessions, "book_optimized")
    t2 = time.time()

    ge_base, invega_base = summarize("contract_by_contract (baseline)", baseline, t1 - t0)
    ge_opt, invega_opt = summarize("book_optimized (convex program)", optimized, t2 - t1)

    cost_pct = 100.0 * (ge_base - ge_opt) / ge_base
    print("--- comparison ---")
    print(f"net vega held inside budget: {100*invega_opt:.2f}% (optimized) vs {100*invega_base:.2f}% (contract by contract)")
    print(f"cost of the optimized policy in gross edge: {cost_pct:.2f}%")
    print(f"total elapsed: {t2 - t0:.2f}s")


if __name__ == "__main__":
    main()
