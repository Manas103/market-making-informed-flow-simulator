"""Post-trade attribution: reconciles the per-fill P&L decomposition exactly
against the per-session oracle, then summarizes markouts by whether the fill
was informed. This is the DuckDB half of the extension; src/run_attribution.cpp
is the C++ half that produces sessions.csv and fills.csv.

Usage:
    python attribution.py sessions.csv fills.csv
"""
import sys

import duckdb


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sessions_path, fills_path = sys.argv[1], sys.argv[2]

    con = duckdb.connect()
    con.execute(f"CREATE VIEW sessions AS SELECT * FROM read_csv_auto('{sessions_path}')")
    con.execute(f"CREATE VIEW fills AS SELECT * FROM read_csv_auto('{fills_path}')")

    n_sessions = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    n_fills = con.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
    print(f"sessions: {n_sessions}  fills: {n_fills}")

    # Reconciliation: the per-fill ledger must sum, session by session, to
    # the exact same three numbers the C++ oracle already reports -- a
    # second, independent check on top of session_runner.hpp's own
    # pnl_decomposed == pnl_oracle identity, this time checking that the
    # *attribution* (who gets credit for which slice) is consistent with
    # the aggregate, not just that the aggregate is internally consistent.
    mismatches = con.execute(
        """
        WITH per_fill AS (
            SELECT
                session_id,
                SUM(spread_captured) AS spread_sum,
                SUM(adverse_selection) AS adverse_sum,
                SUM(inventory_mark_share) AS invmark_sum
            FROM fills
            GROUP BY session_id
        )
        SELECT s.session_id, s.gross_capture, p.spread_sum,
               s.adverse_cost, p.adverse_sum,
               s.inventory_mark, p.invmark_sum
        FROM sessions s
        JOIN per_fill p USING (session_id)
        WHERE s.gross_capture != p.spread_sum
           OR s.adverse_cost != p.adverse_sum
           OR s.inventory_mark != p.invmark_sum
        """
    ).fetchall()

    if mismatches:
        print(f"RECONCILIATION FAILED: {len(mismatches)} session(s) disagree")
        for row in mismatches[:5]:
            print("  ", row)
        sys.exit(1)
    print(f"reconciliation: {n_sessions} sessions, 0 mismatches "
          "(sum of per-fill spread/adverse/inventory-mark == session oracle, exactly)")

    # Markouts by whether the fill was informed. A market maker who is
    # losing to informed flow should see a *negative* mean markout on
    # informed fills (price keeps moving against the fill after it trades)
    # and something close to zero on uninformed fills (no reason for random
    # flow to predict future price moves).
    print()
    print(f"{'informed':<10}{'n':>10}{'mean markout_1':>18}{'mean markout_10':>18}{'mean markout_60':>18}")
    for informed in (0, 1):
        row = con.execute(
            """
            SELECT COUNT(*), AVG(markout_1), AVG(markout_10), AVG(markout_60)
            FROM fills WHERE informed = ?
            """,
            [informed],
        ).fetchone()
        n, m1, m10, m60 = row
        print(f"{bool(informed):<10}{n:>10}{m1:>18.4f}{m10:>18.4f}{m60:>18.4f}")

    clamped = con.execute(
        "SELECT SUM(markout_60_clamped::INT) FROM fills"
    ).fetchone()[0]
    print(f"\nfills with markout_60 clamped to session end: {clamped} / {n_fills}")


if __name__ == "__main__":
    main()
