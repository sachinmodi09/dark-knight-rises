"""
bt_build_clear_sample.py
Step 1 of the trading-vs-holding backtest.

Builds bt_clear_breakouts: every CLEAR breakout out of a long base over a
12-month window, with the entry and stop levels the strategy would have
used. Nothing is simulated here -- this is just the trade candidate list
that later steps run the 5% / 10% / stop-loss race against.

Selection matches the live alert exactly by importing compute_quality()
and split_valid_and_excluded() from scan_fresh_breakouts.py rather than
reimplementing them, so the backtest measures the system that actually
sends the emails, not a lookalike of it.

Filters:
  - breakout_month inside the window (default: last 12 COMPLETE months)
  - consolidation_months > 3   (a real base, not a one-month pop)
  - breakout_date known        (enrich_breakouts.py found the day)
  - close cleared the true prior high
  - is_clear: body >= 4%, clearance >= 20% of body, volume >= 1.5x 50d avg

Entry/stop, per the strategy:
  entry = true_prior_high  -- the all-time high level that was broken,
                              bought on a retest back down to it
  stop  = breakout day low

Note stop_above_entry: for some breakouts (typically big gap-ups) the
breakout candle's low is ABOVE the prior high, so price has to fall
through the stop to ever reach the entry. Those trades cannot be taken as
specified; they are kept in the table but flagged so later steps can
exclude them deliberately rather than silently mismodelling them.
"""

import sys
from datetime import date

import duckdb
import pandas as pd

sys.path.insert(0, "scripts")
import scan_fresh_breakouts as sfb  # noqa: E402  (live selection logic)

DB_PATH = "data/market.db"
WINDOW_START = date(2025, 9, 1)
WINDOW_END = date(2026, 9, 1)   # exclusive
MIN_CONSOLIDATION_MONTHS = 3


def build(con, start=WINDOW_START, end=WINDOW_END,
          min_consolidation=MIN_CONSOLIDATION_MONTHS):
    cand = con.execute("""
        SELECT symbol, breakout_month, breakout_date,
               breakout_day_open, breakout_day_high,
               breakout_day_low, breakout_day_close, consolidation_months
        FROM breakout_monthly
        WHERE breakout_month >= ? AND breakout_month < ?
          AND consolidation_months > ?
          AND breakout_date IS NOT NULL
        ORDER BY breakout_date
    """, [start, end, min_consolidation]).fetchdf()
    if cand.empty:
        return cand
    cand["is_repeat"] = False

    q = sfb.compute_quality(con, cand, cand["breakout_date"].max())
    valid, excluded = sfb.split_valid_and_excluded(q)
    clear = valid[valid["is_clear"]].copy()
    print(f"candidates (consolidation > {min_consolidation}m): {len(cand)}   "
          f"cleared prior high: {len(valid)} (dropped {excluded})   "
          f"CLEAR: {len(clear)}")

    clear = clear.rename(columns={
        "breakout_day_open": "bo_open", "breakout_day_high": "bo_high",
        "breakout_day_low": "bo_low", "breakout_day_close": "bo_close",
        "breakout_day_volume": "bo_volume",
    })
    clear["entry_price"] = clear["true_prior_high"]
    clear["stop_price"] = clear["bo_low"]
    clear["risk_pct"] = (clear["entry_price"] - clear["stop_price"]) / clear["entry_price"] * 100
    clear["stop_above_entry"] = clear["stop_price"] >= clear["entry_price"]
    clear["prior_high_date"] = pd.to_datetime(clear["true_prior_high_date"]).dt.date
    return clear[[
        "symbol", "breakout_month", "breakout_date", "consolidation_months",
        "bo_open", "bo_high", "bo_low", "bo_close", "bo_volume",
        "entry_price", "prior_high_date", "stop_price", "risk_pct",
        "stop_above_entry", "body_pct", "clearance_pct", "vol_ratio",
    ]]


def main():
    con = duckdb.connect(DB_PATH)
    df = build(con)
    if df.empty:
        print("No CLEAR breakouts in window.")
        return

    con.execute("DROP TABLE IF EXISTS bt_clear_breakouts")
    con.execute("CREATE TABLE bt_clear_breakouts AS SELECT * FROM df")
    n = con.execute("SELECT COUNT(*) FROM bt_clear_breakouts").fetchone()[0]

    print(f"\nbt_clear_breakouts: {n} rows "
          f"({WINDOW_START} to {WINDOW_END}, exclusive)")
    print("\nby month:")
    print(con.execute("""
        SELECT strftime(breakout_month,'%Y-%m') AS month, COUNT(*) AS clear_breakouts,
               ROUND(AVG(risk_pct),2) AS avg_risk_pct,
               SUM(CASE WHEN stop_above_entry THEN 1 ELSE 0 END) AS untradeable
        FROM bt_clear_breakouts GROUP BY month ORDER BY month
    """).fetchdf().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
