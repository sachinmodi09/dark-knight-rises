"""
enrich_breakouts.py
Runs AFTER init_daily.py.
For each row in breakout_monthly where breakout_date is NULL,
finds the first daily date in that month where close > prev_ath,
and fills breakout_date, breakout_day_open/high/low/close.
Also updates invalidation status using breakout_day_low.
Run once during init. Also runs monthly after mark_breakouts.py.
"""

import duckdb
import pandas as pd
from datetime import datetime

DB_PATH = "data/market.db"

def enrich_breakout_dates(con):
    """Fill breakout_date and breakout_day_* for rows where it is NULL."""

    rows = con.execute("""
        SELECT symbol, breakout_month, prev_ath
        FROM breakout_monthly
        WHERE breakout_date IS NULL
    """).fetchdf()

    print(f"Rows needing breakout_date enrichment: {len(rows)}")
    enriched = 0

    for _, row in rows.iterrows():
        sym       = row["symbol"]
        bo_month  = str(row["breakout_month"])[:7]  # YYYY-MM
        prev_ath  = float(row["prev_ath"])

        # Find first daily date in breakout month where close > prev_ath
        daily = con.execute("""
            SELECT date, open, high, low, close
            FROM daily_ohlc
            WHERE symbol = ?
            AND strftime(date, '%Y-%m') = ?
            AND close > ?
            ORDER BY date ASC
            LIMIT 1
        """, [sym, bo_month, prev_ath]).fetchone()

        if daily is None:
            continue

        bo_date     = daily[0]
        bo_day_open  = round(float(daily[1]), 4)
        bo_day_high  = round(float(daily[2]), 4)
        bo_day_low   = round(float(daily[3]), 4)
        bo_day_close = round(float(daily[4]), 4)

        con.execute("""
            UPDATE breakout_monthly
            SET breakout_date    = ?,
                breakout_day_open  = ?,
                breakout_day_high  = ?,
                breakout_day_low   = ?,
                breakout_day_close = ?
            WHERE symbol = ? AND breakout_month = ?
        """, [bo_date, bo_day_open, bo_day_high, bo_day_low, bo_day_close, sym, row["breakout_month"]])
        enriched += 1

    print(f"  Enriched {enriched} rows with breakout_date and daily candle data.")

def update_invalidation(con):
    """
    Invalidate breakout if any daily close AFTER breakout_date
    goes below breakout_day_low.
    Only processes rows where breakout_date and breakout_day_low are known.
    """
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_date, breakout_day_low
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_date IS NOT NULL
        AND breakout_day_low IS NOT NULL
    """).fetchdf()

    print(f"Checking invalidation for {len(breakouts)} active breakouts...")
    invalidated = 0

    for _, row in breakouts.iterrows():
        sym        = row["symbol"]
        bo_date    = row["breakout_date"]
        bo_day_low = float(row["breakout_day_low"])
        bo_month   = row["breakout_month"]

        result = con.execute("""
            SELECT COUNT(*) FROM daily_ohlc
            WHERE symbol = ?
            AND date > ?
            AND close < ?
        """, [sym, bo_date, bo_day_low]).fetchone()[0]

        if result > 0:
            con.execute("""
                UPDATE breakout_monthly
                SET status = 'invalidated'
                WHERE symbol = ? AND breakout_month = ?
            """, [sym, bo_month])
            invalidated += 1

    print(f"  Invalidated {invalidated} breakouts.")

def main():
    print(f"=== enrich_breakouts.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)

    enrich_breakout_dates(con)
    update_invalidation(con)

    total      = con.execute("SELECT COUNT(*) FROM breakout_monthly").fetchone()[0]
    active     = con.execute("SELECT COUNT(*) FROM breakout_monthly WHERE status='active'").fetchone()[0]
    enriched   = con.execute("SELECT COUNT(*) FROM breakout_monthly WHERE breakout_date IS NOT NULL").fetchone()[0]
    print(f"breakout_monthly — total: {total}, active: {active}, with breakout_date: {enriched}")

    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
