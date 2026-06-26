"""
update_perf.py
Runs every market day at 3:45 PM IST via GitHub Actions.
For each row in retest_history where perf fields are NULL,
fills in % return at 5, 10, 20, 30 trading days after retest_date
using daily_ohlc data (whatever is available).
"""

import duckdb
import pandas as pd
from datetime import datetime

DB_PATH = "data/market.db"

def get_nth_trading_day_close(daily_df, retest_date, n):
    """
    Given daily OHLCV for a stock, returns the close n trading days
    after retest_date. Returns None if not enough data yet.
    """
    future = daily_df[daily_df["date"] > retest_date].sort_values("date")
    if len(future) >= n:
        return float(future.iloc[n - 1]["close"])
    return None

def main():
    print(f"=== update_perf.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)

    # Get all retest rows where at least one perf field is NULL
    pending = con.execute("""
        SELECT symbol, retest_date, retest_close
        FROM retest_history
        WHERE perf_5d IS NULL OR perf_10d IS NULL
           OR perf_20d IS NULL OR perf_30d IS NULL
    """).fetchdf()

    print(f"Retest rows to update: {len(pending)}")

    updated = 0
    for _, row in pending.iterrows():
        sym = row["symbol"]
        retest_date = pd.to_datetime(row["retest_date"]).date()
        retest_close = float(row["retest_close"])

        # Load all daily data for this stock after retest_date
        daily_df = con.execute("""
            SELECT date, close FROM daily_ohlc
            WHERE symbol = ?
            AND date > ?
            ORDER BY date
        """, [sym, retest_date]).fetchdf()

        if daily_df.empty:
            continue

        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

        def perf(n):
            close_n = get_nth_trading_day_close(daily_df, retest_date, n)
            if close_n is not None:
                return round((close_n - retest_close) / retest_close * 100, 4)
            return None

        p5  = perf(5)
        p10 = perf(10)
        p20 = perf(20)
        p30 = perf(30)

        # Only update fields that now have data
        updates = []
        params = []
        if p5  is not None: updates.append("perf_5d = ?");  params.append(p5)
        if p10 is not None: updates.append("perf_10d = ?"); params.append(p10)
        if p20 is not None: updates.append("perf_20d = ?"); params.append(p20)
        if p30 is not None: updates.append("perf_30d = ?"); params.append(p30)

        if updates:
            params.extend([sym, retest_date])
            con.execute(f"""
                UPDATE retest_history
                SET {', '.join(updates)}
                WHERE symbol = ? AND retest_date = ?
            """, params)
            updated += 1

    print(f"Updated perf fields for {updated} retest rows.")
    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
