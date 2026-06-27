"""
backfill_retest.py
One-time script to scan ALL historical daily data against breakout_monthly
and populate retest_history for every day price was in retest zone.
Retest zone = within 3% ABOVE breakout_day_low (daily candle low of breakout_date).
Multiple retest entries per stock are valid — every touch of zone is recorded.
Invalidation is handled by mark_breakouts.py, not here.
Run once after mark_breakouts.py completes.
"""

import duckdb
import pandas as pd
from datetime import datetime, date

DB_PATH = "data/market.db"
RETEST_ZONE_PCT = 0.03
INDEX_SYMBOL = "NIFTY500"

def init_retest_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS retest_history (
            symbol                          VARCHAR,
            breakout_month                  DATE,
            retest_date                     DATE,
            retest_open                     DOUBLE,
            retest_high                     DOUBLE,
            retest_low                      DOUBLE,
            retest_close                    DOUBLE,
            retest_volume                   BIGINT,
            days_since_breakout             INTEGER,
            retest_pct_from_breakout_day_low DOUBLE,
            nifty500_above_50dma            BOOLEAN,
            perf_5d                         DOUBLE,
            perf_10d                        DOUBLE,
            perf_20d                        DOUBLE,
            perf_30d                        DOUBLE,
            PRIMARY KEY (symbol, retest_date)
        )
    """)
    print("Table retest_history ready.")

def get_nifty_50dma_map(con):
    df = con.execute("""
        SELECT date, close FROM index_daily_ohlc
        WHERE symbol = ? ORDER BY date
    """, [INDEX_SYMBOL]).fetchdf()

    if df.empty:
        print("  Warning: No index data. Defaulting nifty_above_50dma to True.")
        return {}

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["ma50"] = df["close"].rolling(window=50, min_periods=10).mean()
    df["above_50dma"] = df["close"] > df["ma50"]
    return dict(zip(df["date"], df["above_50dma"]))

def fill_performance(con):
    pending = con.execute("""
        SELECT symbol, retest_date, retest_close
        FROM retest_history
        WHERE perf_5d IS NULL OR perf_10d IS NULL
           OR perf_20d IS NULL OR perf_30d IS NULL
    """).fetchdf()

    print(f"  Rows needing perf fill: {len(pending)}")
    updated = 0

    for _, row in pending.iterrows():
        sym          = row["symbol"]
        retest_date  = pd.to_datetime(row["retest_date"]).date()
        retest_close = float(row["retest_close"])

        future_df = con.execute("""
            SELECT date, close FROM daily_ohlc
            WHERE symbol = ? AND date > ?
            ORDER BY date
        """, [sym, retest_date]).fetchdf()

        if future_df.empty:
            continue

        def perf(n):
            if len(future_df) >= n:
                return round((float(future_df.iloc[n-1]["close"]) - retest_close) / retest_close * 100, 4)
            return None

        p5, p10, p20, p30 = perf(5), perf(10), perf(20), perf(30)

        updates, params = [], []
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

    print(f"  Performance filled for {updated} rows.")

def main():
    print(f"=== backfill_retest.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)
    init_retest_table(con)

    # Load active breakouts that have a valid breakout_date and breakout_day_low
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_date, breakout_day_low
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_date IS NOT NULL
        AND breakout_day_low IS NOT NULL
        ORDER BY symbol, breakout_month
    """).fetchdf()

    print(f"Active breakouts with daily candle data: {len(breakouts)}")

    # Precompute Nifty 50DMA map
    print("Computing Nifty 500 50DMA map...")
    nifty_map = get_nifty_50dma_map(con)

    total_inserted = 0

    for _, bo in breakouts.iterrows():
        sym            = bo["symbol"]
        bo_month       = pd.to_datetime(bo["breakout_month"]).date()
        bo_date        = pd.to_datetime(bo["breakout_date"]).date()
        bo_day_low     = float(bo["breakout_day_low"])

        zone_low  = bo_day_low
        zone_high = bo_day_low * (1 + RETEST_ZONE_PCT)

        # Load daily data AFTER breakout_date
        daily_df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM daily_ohlc
            WHERE symbol = ?
            AND date > ?
            ORDER BY date
        """, [sym, bo_date]).fetchdf()

        if daily_df.empty:
            continue

        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

        rows = []
        for _, day in daily_df.iterrows():
            current_close = float(day["close"])
            current_date  = day["date"]

            # Must be above breakout_day_low and within 3% zone
            if not (zone_low <= current_close <= zone_high):
                continue

            days_since   = (current_date - bo_date).days
            pct_from_low = round((current_close - bo_day_low) / bo_day_low * 100, 4)
            nifty_above  = bool(nifty_map.get(current_date, True))

            rows.append({
                "symbol":                           sym,
                "breakout_month":                   bo_month,
                "retest_date":                      current_date,
                "retest_open":                      round(float(day["open"]), 4),
                "retest_high":                      round(float(day["high"]), 4),
                "retest_low":                       round(float(day["low"]), 4),
                "retest_close":                     round(current_close, 4),
                "retest_volume":                    int(day["volume"]),
                "days_since_breakout":              days_since,
                "retest_pct_from_breakout_day_low": pct_from_low,
                "nifty500_above_50dma":             nifty_above,
                "perf_5d":  None,
                "perf_10d": None,
                "perf_20d": None,
                "perf_30d": None
            })

        if rows:
            df_insert = pd.DataFrame(rows)
            con.execute("""
                INSERT OR IGNORE INTO retest_history
                SELECT
                    symbol, breakout_month, retest_date,
                    retest_open, retest_high, retest_low, retest_close, retest_volume,
                    days_since_breakout, retest_pct_from_breakout_day_low,
                    nifty500_above_50dma,
                    perf_5d, perf_10d, perf_20d, perf_30d
                FROM df_insert
            """)
            total_inserted += len(rows)
            print(f"  {sym} (breakout {bo_date}): {len(rows)} retest days inserted.")

    print(f"\nTotal retest rows inserted: {total_inserted}")

    print("\nFilling performance fields...")
    fill_performance(con)

    total = con.execute("SELECT COUNT(*) FROM retest_history").fetchone()[0]
    print(f"Total rows in retest_history: {total}")
    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
