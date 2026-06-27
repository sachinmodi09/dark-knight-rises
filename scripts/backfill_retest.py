"""
backfill_retest.py
One-time script to scan ALL historical daily data against breakout_monthly
and populate retest_history for every day that was in retest zone.
Run once after init_daily.py and mark_breakouts.py are complete.
"""

import duckdb
import pandas as pd
from datetime import datetime, date, timedelta

DB_PATH = "data/market.db"
RETEST_ZONE_PCT = 0.03  # 3% above breakout_low
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
            retest_pct_from_breakout_low    DOUBLE,
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
    """
    Returns a dict of date -> bool (True if Nifty500 close > 50DMA on that date).
    Precomputed for all dates to avoid per-row queries.
    """
    df = con.execute("""
        SELECT date, close FROM index_daily_ohlc
        WHERE symbol = ?
        ORDER BY date
    """, [INDEX_SYMBOL]).fetchdf()

    if df.empty:
        print("  Warning: No index data found. Defaulting nifty_above_50dma to True.")
        return {}

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    df["ma50"] = df["close"].rolling(window=50, min_periods=10).mean()
    df["above_50dma"] = df["close"] > df["ma50"]
    return dict(zip(df["date"], df["above_50dma"]))

def main():
    print(f"=== backfill_retest.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)
    init_retest_table(con)

    # Load all active breakouts
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_low, breakout_close
        FROM breakout_monthly
        WHERE status = 'active'
        ORDER BY symbol, breakout_month
    """).fetchdf()

    print(f"Active breakouts to scan: {len(breakouts)}")

    # Precompute Nifty 50DMA map
    print("Computing Nifty 500 50DMA map...")
    nifty_map = get_nifty_50dma_map(con)

    total_inserted = 0

    for _, bo in breakouts.iterrows():
        sym = bo["symbol"]
        bo_month = pd.to_datetime(bo["breakout_month"]).date()
        bo_low = float(bo["breakout_low"])

        zone_low  = bo_low
        zone_high = bo_low * (1 + RETEST_ZONE_PCT)

        # Load all daily data for this stock AFTER breakout month
        daily_df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM daily_ohlc
            WHERE symbol = ?
            AND date > ?
            ORDER BY date
        """, [sym, bo_month]).fetchdf()

        if daily_df.empty:
            continue

        daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date

        rows = []
        for _, day in daily_df.iterrows():
            current_close = float(day["close"])
            current_date  = day["date"]

            # Retest condition
            if not (zone_low <= current_close <= zone_high):
                continue

            days_since = (current_date - bo_month).days
            pct_from_bo_low = round((current_close - bo_low) / bo_low * 100, 4)
            nifty_above = bool(nifty_map.get(current_date, True))

            rows.append({
                "symbol": sym,
                "breakout_month": bo_month,
                "retest_date": current_date,
                "retest_open": round(float(day["open"]), 4),
                "retest_high": round(float(day["high"]), 4),
                "retest_low": round(float(day["low"]), 4),
                "retest_close": round(current_close, 4),
                "retest_volume": int(day["volume"]),
                "days_since_breakout": days_since,
                "retest_pct_from_breakout_low": pct_from_bo_low,
                "nifty500_above_50dma": nifty_above,
                "perf_5d": None,
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
                    days_since_breakout, retest_pct_from_breakout_low, nifty500_above_50dma,
                    perf_5d, perf_10d, perf_20d, perf_30d
                FROM df_insert
            """)
            total_inserted += len(rows)
            print(f"  {sym} (breakout {bo_month}): {len(rows)} retest days inserted.")

    print(f"\nTotal retest rows inserted: {total_inserted}")

    # Now fill performance fields
    print("\nFilling performance fields...")
    fill_performance(con)

    total = con.execute("SELECT COUNT(*) FROM retest_history").fetchone()[0]
    print(f"Total rows in retest_history: {total}")
    con.close()
    print("=== Done ===")

def fill_performance(con):
    """Fill perf_5d/10d/20d/30d for all retest rows using daily_ohlc."""

    pending = con.execute("""
        SELECT symbol, retest_date, retest_close
        FROM retest_history
        WHERE perf_5d IS NULL OR perf_10d IS NULL
           OR perf_20d IS NULL OR perf_30d IS NULL
    """).fetchdf()

    print(f"  Rows needing perf fill: {len(pending)}")
    updated = 0

    for _, row in pending.iterrows():
        sym = row["symbol"]
        retest_date = pd.to_datetime(row["retest_date"]).date()
        retest_close = float(row["retest_close"])

        future_df = con.execute("""
            SELECT date, close FROM daily_ohlc
            WHERE symbol = ?
            AND date > ?
            ORDER BY date
        """, [sym, retest_date]).fetchdf()

        if future_df.empty:
            continue

        future_df["date"] = pd.to_datetime(future_df["date"]).dt.date

        def perf(n):
            if len(future_df) >= n:
                close_n = float(future_df.iloc[n - 1]["close"])
                return round((close_n - retest_close) / retest_close * 100, 4)
            return None

        p5  = perf(5)
        p10 = perf(10)
        p20 = perf(20)
        p30 = perf(30)

        updates = []
        params  = []
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

if __name__ == "__main__":
    main()
