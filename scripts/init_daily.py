"""
init_daily.py
One-time script to seed daily_ohlc table in DuckDB.
Downloads daily OHLCV only for stocks present in breakout_monthly table,
for the last 2 years only.
Run AFTER init_monthly.py and mark_breakouts.py have completed.
"""

import os
import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta

DB_PATH = "data/market.db"
BATCH_SIZE = 50

def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_ohlc (
            symbol  VARCHAR,
            date    DATE,
            open    DOUBLE,
            high    DOUBLE,
            low     DOUBLE,
            close   DOUBLE,
            volume  BIGINT,
            PRIMARY KEY (symbol, date)
        )
    """)
    print("Table daily_ohlc ready.")

def main():
    print(f"=== init_daily.py started at {datetime.now()} ===")

    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_PATH)
    init_db(con)

    # Only download stocks that have a breakout in last 2 years
    cutoff = date.today() - timedelta(days=730)
    start_date = cutoff.strftime("%Y-%m-%d")

    symbols_df = con.execute("""
        SELECT DISTINCT symbol FROM breakout_monthly
        WHERE breakout_month >= ?
    """, [cutoff]).fetchdf()

    symbols = symbols_df["symbol"].tolist()
    print(f"Breakout stocks to download daily data for: {len(symbols)}")
    print(f"Date range: {start_date} to today")

    if not symbols:
        print("No breakout stocks found. Run mark_breakouts.py first.")
        con.close()
        return

    inserted_total = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_ns = [s + ".NS" for s in batch]
        print(f"\nBatch {i//BATCH_SIZE + 1}/{(len(symbols)-1)//BATCH_SIZE + 1}: {len(batch)} stocks...")

        try:
            raw = yf.download(
                tickers=batch_ns,
                start=start_date,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"  Batch download failed: {e}")
            time.sleep(5)
            continue

        rows = []
        for sym, sym_ns in zip(batch, batch_ns):
            try:
                if len(batch_ns) == 1:
                    df = raw.copy()
                else:
                    df = raw[sym_ns].copy()

                df = df.dropna(subset=["Close"])
                if df.empty:
                    print(f"  No data: {sym}")
                    continue

                df.index = pd.to_datetime(df.index)

                for dt, row in df.iterrows():
                    rows.append({
                        "symbol": sym,
                        "date": dt.date(),
                        "open": round(float(row["Open"]), 4),
                        "high": round(float(row["High"]), 4),
                        "low": round(float(row["Low"]), 4),
                        "close": round(float(row["Close"]), 4),
                        "volume": int(row["Volume"])
                    })

            except Exception as e:
                print(f"  Error processing {sym}: {e}")
                continue

        if rows:
            df_insert = pd.DataFrame(rows)
            con.execute("""
                INSERT OR IGNORE INTO daily_ohlc
                SELECT symbol, date, open, high, low, close, volume
                FROM df_insert
            """)
            inserted_total += len(rows)
            print(f"  Inserted {len(rows)} rows.")

        time.sleep(2)

    count = con.execute("SELECT COUNT(*) FROM daily_ohlc").fetchone()[0]
    print(f"\n=== Done. Total rows in daily_ohlc: {count} ({inserted_total} inserted this run) ===")
    con.close()

if __name__ == "__main__":
    main()
