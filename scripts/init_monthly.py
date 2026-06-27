"""
init_monthly.py
One-time script to seed monthly_ohlc table in DuckDB.
Downloads monthly OHLCV for all stocks in data/stocks.csv from 2016-01-01.
Run once manually before any automated workflows.
"""

import os
import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime

DB_PATH = "data/market.db"
STOCKS_CSV = "data/stocks.csv"
START_DATE = "2016-01-01"
BATCH_SIZE = 50

def get_last_trading_day_of_month(df):
    """Given a monthly OHLCV dataframe, resample to get last trading day per month."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    # Resample to month-end using last available trading day
    monthly = df.resample("ME").last()
    monthly = monthly.dropna(subset=["Close"])
    return monthly

def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS monthly_ohlc (
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
    print("Table monthly_ohlc ready.")

def download_and_insert(con, symbols):
    total = len(symbols)
    inserted_total = 0

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_ns = [s if s.endswith(".NS") else s + ".NS" for s in batch]
        print(f"\nBatch {i//BATCH_SIZE + 1}: downloading {len(batch)} stocks...")

        try:
            raw = yf.download(
                tickers=batch_ns,
                start=START_DATE,
                interval="1mo",
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
                # Use last trading day of each month
                monthly = df.resample("ME").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna(subset=["Close"])

                for date, row in monthly.iterrows():
                    rows.append({
                        "symbol": sym,
                        "date": date.date(),
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
                INSERT OR IGNORE INTO monthly_ohlc
                SELECT symbol, date, open, high, low, close, volume
                FROM df_insert
            """)
            inserted_total += len(rows)
            print(f"  Inserted {len(rows)} rows.")

        time.sleep(2)

    return inserted_total

def main():
    print(f"=== init_monthly.py started at {datetime.now()} ===")

    # Load stock list
    stocks_df = pd.read_csv(STOCKS_CSV)
    # Expect column named 'symbol' — strip .NS if present for clean storage
    symbols = stocks_df["symbol"].str.replace(".NS", "", regex=False).str.strip().tolist()
    print(f"Loaded {len(symbols)} stocks from {STOCKS_CSV}")

    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_PATH)
    init_db(con)

    total_inserted = download_and_insert(con, symbols)

    count = con.execute("SELECT COUNT(*) FROM monthly_ohlc").fetchone()[0]
    print(f"\n=== Done. Total rows in monthly_ohlc: {count} ===")
    con.close()

if __name__ == "__main__":
    main()
