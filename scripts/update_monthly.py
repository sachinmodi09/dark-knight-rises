"""
update_monthly.py
Runs on last trading day of every month at 4 PM IST via GitHub Actions.
Downloads current month's final OHLCV for all stocks and appends to monthly_ohlc.
"""

import os
import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, date

DB_PATH = "data/market.db"
STOCKS_CSV = "data/stocks.csv"
BATCH_SIZE = 50

def main():
    print(f"=== update_monthly.py started at {datetime.now()} ===")

    stocks_df = pd.read_csv(STOCKS_CSV)
    symbols = stocks_df["symbol"].str.replace(".NS", "", regex=False).str.strip().tolist()
    print(f"Loaded {len(symbols)} stocks.")

    con = duckdb.connect(DB_PATH)

    today = date.today()
    # Download last 5 days to ensure we catch the last trading day of month
    period = "5d"

    inserted_total = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_ns = [s + ".NS" for s in batch]
        print(f"Batch {i//BATCH_SIZE + 1}: {len(batch)} stocks...")

        try:
            raw = yf.download(
                tickers=batch_ns,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"  Download failed: {e}")
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
                    continue

                df.index = pd.to_datetime(df.index)

                # Resample to get last trading day of current month
                monthly = df.resample("ME").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna(subset=["Close"])

                # Only insert rows for current month
                current_month = today.strftime("%Y-%m")
                monthly = monthly[monthly.index.strftime("%Y-%m") == current_month]

                for dt, row in monthly.iterrows():
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
                print(f"  Error {sym}: {e}")
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

    count = con.execute("SELECT COUNT(*) FROM monthly_ohlc").fetchone()[0]
    print(f"\nTotal rows in monthly_ohlc: {count}")
    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
