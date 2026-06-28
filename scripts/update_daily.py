"""
update_daily.py
Runs every market day at 3:30 PM IST via GitHub Actions.
Downloads today's OHLCV for all active breakout stocks (last 12 months)
and appends to daily_ohlc. Also updates index_daily_ohlc.
"""

import os
import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta

DB_PATH = "data/market.db"
INDEX_TICKER = "^CRSLDX"
INDEX_SYMBOL = "NIFTY500"

def main():
    print(f"=== update_daily.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)

    # Get active breakout stocks from last 12 months
    cutoff = date.today() - timedelta(days=365)
    symbols_df = con.execute("""
        SELECT DISTINCT symbol FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_month >= ?
    """, [cutoff]).fetchdf()

    symbols = symbols_df["symbol"].tolist()
    print(f"Active breakout stocks to update: {len(symbols)}")

    if not symbols:
        print("No active stocks. Exiting.")
        con.close()
        return

    # Batch download today's data
    BATCH_SIZE = 100
    inserted_total = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        batch_ns = [s + ".NS" for s in batch]

        try:
            raw = yf.download(
                tickers=batch_ns,
                period="2d",  # 2d to ensure we get today even if slight delay
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"  Batch download failed: {e}")
            time.sleep(3)
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

                # Only today's row
                today_df = df[df.index.date == date.today()]
                if today_df.empty:
                    # Take last available row
                    today_df = df.tail(1)

                for dt, row in today_df.iterrows():
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
                INSERT OR IGNORE INTO daily_ohlc
                SELECT symbol, date, open, high, low, close, volume
                FROM df_insert
            """)
            inserted_total += len(rows)

        time.sleep(1)

    print(f"Inserted {inserted_total} rows into daily_ohlc.")

    # Update index
    print("Updating Nifty 500 index...")
    try:
        idx_raw = yf.download(
            tickers=INDEX_TICKER,
            period="2d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )
        idx_raw = idx_raw.dropna(subset=["Close"])
        idx_raw.index = pd.to_datetime(idx_raw.index)

        idx_rows = []
        for dt, row in idx_raw.iterrows():
            idx_rows.append({
                "symbol": INDEX_SYMBOL,
                "date": dt.date(),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
            })

        if idx_rows:
            df_idx = pd.DataFrame(idx_rows)
            con.execute("""
                INSERT OR REPLACE INTO daily_ohlc
                SELECT symbol, date, open, high, low, close, volume
                FROM df_idx
            """)
            print(f"  Index rows inserted: {len(idx_rows)}")

    except Exception as e:
        print(f"  Index update failed: {e}")

    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
