"""
update_monthly.py

Runs once on the last trading day of every month.

Downloads the latest MONTHLY OHLCV candles from Yahoo Finance
and updates the current month's record in monthly_ohlc.
"""

import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime

DB_PATH = "data/market.db"
STOCKS_CSV = "data/stocks.csv"
BATCH_SIZE = 50
PERIOD = "3mo"


def main():
    print(f"=== update_monthly.py started at {datetime.now()} ===")

    stocks_df = pd.read_csv(STOCKS_CSV)
    symbols = (
        stocks_df["symbol"]
        .str.replace(".NS", "", regex=False)
        .str.strip()
        .tolist()
    )

    print(f"Loaded {len(symbols)} stocks.")

    con = duckdb.connect(DB_PATH)

    updated = 0

    for i in range(0, len(symbols), BATCH_SIZE):

        batch = symbols[i:i + BATCH_SIZE]
        batch_ns = [s + ".NS" for s in batch]

        print(f"\nBatch {i // BATCH_SIZE + 1}: {len(batch)} stocks")

        try:
            raw = yf.download(
                tickers=batch_ns,
                period=PERIOD,
                interval="1mo",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"Download failed: {e}")
            time.sleep(5)
            continue

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

                # Latest monthly candle
                row = df.iloc[-1]
                month_date = df.index[-1].date()

                # Replace existing record
                con.execute(
                    """
                    DELETE FROM monthly_ohlc
                    WHERE symbol = ? AND date = ?
                    """,
                    [sym, month_date]
                )

                con.execute(
                    """
                    INSERT INTO monthly_ohlc
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        sym,
                        month_date,
                        round(float(row["Open"]), 4),
                        round(float(row["High"]), 4),
                        round(float(row["Low"]), 4),
                        round(float(row["Close"]), 4),
                        int(row["Volume"])
                    ]
                )

                updated += 1

            except Exception as e:
                print(f"Error processing {sym}: {e}")

        time.sleep(2)

    total = con.execute(
        "SELECT COUNT(*) FROM monthly_ohlc"
    ).fetchone()[0]

    con.close()

    print("\n===================================")
    print(f"Monthly rows updated : {updated}")
    print(f"Total rows in table  : {total}")
    print("===================================")
    print("=== update_monthly.py completed ===")


if __name__ == "__main__":
    main()
