"""
init_index.py
One-time script to seed index_daily_ohlc table in DuckDB.
Downloads daily OHLCV for Nifty 500 index from 2016-01-01.
Run once manually before any automated workflows.
"""

import os
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime

DB_PATH = "data/market.db"
START_DATE = "2016-01-01"
INDEX_TICKER = "^CRSLDX"
INDEX_SYMBOL = "NIFTY500"

def init_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS index_daily_ohlc (
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
    print("Table index_daily_ohlc ready.")

def main():
    print(f"=== init_index.py started at {datetime.now()} ===")

    os.makedirs("data", exist_ok=True)
    con = duckdb.connect(DB_PATH)
    init_db(con)

    print(f"Downloading {INDEX_TICKER} from {START_DATE}...")
    raw = yf.download(
        tickers=INDEX_TICKER,
        start=START_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.dropna(subset=["Close"])
    raw.index = pd.to_datetime(raw.index)

    rows = []
    for date, row in raw.iterrows():
        rows.append({
            "symbol": INDEX_SYMBOL,
            "date": date.date(),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
        })

    df_insert = pd.DataFrame(rows)
    con.execute("""
        INSERT OR IGNORE INTO index_daily_ohlc
        SELECT symbol, date, open, high, low, close, volume
        FROM df_insert
    """)

    count = con.execute("SELECT COUNT(*) FROM index_daily_ohlc").fetchone()[0]
    print(f"Inserted {len(rows)} rows. Total in index_daily_ohlc: {count}")
    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
