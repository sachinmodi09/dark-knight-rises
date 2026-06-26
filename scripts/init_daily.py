import yfinance as yf
import pandas as pd
import duckdb
from datetime import datetime

DATABASE_PATH = "data/market.db"
STOCKS_CSV_PATH = "data/stocks.csv"

def init_daily_data():
    print("--- Running init_daily.py ---")
    conn = duckdb.connect(database=DATABASE_PATH, read_only=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_ohlc (
            symbol VARCHAR,
            date DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume BIGINT,
            PRIMARY KEY (symbol, date)
        )
    """)
    print("Table 'daily_ohlc' ensured to exist.")

    try:
        stocks_df = pd.read_csv(STOCKS_CSV_PATH)
        symbols = stocks_df["symbol"].tolist()
        print(f"Found {len(symbols)} symbols in {STOCKS_CSV_PATH}.")
    except FileNotFoundError:
        print(f"Error: {STOCKS_CSV_PATH} not found. Please create it.")
        return

    start_date = "2016-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading daily OHLCV data from {start_date} to {end_date} for {len(symbols)} stocks...")
    
    all_data = []
    # yfinance can download multiple tickers using threads=True
    try:
        data = yf.download(symbols, start=start_date, end=end_date, interval="1d", multi_level_index=False, threads=True)
        
        if not data.empty:
            # yfinance returns a DataFrame with ticker as a column when multiple tickers are downloaded
            # We need to melt this DataFrame to get the desired format (symbol, date, open, high, low, close, volume)
            data.index.name = "date"
            data = data.stack(level=0).reset_index()
            data.rename(columns={
                "level_1": "symbol",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            data = data[["symbol", "date", "open", "high", "low", "close", "volume"]]
            all_data.append(data)
            print(f"Successfully downloaded daily data for all {len(symbols)} symbols.")
        else:
            print("No daily data found for the specified range.")

    except Exception as e:
        print(f"Error downloading daily data: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date

        print(f"Inserting {len(combined_df)} daily data rows into 'daily_ohlc'...")
        # Use DuckDB's `INSERT OR IGNORE` to handle duplicates
        conn.execute("BEGIN TRANSACTION")
        conn.execute("INSERT OR IGNORE INTO daily_ohlc SELECT * FROM combined_df")
        conn.execute("COMMIT")
        print("Daily OHLCV data insertion complete (duplicates ignored).")
    else:
        print("No daily data to insert.")

    conn.close()
    print("--- init_daily.py finished ---")

if __name__ == "__main__":
    init_daily_data()
