import yfinance as yf
import pandas as pd
import duckdb
from datetime import datetime

DATABASE_PATH = "data/market.db"
STOCKS_CSV_PATH = "data/stocks.csv"

def init_monthly_data():
    print("--- Running init_monthly.py ---")
    conn = duckdb.connect(database=DATABASE_PATH, read_only=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS monthly_ohlc (
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
    print("Table 'monthly_ohlc' ensured to exist.")

    try:
        stocks_df = pd.read_csv(STOCKS_CSV_PATH)
        symbols = stocks_df["symbol"].tolist()
        print(f"Found {len(symbols)} symbols in {STOCKS_CSV_PATH}.")
    except FileNotFoundError:
        print(f"Error: {STOCKS_CSV_PATH} not found. Please create it.")
        return

    start_date = "2016-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    # yfinance can handle multiple tickers for download
    # However, for monthly data, it's often more robust to download individually
    # to avoid issues with missing data for some tickers in a batch.
    # We will still use a session for efficiency.

    print(f"Downloading monthly OHLCV data from {start_date} to {end_date} for {len(symbols)} stocks...")
    
    all_data = []
    for symbol in symbols:
        print(f"Downloading monthly data for {symbol}...")
        try:
            # Use interval="1mo" for monthly data. multi_level_index=False for flat DataFrame.
            data = yf.download(symbol, start=start_date, end=end_date, interval="1mo", multi_level_index=False)
            if not data.empty:
                data = data.reset_index()
                # Rename columns to match the DuckDB table schema
                data.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                }, inplace=True)
                data["symbol"] = symbol
                data = data[["symbol", "date", "open", "high", "low", "close", "volume"]]
                all_data.append(data)
                print(f"Successfully downloaded monthly data for {symbol}.")
            else:
                print(f"No monthly data found for {symbol} in the specified range.")
        except Exception as e:
            print(f"Error downloading monthly data for {symbol}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Ensure 'date' column is datetime and then convert to date for DuckDB
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date

        print(f"Inserting {len(combined_df)} monthly data rows into 'monthly_ohlc'...")
        # Use DuckDB's `INSERT OR IGNORE` to handle duplicates
        conn.execute("BEGIN TRANSACTION")
        conn.execute("INSERT OR IGNORE INTO monthly_ohlc SELECT * FROM combined_df")
        conn.execute("COMMIT")
        print("Monthly OHLCV data insertion complete (duplicates ignored).")
    else:
        print("No monthly data to insert.")

    conn.close()
    print("--- init_monthly.py finished ---")

if __name__ == "__main__":
    init_monthly_data()
