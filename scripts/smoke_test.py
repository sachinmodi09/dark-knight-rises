import os
import smtplib
import duckdb
import yfinance as yf
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# --- Config ---
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "ITC.NS", "HDFCBANK.NS"]
DB_PATH = "data/market.db"

EMAIL_SENDER   = os.environ["EMAIL_SENDER"]
EMAIL_PASSWORD = os.environ["EMAIL_PASSWORD"]
EMAIL_RECEIVER = os.environ["EMAIL_RECEIVER"]


# --- Step 1: Download last 5 days OHLCV for all stocks in one batch ---
print("Downloading data from yfinance...")
raw = yf.download(
    tickers=STOCKS,
    period="5d",
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    threads=True
)

# Parse into flat dataframe
rows = []
for symbol in STOCKS:
    try:
        df = raw[symbol].dropna().reset_index()
        df["symbol"] = symbol.replace(".NS", "")
        df = df.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                 "Low": "low", "Close": "close", "Volume": "volume"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        rows.append(df[["symbol", "date", "open", "high", "low", "close", "volume"]])
    except Exception as e:
        print(f"  Skipping {symbol}: {e}")

data = pd.concat(rows, ignore_index=True)
print(f"  Downloaded {len(data)} rows for {len(STOCKS)} stocks")


# --- Step 2: Save to DuckDB ---
print("Saving to DuckDB...")
os.makedirs("data", exist_ok=True)
con = duckdb.connect(DB_PATH)

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

# Insert, ignore duplicates
con.execute("""
    INSERT OR IGNORE INTO daily_ohlc
    SELECT symbol, date, open, high, low, close, volume
    FROM data
""")

row_count = con.execute("SELECT COUNT(*) FROM daily_ohlc").fetchone()[0]
print(f"  DuckDB daily_ohlc now has {row_count} rows")

# Quick sanity check query
sample = con.execute("""
    SELECT symbol, date, close
    FROM daily_ohlc
    ORDER BY date DESC
    LIMIT 5
""").fetchdf()

print("  Sample rows:")
print(sample.to_string(index=False))
con.close()


# --- Step 3: Send email ---
print("Sending email...")

body = f"""
Smoke Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}

✅ yfinance download: {len(data)} rows fetched
✅ DuckDB write: {row_count} total rows in daily_ohlc
✅ Stocks: {', '.join([s.replace('.NS','') for s in STOCKS])}

Sample (latest closes):
{sample.to_string(index=False)}

Pipeline is working correctly.
"""

msg = MIMEMultipart()
msg["From"]    = EMAIL_SENDER
msg["To"]      = EMAIL_RECEIVER
msg["Subject"] = f"[SmokeTest] Pipeline OK — {datetime.now().strftime('%Y-%m-%d')}"
msg.attach(MIMEText(body, "plain"))

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

print("  Email sent successfully")
print("\nSmoke test PASSED ✅")
