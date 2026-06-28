"""
download_stocks.py
Downloads full NSE stock universe, enriches with market cap and sector data,
filters by market cap range, and saves to data/stocks.csv.

Configure MIN_MCAP_CR and MAX_MCAP_CR below before running.
Market cap is in INR Crores.

Usage:
  python scripts/download_stocks.py
"""

import pandas as pd
import yfinance as yf
import time
import os
from datetime import datetime

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MIN_MCAP_CR = 5000      # minimum market cap in crores (e.g. 5000 = 50B INR)
MAX_MCAP_CR = 1000000   # maximum market cap in crores (set high to include all)
BATCH_SIZE  = 50        # stocks per yfinance batch call
SLEEP_SEC   = 2         # sleep between batches to avoid rate limits
OUTPUT_PATH = "data/stocks.csv"
# ──────────────────────────────────────────────────────────────────────────────

def download_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    print(f"Downloading NSE symbol list from NSE...")
    df = pd.read_csv(url)
    symbols = df["SYMBOL"].astype(str).str.strip().tolist()
    print(f"  Total NSE symbols: {len(symbols)}")
    return symbols

def fetch_info_batch(symbols_ns):
    """Fetch market cap and sector for a batch of symbols."""
    rows = []
    try:
        tickers_obj = yf.Tickers(" ".join(symbols_ns))
        for sym_ns in symbols_ns:
            sym = sym_ns.replace(".NS", "")
            try:
                info = tickers_obj.tickers[sym_ns].info
                mcap_inr   = info.get("marketCap", None)
                mcap_cr    = round(mcap_inr / 1e7, 2) if mcap_inr else None
                rows.append({
                    "symbol":       sym,
                    "company_name": info.get("longName", info.get("shortName", sym)),
                    "market_cap":   mcap_cr,
                    "sector":       info.get("sector", None),
                    "industry":     info.get("industry", None),
                    "last_price":   info.get("currentPrice", info.get("regularMarketPrice", None)),
                    "pe_ratio":     info.get("trailingPE", None),
                    "pb_ratio":     info.get("priceToBook", None),
                    "52w_high":     info.get("fiftyTwoWeekHigh", None),
                    "52w_low":      info.get("fiftyTwoWeekLow", None),
                    "currency":     info.get("currency", "INR"),
                })
            except Exception:
                rows.append({
                    "symbol": sym, "company_name": sym,
                    "market_cap": None, "sector": None, "industry": None,
                    "last_price": None, "pe_ratio": None, "pb_ratio": None,
                    "52w_high": None, "52w_low": None, "currency": "INR"
                })
    except Exception as e:
        print(f"  Batch fetch error: {e}")
    return rows

def main():
    print(f"=== download_stocks.py started at {datetime.now()} ===")
    print(f"Filter: Market Cap between {MIN_MCAP_CR:,} Cr and {MAX_MCAP_CR:,} Cr")

    os.makedirs("data", exist_ok=True)

    # Step 1: Get all NSE symbols
    symbols = download_nse_symbols()
    symbols_ns = [s + ".NS" for s in symbols]

    # Step 2: Fetch info in batches
    all_rows = []
    total_batches = (len(symbols_ns) - 1) // BATCH_SIZE + 1

    for i in range(0, len(symbols_ns), BATCH_SIZE):
        batch = symbols_ns[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Batch {batch_num}/{total_batches}: fetching {len(batch)} stocks...")

        rows = fetch_info_batch(batch)
        all_rows.extend(rows)

        # Show running filter count
        valid = [r for r in all_rows if r["market_cap"] is not None
                 and MIN_MCAP_CR <= r["market_cap"] <= MAX_MCAP_CR]
        print(f"  Running total passing filter: {len(valid)}")

        time.sleep(SLEEP_SEC)

    # Step 3: Build dataframe
    df = pd.DataFrame(all_rows)
    print(f"\nTotal symbols fetched: {len(df)}")
    print(f"Symbols with market cap data: {df['market_cap'].notna().sum()}")

    # Step 4: Filter by market cap
    df_filtered = df[
        df["market_cap"].notna() &
        (df["market_cap"] >= MIN_MCAP_CR) &
        (df["market_cap"] <= MAX_MCAP_CR)
    ].copy()

    # Step 5: Sort by market cap descending
    df_filtered = df_filtered.sort_values("market_cap", ascending=False).reset_index(drop=True)

    # Step 6: Save
    df_filtered.to_csv(OUTPUT_PATH, index=False)

    print(f"\n=== Done ===")
    print(f"Stocks passing filter: {len(df_filtered)}")
    print(f"Market cap range: {df_filtered['market_cap'].min():,.0f} Cr — {df_filtered['market_cap'].max():,.0f} Cr")
    print(f"Sectors: {df_filtered['sector'].nunique()} unique")
    print(f"Saved to: {OUTPUT_PATH}")

    # Preview
    print(f"\nTop 10 by market cap:")
    print(df_filtered[["symbol","company_name","market_cap","sector"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
