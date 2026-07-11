"""
update_daily.py
Runs once daily, after market close, via GitHub Actions.
Downloads OHLCV for all active breakout stocks (last 12 months)
and appends to daily_ohlc. Also updates index_daily_ohlc.

For any active symbol whose daily_ohlc coverage for its earliest active
breakout month is incomplete (e.g. a brand-new breakout that has never
been fetched before), backfills that symbol's full history from the
start of that month instead of only pulling the last few days. Without
this, enrich_breakouts.py cannot find the true breakout day for new
breakouts because the early days of the month were never downloaded.

GitHub Actions scheduled crons are not guaranteed to fire on time -- this
job has been observed running hours late, including during the NEXT day's
market session. yfinance's "daily" interval returns a live, still-forming
bar for whatever session is currently open, with close = the last traded
price at fetch time, not a settled EOD close. If that partial bar gets
written to daily_ohlc, every downstream check that trusts "close" as final
(breakout departure, invalidation, scoring) can fire on a mid-session
price that has nothing to do with where the stock actually closed -- this
produced a real false alert (CHOLAFIN, 2026-07-07) before being corrected
by the next day's run. Guard against it by never accepting a row for a
session that isn't confirmed closed yet (see safe_cutoff_date() below).
"""

import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta, timezone, time as dtime

DB_PATH = "data/market.db"
INDEX_TICKER = "^CRSLDX"
INDEX_SYMBOL = "NIFTY500"
COVERAGE_TOLERANCE_DAYS = 2  # allow this many fewer rows than the index before flagging as incomplete
IST = timezone(timedelta(hours=5, minutes=30))
MARKET_CLOSE_IST = dtime(15, 30)

def safe_cutoff_date():
    """
    Latest calendar date (IST) trustworthy as a genuinely settled close.
    "Today" only counts once the market has actually closed for the day;
    otherwise fall back to yesterday, since today's bar so far is live.
    """
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    if now_ist.time() >= MARKET_CLOSE_IST:
        return now_ist.date()
    return now_ist.date() - timedelta(days=1)

def get_index_trading_days(con, year_month):
    """Count of index trading days in a given YYYY-MM, used as ground truth for expected coverage."""
    return con.execute("""
        SELECT COUNT(*) FROM index_daily_ohlc
        WHERE symbol = ? AND strftime(date, '%Y-%m') = ?
    """, [INDEX_SYMBOL, year_month]).fetchone()[0]

def split_symbols_by_coverage(con, symbols_df):
    """
    For each active symbol, check whether daily_ohlc already has (near) full
    coverage for the month of its earliest active breakout. Returns:
      incremental: list of symbols with adequate coverage (just fetch recent days)
      backfill: list of (symbol, start_date) needing a full historical fetch
    """
    incremental = []
    backfill = []

    for _, row in symbols_df.iterrows():
        sym = row["symbol"]
        min_month = pd.to_datetime(row["min_month"]).date()
        year_month = min_month.strftime("%Y-%m")

        daily_count = con.execute("""
            SELECT COUNT(*) FROM daily_ohlc
            WHERE symbol = ? AND strftime(date, '%Y-%m') = ?
        """, [sym, year_month]).fetchone()[0]

        expected = get_index_trading_days(con, year_month)

        if expected > 0 and daily_count < expected - COVERAGE_TOLERANCE_DAYS:
            backfill.append((sym, min_month.replace(day=1)))
        else:
            incremental.append(sym)

    return incremental, backfill

def download_batch(batch_ns, **kwargs):
    try:
        return yf.download(
            tickers=batch_ns,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
            **kwargs
        )
    except Exception as e:
        print(f"  Batch download failed: {e}")
        time.sleep(3)
        return None

def rows_from_raw(raw, batch, batch_ns, cutoff):
    rows = []
    for sym, sym_ns in zip(batch, batch_ns):
        try:
            df = raw.copy() if len(batch_ns) == 1 else raw[sym_ns].copy()
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue

            df.index = pd.to_datetime(df.index)

            for dt, row in df.iterrows():
                if dt.date() > cutoff:
                    continue  # session not confirmed closed yet -- see safe_cutoff_date()
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
    return rows

def insert_rows(con, rows):
    if not rows:
        return 0
    df_insert = pd.DataFrame(rows)
    con.execute("""
        INSERT OR REPLACE INTO daily_ohlc
        SELECT symbol, date, open, high, low, close, volume
        FROM df_insert
    """)
    return len(rows)

def main():
    print(f"=== update_daily.py started at {datetime.now()} ===")

    cutoff_date = safe_cutoff_date()
    print(f"Settlement cutoff: accepting data through {cutoff_date} (IST) only.")

    con = duckdb.connect(DB_PATH)

    # Get active breakout stocks from last 12 months, along with the earliest
    # active breakout month for each (this is where coverage needs to start).
    cutoff = date.today() - timedelta(days=365)
    symbols_df = con.execute("""
        SELECT symbol, MIN(breakout_month) AS min_month
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_month >= ?
        GROUP BY symbol
    """, [cutoff]).fetchdf()

    print(f"Active breakout stocks to update: {len(symbols_df)}")

    if symbols_df.empty:
        print("No active stocks. Exiting.")
        con.close()
        return

    incremental_symbols, backfill_symbols = split_symbols_by_coverage(con, symbols_df)
    print(f"  Incremental (already covered): {len(incremental_symbols)}")
    print(f"  Needs backfill (new/incomplete month): {len(backfill_symbols)}")

    BATCH_SIZE = 100
    inserted_total = 0

    # Incremental: small lookback window, wide enough to cover a missed run.
    for i in range(0, len(incremental_symbols), BATCH_SIZE):
        batch = incremental_symbols[i:i + BATCH_SIZE]
        batch_ns = [s + ".NS" for s in batch]

        raw = download_batch(batch_ns, period="5d", interval="1d")
        if raw is None:
            continue

        rows = rows_from_raw(raw, batch, batch_ns, cutoff_date)
        inserted_total += insert_rows(con, rows)
        time.sleep(1)

    # Backfill: group symbols by their required start date and fetch full history.
    backfill_by_start = {}
    for sym, start_date in backfill_symbols:
        backfill_by_start.setdefault(start_date, []).append(sym)

    for start_date, group in backfill_by_start.items():
        start_str = start_date.strftime("%Y-%m-%d")
        for i in range(0, len(group), BATCH_SIZE):
            batch = group[i:i + BATCH_SIZE]
            batch_ns = [s + ".NS" for s in batch]

            print(f"  Backfilling {len(batch)} stocks from {start_str}...")
            raw = download_batch(batch_ns, start=start_str, interval="1d")
            if raw is None:
                continue

            rows = rows_from_raw(raw, batch, batch_ns, cutoff_date)
            inserted_total += insert_rows(con, rows)
            time.sleep(2)

    print(f"Inserted {inserted_total} rows into daily_ohlc.")

    # Update index
    print("Updating Nifty 500 index...")
    try:
        idx_raw = yf.download(
            tickers=INDEX_TICKER,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(idx_raw.columns, pd.MultiIndex):
            idx_raw.columns = idx_raw.columns.get_level_values(0)

        idx_raw = idx_raw.dropna(subset=["Close"])
        idx_raw.index = pd.to_datetime(idx_raw.index)

        idx_rows = []
        for dt, row in idx_raw.iterrows():
            if dt.date() > cutoff_date:
                continue  # session not confirmed closed yet
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
                INSERT OR REPLACE INTO index_daily_ohlc
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
