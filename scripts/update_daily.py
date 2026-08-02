"""
update_daily.py
Runs once daily, after market close, via GitHub Actions.
Downloads OHLCV for the FULL stock universe (data/stocks.csv, same
source update_monthly.py already uses) and appends to daily_ohlc. Also
updates index_daily_ohlc.

BUG FIX (2026-07-26): this used to source its symbol list from
breakout_monthly WHERE status='active', which meant any symbol whose
breakout got invalidated (price closed below breakout_day_low) silently
and PERMANENTLY dropped out of daily updates from that day forward --
found via a systemic check that 621 of 762 tracked symbols (81.5%) were
stale by more than 7 days, all of them with status='invalidated' on
their latest breakout. Beyond just staleness, this could make a genuinely
fresh, later breakout invisible forever: enrich_breakouts.py needs
daily_ohlc coverage for the new breakout month to fill in breakout_date,
and a frozen daily_ohlc history means that search finds nothing. Fixed
by tracking the same full universe update_monthly.py already covers,
with coverage/backfill now judged by whether each symbol's daily_ohlc is
actually fresh, not by its breakout status.

For any symbol whose daily_ohlc has no rows, or whose latest row is more
than BACKFILL_STALENESS_DAYS old, backfills its full history from
BACKFILL_LOOKBACK_DAYS ago instead of only pulling the last few days.

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

import sys
import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta, timezone, time as dtime

DB_PATH = "data/market.db"
STOCKS_CSV = "data/stocks.csv"
INDEX_TICKER = "^CRSLDX"
INDEX_SYMBOL = "NIFTY500"
BACKFILL_STALENESS_DAYS = 10   # a symbol whose last daily_ohlc row is older than this gets a full backfill, not just a 5-day catch-up
BACKFILL_LOOKBACK_DAYS = 365   # how far back to backfill a stale/never-tracked symbol
STALE_HEALTH_THRESHOLD_PCT = 15  # if more than this % of the universe is still stale after a run, fail the job so the failure-email fires
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

def get_full_universe():
    """Same source update_monthly.py already uses -- keeps daily_ohlc and
    monthly_ohlc tracking the identical symbol set, so a symbol never
    silently falls out of one while staying in the other."""
    stocks_df = pd.read_csv(STOCKS_CSV)
    return (
        stocks_df["symbol"]
        .astype(str).str.replace(".NS", "", regex=False).str.strip()
        .tolist()
    )

def split_symbols_by_freshness(con, symbols, cutoff_date):
    """
    incremental: symbol's daily_ohlc is already fresh (last row within
      BACKFILL_STALENESS_DAYS) -- just fetch a short recent window.
    backfill: symbol has no daily_ohlc rows at all, or its last row is
      older than that -- fetch BACKFILL_LOOKBACK_DAYS of history to catch
      it up fully, not just the last few days.
    """
    last_dates = con.execute(
        "SELECT symbol, MAX(date) AS last_date FROM daily_ohlc GROUP BY symbol"
    ).fetchdf().set_index("symbol")["last_date"].to_dict()

    incremental, backfill = [], []
    for sym in symbols:
        last = last_dates.get(sym)
        if last is not None and (cutoff_date - pd.Timestamp(last).date()).days <= BACKFILL_STALENESS_DAYS:
            incremental.append(sym)
        else:
            backfill.append(sym)
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

    symbols = get_full_universe()
    print(f"Full universe to update: {len(symbols)}")

    if not symbols:
        print("No symbols in universe. Exiting.")
        con.close()
        return

    incremental_symbols, backfill_symbols = split_symbols_by_freshness(con, symbols, cutoff_date)
    print(f"  Incremental (already fresh): {len(incremental_symbols)}")
    print(f"  Needs backfill (stale/never tracked): {len(backfill_symbols)}")

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

    # Backfill: stale or never-tracked symbols all get the same lookback window.
    backfill_start = (date.today() - timedelta(days=BACKFILL_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for i in range(0, len(backfill_symbols), BATCH_SIZE):
        batch = backfill_symbols[i:i + BATCH_SIZE]
        batch_ns = [s + ".NS" for s in batch]

        print(f"  Backfilling {len(batch)} stocks from {backfill_start}...")
        raw = download_batch(batch_ns, start=backfill_start, interval="1d")
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

    # Health check: if a meaningful chunk of the universe is still stale
    # after this run, something is systemically wrong (rate limiting, a
    # bad symbol source, a reintroduced version of the active-only bug
    # this script was just fixed for) -- fail loudly instead of letting
    # staleness silently accumulate for weeks like it did before.
    last_dates = con.execute(
        "SELECT symbol, MAX(date) AS last_date FROM daily_ohlc GROUP BY symbol"
    ).fetchdf().set_index("symbol")["last_date"].to_dict()
    stale_count = sum(
        1 for sym in symbols
        if sym not in last_dates or (cutoff_date - pd.Timestamp(last_dates[sym]).date()).days > BACKFILL_STALENESS_DAYS
    )
    stale_pct = stale_count / len(symbols) * 100
    print(f"Post-run staleness check: {stale_count}/{len(symbols)} symbols ({stale_pct:.1f}%) still stale by >{BACKFILL_STALENESS_DAYS}d.")

    con.close()
    print("=== Done ===")

    if stale_pct > STALE_HEALTH_THRESHOLD_PCT:
        print(f"FAILING: {stale_pct:.1f}% of the universe is stale, above the {STALE_HEALTH_THRESHOLD_PCT}% threshold.")
        sys.exit(1)

if __name__ == "__main__":
    main()
