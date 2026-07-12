"""
scan_retest_live.py
Runs during market hours, from Google Cloud Functions (Cloud Scheduler
trigger), NOT GitHub Actions. Reads breakout_monthly (fact, read-only) and
fetches LIVE intraday bars per active/preliminary breakout symbol directly
from Yahoo Finance -- it never writes to daily_ohlc or any other table, so
it can't reintroduce the old bug where intraday runs corrupted the
official end-of-day close. This is purely a read + email side channel.

Unlike the once-daily EOD scan (scan_retest.py), this fetches each
symbol's full intraday range so far today (not just the latest tick) --
a stock that dipped into the entry zone at 10 AM and rallied away by
the time this runs would be invisible if we only looked at the current
price. The intraday low-so-far is appended as a synthetic "today" row
onto each symbol's historical daily series, so the exact same validated
_score_features/compute_candidates logic the EOD scan uses can evaluate
it -- no separate/divergent scoring path to maintain.

Market-hours guard: skips (no email, no API calls) outside NSE trading
hours on weekdays, so a stray/early/late Scheduler firing is a no-op.
"""

import time
import duckdb
import pandas as pd
import yfinance as yf
from datetime import datetime, date, time as dtime, timezone, timedelta

import retest_common as rc
from scan_retest import send_email  # reuse the same email sender

DB_PATH = rc.DB_PATH
BATCH_SIZE = 50
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    """
    Cloud Functions/Cloud Run containers run in UTC by default. Market hours
    are IST-defined (9:15-15:30) -- comparing a naive datetime.now() against
    those boundaries silently checks the wrong 5.5-hour window on any
    platform not already set to IST. Always convert explicitly.
    """
    return datetime.now(timezone.utc).astimezone(IST)

def is_market_hours(now):
    if now.weekday() >= 5:  # Sat/Sun
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE

def _fetch_batch(batch):
    """One batch download attempt. Returns {symbol: {low, close}} for
    whatever succeeded; symbols that fail (missing data, transient errors
    like yfinance's internal SQLite cache lock under concurrent threads)
    are simply absent from the result, not raised."""
    result = {}
    batch_ns = [s + ".NS" for s in batch]
    try:
        raw = yf.download(
            tickers=batch_ns, period="1d", interval="5m",
            group_by="ticker", auto_adjust=True, threads=True, progress=False
        )
    except Exception as e:
        print(f"  Batch fetch failed: {e}")
        return result

    for sym, sym_ns in zip(batch, batch_ns):
        try:
            df = raw.copy() if len(batch_ns) == 1 else raw[sym_ns].copy()
            df = df.dropna(subset=["Close"])
            if df.empty:
                continue
            result[sym] = {"low": float(df["Low"].min()), "close": float(df["Close"].iloc[-1])}
        except Exception:
            continue
    return result

def get_live_intraday(symbols):
    """
    Today's low-so-far and latest price per symbol, from live intraday
    bars -- read-only, never written to daily_ohlc. Symbols that fail on
    the first pass (e.g. a transient SQLite cache lock under yfinance's
    concurrent threaded fetch) get one retry as a follow-up pass, since
    that kind of collision usually clears on its own a moment later.
    """
    result = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        result.update(_fetch_batch(batch))
        time.sleep(0.5)

    missing = [s for s in symbols if s not in result]
    if missing:
        print(f"  {len(missing)} symbol(s) failed on first pass, retrying once: {missing}")
        time.sleep(2)
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i:i + BATCH_SIZE]
            result.update(_fetch_batch(batch))
            time.sleep(0.5)
        still_missing = [s for s in missing if s not in result]
        if still_missing:
            print(f"  Still missing after retry: {still_missing}")

    return result

def build_live_daily_lookup(daily_lookup, live_quotes, today):
    """
    Append a synthetic "today" row (from live intraday data) onto each
    symbol's historical daily series, so _score_features can evaluate
    today's live approach exactly like it would a settled EOD day.
    """
    today_ts = pd.Timestamp(today)
    for sym, quote in live_quotes.items():
        if sym not in daily_lookup:
            continue
        d = daily_lookup[sym]
        if not d.empty and d.iloc[-1]["date"] == today_ts:
            continue  # already has a settled row for today -- don't duplicate
        new_row = pd.DataFrame([{
            "date": today_ts, "high": quote["close"], "low": quote["low"], "close": quote["close"],
        }])
        combined = pd.concat([d, new_row], ignore_index=True)
        combined["ema20"] = combined["close"].ewm(span=20, adjust=False).mean()
        combined["ema50"] = combined["close"].ewm(span=50, adjust=False).mean()
        daily_lookup[sym] = combined
    return daily_lookup

def main(force=False, approach_pct_override=None):
    if approach_pct_override is not None:
        # Test-only: temporarily widen (or narrow) the trigger threshold so
        # a validation run actually surfaces candidates to look at, instead
        # of legitimately finding nothing. Each Cloud Function invocation is
        # a fresh process, so this never leaks into any other run -- it's
        # not persisted anywhere. Do not use this for real trading decisions.
        rc.APPROACH_PCT = approach_pct_override
        print(f"NOTE: APPROACH_PCT overridden to {approach_pct_override}% for this test run only "
              f"(validated production default is 1.0%).")

    now = now_ist()
    print(f"=== scan_retest_live.py started at {now} (IST){' [FORCED TEST RUN]' if force else ''} ===")

    if not force and not is_market_hours(now):
        print("Outside market hours. Skipping.")
        return

    if force and not is_market_hours(now):
        print("NOTE: forced run outside market hours -- yfinance will return the most recent "
              "completed session's data (e.g. Friday's, if run on a weekend), not genuinely live "
              "prices. Useful for testing the pipeline mechanics, not for a real trading decision.")

    today = now.date()  # derived from IST now, not the container's local date
    con = duckdb.connect(DB_PATH, read_only=True)
    nifty_above = rc.is_nifty500_above_50dma(con)
    breakouts = rc.get_active_breakouts(con)
    prelim = rc.get_preliminary_breakouts(con)
    all_bo = pd.concat([breakouts, prelim], ignore_index=True)
    daily_lookup = rc.get_daily_since_breakout(con, all_bo)
    con.close()

    print(f"Active breakouts: {len(breakouts)}, preliminary: {len(prelim)}")
    if all_bo.empty:
        print("Nothing to scan. Skipping.")
        return

    if not nifty_above:
        print("Nifty 500 below 50DMA. No live signals.")
        return

    live_quotes = get_live_intraday(all_bo["symbol"].unique().tolist())
    print(f"Live quotes fetched: {len(live_quotes)}")

    daily_lookup = build_live_daily_lookup(daily_lookup, live_quotes, today)
    price_map = {sym: {"price": q["close"], "date": today} for sym, q in live_quotes.items()}

    candidates = rc.compute_candidates(breakouts, price_map, today, daily_lookup, tier="confirmed")
    pcandidates = rc.compute_candidates(prelim, price_map, today, daily_lookup, tier="preliminary")
    print(f"Live candidates: {len(candidates)} confirmed, {len(pcandidates)} preliminary")

    if not candidates and not pcandidates:
        print("Nothing new intraday. No email sent.")
        return

    as_of_label = now.strftime("%Y-%m-%d %H:%M") + " (LIVE, intraday)"
    subject = f"[Retest LIVE] {len(candidates) + len(pcandidates)} approaching entry — {as_of_label}"
    body = rc.format_email_body(candidates, nifty_above, as_of_label, pcandidates)
    send_email(subject, body)

    print("=== Done ===")

if __name__ == "__main__":
    main()
