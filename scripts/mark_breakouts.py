"""
mark_breakouts.py
Reads monthly_ohlc from DuckDB only.
Identifies breakout months for each stock.
Breakout = monthly close > all previous monthly highs.
Does NOT depend on daily_ohlc.
breakout_date and breakout_day_* fields are filled later by enrich_breakouts.py.
Runs on last trading day of month at 4 PM IST via GitHub Actions.
"""

import duckdb
import pandas as pd
from datetime import datetime, date

DB_PATH = "data/market.db"

def init_breakout_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS breakout_monthly (
            symbol                  VARCHAR,
            breakout_month          DATE,
            breakout_date           DATE,
            breakout_open           DOUBLE,
            breakout_high           DOUBLE,
            breakout_low            DOUBLE,
            breakout_close          DOUBLE,
            breakout_volume         BIGINT,
            breakout_day_open       DOUBLE,
            breakout_day_high       DOUBLE,
            breakout_day_low        DOUBLE,
            breakout_day_close      DOUBLE,
            prev_ath                DOUBLE,
            prev_ath_month          DATE,
            consolidation_months    INTEGER,
            ema_20_monthly          DOUBLE,
            volume_ratio            DOUBLE,
            breakout_strength       DOUBLE,
            status                  VARCHAR DEFAULT 'active',
            PRIMARY KEY (symbol, breakout_month)
        )
    """)
    print("Table breakout_monthly ready.")

def compute_breakouts_for_symbol(symbol, monthly_df):
    df = monthly_df.copy().sort_values("date").reset_index(drop=True)

    # Drop stale-quote placeholder rows: yfinance sometimes fills months it has
    # no real data for with a flat, zero-volume repeat of the last known price.
    # Treating those as genuine history would let a fake price set a false ATH
    # ceiling (or ceiling too low) that has nothing to do with real trading.
    is_placeholder = (
        (df["volume"] == 0) &
        (df["open"] == df["high"]) &
        (df["high"] == df["low"]) &
        (df["low"] == df["close"])
    )
    df = df[~is_placeholder].reset_index(drop=True)

    if len(df) < 3:
        return []

    results = []
    last_breakout_month = None  # month of this symbol's last CONFIRMED breakout

    for i in range(1, len(df)):
        prev = df.iloc[:i]
        curr = df.iloc[i]

        prev_ath     = prev["high"].max()
        prev_ath_idx = prev["high"].idxmax()
        prev_ath_month = prev.loc[prev_ath_idx, "date"]

        # Breakout = monthly close > highest HIGH of all previous months
        if curr["close"] <= prev_ath:
            continue

        breakout_month_date = curr["date"]

        # Consolidation duration = months since this symbol's last CONFIRMED
        # breakout, not since prev_ath_month. prev_ath_month can land on a
        # month that merely poked a new intraday/monthly high without ever
        # closing above the prior level (a false breakout) -- that would
        # understate how long the stock actually spent below resistance.
        # Only fall back to prev_ath_month when there's no earlier confirmed
        # breakout to reference (this is the symbol's first one).
        reference_month = last_breakout_month if last_breakout_month is not None else prev_ath_month
        pm = pd.Period(str(reference_month)[:7], freq="M")
        bm = pd.Period(str(breakout_month_date)[:7], freq="M")
        consolidation_months = (bm - pm).n

        # EMA 20 monthly on closes up to and including current month
        closes_so_far = df.iloc[:i+1]["close"].values
        span = min(20, len(closes_so_far))
        ema_series = pd.Series(closes_so_far).ewm(span=span, adjust=False).mean()
        ema_20 = round(float(ema_series.iloc[-1]), 4)

        # Volume ratio = breakout volume / avg of previous 6 months
        prev_6m_vol = prev.tail(6)["volume"]
        volume_ratio = round(float(curr["volume"]) / float(prev_6m_vol.mean()), 4) \
            if len(prev_6m_vol) > 0 and prev_6m_vol.mean() > 0 else None

        # Breakout strength
        breakout_strength = round((float(curr["close"]) - float(prev_ath)) / float(prev_ath) * 100, 4)

        results.append({
            "symbol":               symbol,
            "breakout_month":       breakout_month_date,
            "breakout_date":        None,   # filled by enrich_breakouts.py
            "breakout_open":        round(float(curr["open"]), 4),
            "breakout_high":        round(float(curr["high"]), 4),
            "breakout_low":         round(float(curr["low"]), 4),
            "breakout_close":       round(float(curr["close"]), 4),
            "breakout_volume":      int(curr["volume"]),
            "breakout_day_open":    None,   # filled by enrich_breakouts.py
            "breakout_day_high":    None,   # filled by enrich_breakouts.py
            "breakout_day_low":     None,   # filled by enrich_breakouts.py
            "breakout_day_close":   None,   # filled by enrich_breakouts.py
            "prev_ath":             round(float(prev_ath), 4),
            "prev_ath_month":       prev_ath_month,
            "consolidation_months": consolidation_months,
            "ema_20_monthly":       ema_20,
            "volume_ratio":         volume_ratio,
            "breakout_strength":    breakout_strength,
            "status":               "active"
        })

        last_breakout_month = breakout_month_date

    return results

def init_drift_log_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS monthly_drift_log (
            symbol          VARCHAR,
            month_date      DATE,
            run_date        DATE,
            yahoo_open      DOUBLE,
            yahoo_high      DOUBLE,
            yahoo_low       DOUBLE,
            yahoo_close     DOUBLE,
            daily_open      DOUBLE,
            daily_high      DOUBLE,
            daily_low       DOUBLE,
            daily_close     DOUBLE,
            drifted_fields  VARCHAR,
            max_drift_pct   DOUBLE
        )
    """)

def sync_current_month_from_daily(con, mismatch_tol=0.005):
    """
    Only ever touches the CURRENT calendar month's monthly_ohlc row -- the
    one still being formed in daily_ohlc today. Historical months are never
    modified here; once a month is in the past its recorded breakout data
    stays stable, full stop.

    Two things happen, both scoped to this one month:
    1. yfinance's monthly-interval endpoint returns a placeholder row for a
       brand-new calendar month (labeled by its 1st) with real volume but
       NaN OHLC, since the month hasn't traded enough to aggregate yet.
       update_monthly.py's dropna(subset=["Close"]) discards that
       placeholder, so on day 1 of a new month monthly_ohlc can end up with
       NO row at all for it. Insert one, synthesized from daily_ohlc (which
       update_daily.py always keeps current) -- a single day is enough.
    2. Once the row exists (from step 1, or from a real-but-still-forming
       Yahoo candle), Yahoo's version can still drift from the true
       daily-consolidated OHLC as the month progresses. Check open, high,
       low, and close independently against the daily-derived values; if
       any drifts past `mismatch_tol`, replace the row with daily's numbers
       and log the correction (before/after) to monthly_drift_log for the
       daily email.
    """
    latest_day = con.execute("SELECT MAX(date) FROM daily_ohlc").fetchone()[0]
    if latest_day is None:
        return
    ym = latest_day.strftime("%Y-%m")
    month_date = latest_day.replace(day=1)

    daily_agg = con.execute("""
        SELECT symbol, ARG_MIN(open, date) AS d_open, MAX(high) AS d_high,
               MIN(low) AS d_low, ARG_MAX(close, date) AS d_close, SUM(volume) AS d_volume
        FROM daily_ohlc WHERE strftime(date, '%Y-%m') = ?
        GROUP BY symbol
    """, [ym]).fetchdf()

    existing = con.execute("""
        SELECT symbol, date, open, high, low, close FROM monthly_ohlc
        WHERE strftime(date, '%Y-%m') = ?
    """, [ym]).fetchdf().set_index("symbol")

    inserted = 0
    corrected = 0
    for _, row in daily_agg.iterrows():
        sym = row["symbol"]
        o, h, l, c, v = row["d_open"], row["d_high"], row["d_low"], row["d_close"], row["d_volume"]
        if c is None or c == 0:
            continue

        if sym not in existing.index:
            con.execute("""
                INSERT INTO monthly_ohlc VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [sym, month_date, round(o, 4), round(h, 4), round(l, 4), round(c, 4), int(v)])
            inserted += 1
            continue

        cur = existing.loc[sym]
        drifts = {}
        for field, yahoo_val, daily_val in [
            ("open", cur["open"], o), ("high", cur["high"], h),
            ("low", cur["low"], l), ("close", cur["close"], c),
        ]:
            if daily_val:
                pct = abs(yahoo_val - daily_val) / daily_val * 100
                if pct / 100 > mismatch_tol:
                    drifts[field] = pct

        if drifts:
            con.execute("""
                UPDATE monthly_ohlc SET open=?, high=?, low=?, close=?, volume=?
                WHERE symbol = ? AND date = ?
            """, [round(o, 4), round(h, 4), round(l, 4), round(c, 4), int(v), sym, cur["date"]])
            corrected += 1

            drifted_fields = ", ".join(f"{f}:{p:.2f}%" for f, p in drifts.items())
            con.execute("""
                INSERT INTO monthly_drift_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                sym, cur["date"], latest_day,
                round(cur["open"], 4), round(cur["high"], 4), round(cur["low"], 4), round(cur["close"], 4),
                round(o, 4), round(h, 4), round(l, 4), round(c, 4),
                drifted_fields, round(max(drifts.values()), 4),
            ])

    if inserted:
        print(f"Backfilled {inserted} monthly_ohlc rows for {ym} from daily_ohlc "
              f"(Yahoo has not published this month's monthly candle yet).")
    if corrected:
        print(f"Corrected {corrected} monthly_ohlc rows for {ym} from daily_ohlc "
              f"(open/high/low/close drift > {mismatch_tol*100:.1f}%). Historical months untouched.")

def remove_orphaned_breakouts(con):
    """
    breakout_monthly only ever gains rows (mark_breakouts.py uses INSERT OR
    IGNORE, never deletes) -- so if update_monthly.py's date label for a
    given calendar month ever changes between runs (e.g. a still-forming
    month gets a different placeholder date on different days), the OLD
    label's row is never cleaned up and sits there as a stale duplicate
    forever. Remove any breakout_monthly row whose (symbol, breakout_month)
    no longer matches an actual monthly_ohlc row.
    """
    before = con.execute("SELECT COUNT(*) FROM breakout_monthly").fetchone()[0]
    con.execute("""
        DELETE FROM breakout_monthly b
        WHERE NOT EXISTS (
            SELECT 1 FROM monthly_ohlc m
            WHERE m.symbol = b.symbol AND m.date = b.breakout_month
        )
    """)
    after = con.execute("SELECT COUNT(*) FROM breakout_monthly").fetchone()[0]
    if before != after:
        print(f"Removed {before - after} orphaned breakout_monthly rows (stale month date labels).")

def main():
    print(f"=== mark_breakouts.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)
    init_breakout_table(con)
    init_drift_log_table(con)

    sync_current_month_from_daily(con)
    remove_orphaned_breakouts(con)

    symbols = con.execute(
        "SELECT DISTINCT symbol FROM monthly_ohlc ORDER BY symbol"
    ).fetchdf()["symbol"].tolist()
    print(f"Processing {len(symbols)} symbols...")

    all_results = []
    for sym in symbols:
        monthly_df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM monthly_ohlc WHERE symbol = ?
            ORDER BY date
        """, [sym]).fetchdf()

        if monthly_df.empty:
            continue

        breakouts = compute_breakouts_for_symbol(sym, monthly_df)
        all_results.extend(breakouts)

    print(f"Found {len(all_results)} breakout months total.")

    if all_results:
        df_insert = pd.DataFrame(all_results)
        con.execute("""
            INSERT OR IGNORE INTO breakout_monthly
            SELECT
                symbol, breakout_month, breakout_date,
                breakout_open, breakout_high, breakout_low,
                breakout_close, breakout_volume,
                breakout_day_open, breakout_day_high,
                breakout_day_low, breakout_day_close,
                prev_ath, prev_ath_month, consolidation_months,
                ema_20_monthly, volume_ratio, breakout_strength, status
            FROM df_insert
        """)

    total  = con.execute("SELECT COUNT(*) FROM breakout_monthly").fetchone()[0]
    active = con.execute("SELECT COUNT(*) FROM breakout_monthly WHERE status='active'").fetchone()[0]
    
    current_month_breakouts = con.execute("""
    SELECT
    symbol,
    breakout_month,
    breakout_close,
    prev_ath,
    breakout_strength
    FROM breakout_monthly
    WHERE breakout_month = (
    SELECT MAX(breakout_month)
    FROM breakout_monthly
    )
    ORDER BY breakout_strength DESC
    """).fetchdf()

    print("\n========== CURRENT MONTH BREAKOUTS ==========")

    if current_month_breakouts.empty:
     print("No breakouts detected for the latest month.")
    else:
     print(current_month_breakouts.to_string(index=False))

    print("=============================================\n")
    
    
    print(f"breakout_monthly — total: {total}, active: {active}")
    print("Run enrich_breakouts.py next after init_daily.py completes.")

    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
