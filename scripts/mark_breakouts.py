"""
mark_breakouts.py
Reads monthly_ohlc and daily_ohlc from DuckDB.
Identifies breakout months for each stock.
Breakout = monthly close > all previous monthly highs (cummax of High).
Invalidation = any daily close after breakout_date goes below breakout_day_low
               (daily candle low of the breakout_date).
Runs on last trading day of month at 4 PM IST via GitHub Actions.
"""

import duckdb
import pandas as pd
import numpy as np
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

def compute_breakouts_for_symbol(symbol, monthly_df, daily_df):
    df = monthly_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 3:
        return []

    # Exclude current incomplete month
    today = date.today()
    df = df[df["date"].apply(lambda d: pd.to_datetime(d).date()) < date(today.year, today.month, 1)]
    if len(df) < 3:
        return []

    results = []

    for i in range(1, len(df)):
        prev = df.iloc[:i]
        curr = df.iloc[i]

        prev_ath = prev["high"].max()
        prev_ath_idx = prev["high"].idxmax()
        prev_ath_month = prev.loc[prev_ath_idx, "date"]

        # Breakout = monthly close > highest HIGH of all previous months
        if curr["close"] <= prev_ath:
            continue

        breakout_month_date = curr["date"]

        # Duration
        pm = pd.Period(str(prev_ath_month)[:7], freq="M")
        bm = pd.Period(str(breakout_month_date)[:7], freq="M")
        consolidation_months = (bm - pm).n

        # EMA 20 monthly
        closes_so_far = df.iloc[:i+1]["close"].values
        span = min(20, len(closes_so_far))
        ema_series = pd.Series(closes_so_far).ewm(span=span, adjust=False).mean()
        ema_20 = round(float(ema_series.iloc[-1]), 4)

        # Volume ratio = breakout volume / avg volume of previous 6 months
        prev_6m_vol = prev.tail(6)["volume"]
        volume_ratio = round(float(curr["volume"]) / float(prev_6m_vol.mean()), 4) if len(prev_6m_vol) > 0 and prev_6m_vol.mean() > 0 else None

        # Breakout strength
        breakout_strength = round((float(curr["close"]) - float(prev_ath)) / float(prev_ath) * 100, 4)

        # Find breakout_date and daily candle from daily_ohlc
        breakout_date = None
        breakout_day_open = breakout_day_high = breakout_day_low = breakout_day_close = None

        if daily_df is not None and not daily_df.empty:
            month_str = str(breakout_month_date)[:7]
            daily_month = daily_df[daily_df["date"].astype(str).str.startswith(month_str)]
            daily_month = daily_month.sort_values("date")
            crossed = daily_month[daily_month["close"] > prev_ath]
            if not crossed.empty:
                first = crossed.iloc[0]
                breakout_date      = first["date"]
                breakout_day_open  = round(float(first["open"]), 4)
                breakout_day_high  = round(float(first["high"]), 4)
                breakout_day_low   = round(float(first["low"]), 4)
                breakout_day_close = round(float(first["close"]), 4)

        results.append({
            "symbol":               symbol,
            "breakout_month":       breakout_month_date,
            "breakout_date":        breakout_date,
            "breakout_open":        round(float(curr["open"]), 4),
            "breakout_high":        round(float(curr["high"]), 4),
            "breakout_low":         round(float(curr["low"]), 4),
            "breakout_close":       round(float(curr["close"]), 4),
            "breakout_volume":      int(curr["volume"]),
            "breakout_day_open":    breakout_day_open,
            "breakout_day_high":    breakout_day_high,
            "breakout_day_low":     breakout_day_low,
            "breakout_day_close":   breakout_day_close,
            "prev_ath":             round(float(prev_ath), 4),
            "prev_ath_month":       prev_ath_month,
            "consolidation_months": consolidation_months,
            "ema_20_monthly":       ema_20,
            "volume_ratio":         volume_ratio,
            "breakout_strength":    breakout_strength,
            "status":               "active"
        })

    return results

def update_status(con):
    """
    Invalidate breakout if any daily close AFTER breakout_date
    goes below breakout_day_low.
    Only runs for breakouts where breakout_date and breakout_day_low are known.
    """
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_date, breakout_day_low
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_date IS NOT NULL
        AND breakout_day_low IS NOT NULL
    """).fetchdf()

    invalidated = 0
    for _, row in breakouts.iterrows():
        sym           = row["symbol"]
        bo_date       = row["breakout_date"]
        bo_day_low    = float(row["breakout_day_low"])
        bo_month      = row["breakout_month"]

        # Check if any daily close after breakout_date is below breakout_day_low
        result = con.execute("""
            SELECT COUNT(*) FROM daily_ohlc
            WHERE symbol = ?
            AND date > ?
            AND close < ?
        """, [sym, bo_date, bo_day_low]).fetchone()[0]

        if result > 0:
            con.execute("""
                UPDATE breakout_monthly
                SET status = 'invalidated'
                WHERE symbol = ? AND breakout_month = ?
            """, [sym, bo_month])
            invalidated += 1

    print(f"  Invalidated: {invalidated} breakouts.")

def main():
    print(f"=== mark_breakouts.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)
    init_breakout_table(con)

    symbols = con.execute("SELECT DISTINCT symbol FROM monthly_ohlc ORDER BY symbol").fetchdf()["symbol"].tolist()
    print(f"Processing {len(symbols)} symbols...")

    all_results = []

    for sym in symbols:
        monthly_df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM monthly_ohlc WHERE symbol = ?
            ORDER BY date
        """, [sym]).fetchdf()

        daily_df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM daily_ohlc WHERE symbol = ?
            ORDER BY date
        """, [sym]).fetchdf()

        if monthly_df.empty:
            continue

        breakouts = compute_breakouts_for_symbol(
            sym, monthly_df,
            daily_df if not daily_df.empty else None
        )
        all_results.extend(breakouts)

    print(f"Found {len(all_results)} breakout months total.")

    if all_results:
        df_insert = pd.DataFrame(all_results)
        con.execute("""
            INSERT OR IGNORE INTO breakout_monthly
            SELECT
                symbol, breakout_month, breakout_date,
                breakout_open, breakout_high, breakout_low, breakout_close, breakout_volume,
                breakout_day_open, breakout_day_high, breakout_day_low, breakout_day_close,
                prev_ath, prev_ath_month, consolidation_months,
                ema_20_monthly, volume_ratio, breakout_strength, status
            FROM df_insert
        """)

    print("Updating invalidation status...")
    update_status(con)

    total  = con.execute("SELECT COUNT(*) FROM breakout_monthly").fetchone()[0]
    active = con.execute("SELECT COUNT(*) FROM breakout_monthly WHERE status='active'").fetchone()[0]
    print(f"breakout_monthly — total: {total}, active: {active}")

    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
