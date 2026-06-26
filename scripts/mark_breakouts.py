"""
mark_breakouts.py
Reads monthly_ohlc and daily_ohlc from DuckDB.
Identifies breakout months for each stock using original logic:
  - Breakout month = close > all previous monthly highs (cummax of High)
  - prev_ath = highest High across all previous months
  - prev_ath_month = month where that highest High occurred
  - Duration = months between prev_ath_month and breakout_month
  - breakout_date = first daily date within breakout month where daily close > prev_ath
  - Adds EMA20, volume_ratio, breakout_strength
  - Status = active / invalidated based on subsequent monthly closes vs breakout_low
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
    """
    Given monthly OHLCV dataframe for one symbol, compute all breakout months.
    Returns list of dicts, one per breakout month.
    """
    df = monthly_df.copy().sort_values("date").reset_index(drop=True)
    if len(df) < 3:
        return []

    results = []

    for i in range(1, len(df)):
        prev = df.iloc[:i]  # all months before current
        curr = df.iloc[i]

        prev_ath = prev["high"].max()
        prev_ath_idx = prev["high"].idxmax()
        prev_ath_month = prev.loc[prev_ath_idx, "date"]

        # Breakout condition: current close > highest HIGH of all previous months
        if curr["close"] <= prev_ath:
            continue

        # Valid breakout found
        breakout_month_date = curr["date"]

        # Duration in months from prev_ath_month to breakout_month
        pm = pd.Period(str(prev_ath_month)[:7], freq="M")
        bm = pd.Period(str(breakout_month_date)[:7], freq="M")
        consolidation_months = (bm - pm).n

        # EMA 20 monthly — computed on closes up to and including current month
        closes_so_far = df.iloc[:i+1]["close"].values
        if len(closes_so_far) >= 20:
            ema_series = pd.Series(closes_so_far).ewm(span=20, adjust=False).mean()
            ema_20 = round(float(ema_series.iloc[-1]), 4)
        else:
            ema_series = pd.Series(closes_so_far).ewm(span=len(closes_so_far), adjust=False).mean()
            ema_20 = round(float(ema_series.iloc[-1]), 4)

        # Volume ratio = breakout volume / avg volume of previous 6 months
        prev_6m_vol = prev.tail(6)["volume"]
        if len(prev_6m_vol) > 0 and prev_6m_vol.mean() > 0:
            volume_ratio = round(float(curr["volume"]) / float(prev_6m_vol.mean()), 4)
        else:
            volume_ratio = None

        # Breakout strength = (close - prev_ath) / prev_ath * 100
        breakout_strength = round((float(curr["close"]) - float(prev_ath)) / float(prev_ath) * 100, 4)

        # Find breakout_date: first daily date in this month where daily close > prev_ath
        breakout_date = None
        if daily_df is not None and not daily_df.empty:
            month_str = str(breakout_month_date)[:7]  # YYYY-MM
            daily_month = daily_df[daily_df["date"].astype(str).str.startswith(month_str)]
            daily_month = daily_month.sort_values("date")
            crossed = daily_month[daily_month["close"] > prev_ath]
            if not crossed.empty:
                breakout_date = crossed.iloc[0]["date"]

        results.append({
            "symbol": symbol,
            "breakout_month": breakout_month_date,
            "breakout_date": breakout_date,
            "breakout_open": round(float(curr["open"]), 4),
            "breakout_high": round(float(curr["high"]), 4),
            "breakout_low": round(float(curr["low"]), 4),
            "breakout_close": round(float(curr["close"]), 4),
            "breakout_volume": int(curr["volume"]),
            "prev_ath": round(float(prev_ath), 4),
            "prev_ath_month": prev_ath_month,
            "consolidation_months": consolidation_months,
            "ema_20_monthly": ema_20,
            "volume_ratio": volume_ratio,
            "breakout_strength": breakout_strength,
            "status": "active"
        })

    return results

def update_status(con):
    """
    For each breakout in breakout_monthly, check if any subsequent monthly close
    went below breakout_low. If yes, mark status = invalidated.
    """
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_low
        FROM breakout_monthly
        WHERE status = 'active'
    """).fetchdf()

    invalidated = 0
    for _, row in breakouts.iterrows():
        sym = row["symbol"]
        bo_month = row["breakout_month"]
        bo_low = row["breakout_low"]

        # Check if any monthly close after breakout_month is below breakout_low
        result = con.execute("""
            SELECT COUNT(*) FROM monthly_ohlc
            WHERE symbol = ?
            AND date > ?
            AND close < ?
        """, [sym, bo_month, bo_low]).fetchone()[0]

        if result > 0:
            con.execute("""
                UPDATE breakout_monthly
                SET status = 'invalidated'
                WHERE symbol = ? AND breakout_month = ?
            """, [sym, bo_month])
            invalidated += 1

    print(f"  Status update: {invalidated} breakouts invalidated.")

def main():
    print(f"=== mark_breakouts.py started at {datetime.now()} ===")

    con = duckdb.connect(DB_PATH)
    init_breakout_table(con)

    # Load all symbols from monthly_ohlc
    symbols = con.execute("SELECT DISTINCT symbol FROM monthly_ohlc ORDER BY symbol").fetchdf()["symbol"].tolist()
    print(f"Processing {len(symbols)} symbols...")

    all_results = []
    skipped = 0

    for sym in symbols:
        monthly_df = con.execute("""
            SELECT date, open, high, low, close, volume
            FROM monthly_ohlc
            WHERE symbol = ?
            ORDER BY date
        """, [sym]).fetchdf()

        daily_df = con.execute("""
            SELECT date, close
            FROM daily_ohlc
            WHERE symbol = ?
            ORDER BY date
        """, [sym]).fetchdf()

        if monthly_df.empty:
            skipped += 1
            continue

        breakouts = compute_breakouts_for_symbol(sym, monthly_df, daily_df if not daily_df.empty else None)

        if breakouts:
            all_results.extend(breakouts)

    print(f"  Found {len(all_results)} breakout months across all symbols. Skipped {skipped}.")

    if all_results:
        df_insert = pd.DataFrame(all_results)
        con.execute("""
            INSERT OR IGNORE INTO breakout_monthly
            SELECT
                symbol, breakout_month, breakout_date,
                breakout_open, breakout_high, breakout_low, breakout_close, breakout_volume,
                prev_ath, prev_ath_month, consolidation_months,
                ema_20_monthly, volume_ratio, breakout_strength, status
            FROM df_insert
        """)
        print(f"  Inserted breakout rows.")

    # Update invalidation status
    print("Updating invalidation status...")
    update_status(con)

    total = con.execute("SELECT COUNT(*) FROM breakout_monthly").fetchone()[0]
    active = con.execute("SELECT COUNT(*) FROM breakout_monthly WHERE status='active'").fetchone()[0]
    print(f"  breakout_monthly total: {total}, active: {active}")

    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
