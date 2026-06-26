"""
scan_retest.py
Runs every market day at 3:45 PM IST via GitHub Actions.
For each active breakout stock (last 12 months), checks if today's price
is within 3% of breakout_low (above it, not below).
Nifty 500 must be above its 50-day MA.
Inserts retest candidates into retest_history table.
Sends email with today's retest stocks.
"""

import os
import duckdb
import pandas as pd
import smtplib
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DB_PATH = "data/market.db"
INDEX_SYMBOL = "NIFTY500"
RETEST_ZONE_PCT = 0.03  # 3% above breakout_low
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

def init_retest_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS retest_history (
            symbol                      VARCHAR,
            breakout_month              DATE,
            retest_date                 DATE,
            retest_open                 DOUBLE,
            retest_high                 DOUBLE,
            retest_low                  DOUBLE,
            retest_close                DOUBLE,
            retest_volume               BIGINT,
            days_since_breakout         INTEGER,
            retest_pct_from_breakout_low DOUBLE,
            nifty500_above_50dma        BOOLEAN,
            perf_5d                     DOUBLE,
            perf_10d                    DOUBLE,
            perf_20d                    DOUBLE,
            perf_30d                    DOUBLE,
            PRIMARY KEY (symbol, retest_date)
        )
    """)

def is_nifty500_above_50dma(con):
    """Returns True if Nifty 500 latest close is above its 50-day MA."""
    df = con.execute("""
        SELECT close FROM index_daily_ohlc
        WHERE symbol = ?
        ORDER BY date DESC
        LIMIT 50
    """, [INDEX_SYMBOL]).fetchdf()

    if len(df) < 10:
        print("  Warning: insufficient index data for 50DMA check.")
        return True  # Default to True if data insufficient

    latest_close = df["close"].iloc[0]
    ma_50 = df["close"].mean()
    above = bool(latest_close > ma_50)
    print(f"  Nifty500 close={latest_close:.2f}, 50DMA={ma_50:.2f}, above={above}")
    return above

def main():
    print(f"=== scan_retest.py started at {datetime.now()} ===")

    today = date.today()
    cutoff = today - timedelta(days=365)

    con = duckdb.connect(DB_PATH)
    init_retest_table(con)

    # Check index regime
    nifty_above_50dma = is_nifty500_above_50dma(con)
    if not nifty_above_50dma:
        print("Nifty 500 is BELOW 50DMA. No retest signals today.")
        body = f"Retest Scan {today} — Nifty 500 below 50DMA. No signals."
        send_email(f"[Retest] No signals — Index below 50DMA ({today})", body)
        con.close()
        return

    # Load active breakouts from last 12 months
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_low, breakout_close
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_month >= ?
    """, [cutoff]).fetchdf()

    print(f"Active breakouts to scan: {len(breakouts)}")

    # Load today's daily data for all these stocks
    symbols = breakouts["symbol"].tolist()
    if not symbols:
        print("No active breakouts found.")
        con.close()
        return

    today_data = con.execute("""
        SELECT symbol, date, open, high, low, close, volume
        FROM daily_ohlc
        WHERE symbol IN ({})
        AND date = (SELECT MAX(date) FROM daily_ohlc WHERE symbol = daily_ohlc.symbol)
    """.format(",".join(["?" for _ in symbols])), symbols).fetchdf()

    today_map = today_data.set_index("symbol").to_dict("index")

    retest_candidates = []

    for _, bo in breakouts.iterrows():
        sym = bo["symbol"]
        bo_month = bo["breakout_month"]
        bo_low = float(bo["breakout_low"])

        if sym not in today_map:
            continue

        td = today_map[sym]
        current_close = float(td["close"])
        current_date = td["date"]

        # Retest must be AFTER breakout month
        if pd.to_datetime(current_date) <= pd.to_datetime(bo_month):
            continue

        # Retest zone: current_close within 3% ABOVE breakout_low
        zone_low = bo_low  # must be above breakout_low
        zone_high = bo_low * (1 + RETEST_ZONE_PCT)

        if not (zone_low <= current_close <= zone_high):
            continue

        # Calculate how far from breakout_low
        pct_from_bo_low = round((current_close - bo_low) / bo_low * 100, 4)

        # Days since breakout
        days_since = (pd.to_datetime(current_date) - pd.to_datetime(bo_month)).days

        retest_candidates.append({
            "symbol": sym,
            "breakout_month": bo_month,
            "retest_date": current_date,
            "retest_open": round(float(td["open"]), 4),
            "retest_high": round(float(td["high"]), 4),
            "retest_low": round(float(td["low"]), 4),
            "retest_close": round(current_close, 4),
            "retest_volume": int(td["volume"]),
            "days_since_breakout": days_since,
            "retest_pct_from_breakout_low": pct_from_bo_low,
            "nifty500_above_50dma": nifty_above_50dma,
            "perf_5d": None,
            "perf_10d": None,
            "perf_20d": None,
            "perf_30d": None
        })

    print(f"Retest candidates today: {len(retest_candidates)}")

    if retest_candidates:
        df_insert = pd.DataFrame(retest_candidates)
        con.execute("""
            INSERT OR IGNORE INTO retest_history
            SELECT
                symbol, breakout_month, retest_date,
                retest_open, retest_high, retest_low, retest_close, retest_volume,
                days_since_breakout, retest_pct_from_breakout_low, nifty500_above_50dma,
                perf_5d, perf_10d, perf_20d, perf_30d
            FROM df_insert
        """)

    # Build email
    subject = f"[Retest Alert] {len(retest_candidates)} stocks in zone — {today}"
    body = build_email_body(retest_candidates, today, nifty_above_50dma)
    send_email(subject, body)

    con.close()
    print("=== Done ===")

def build_email_body(candidates, today, nifty_above):
    lines = [
        f"Retest Scan — {today}",
        f"Nifty 500 above 50DMA: {nifty_above}",
        f"Stocks in retest zone: {len(candidates)}",
        "",
        f"{'Symbol':<15} {'Breakout Month':<16} {'BO Low':>8} {'Price':>8} {'% from BO Low':>14} {'Days Since BO':>14}",
        "-" * 80
    ]
    for c in sorted(candidates, key=lambda x: x["retest_pct_from_breakout_low"]):
        bo_low = float(c["breakout_month"]) if isinstance(c["breakout_month"], float) else 0
        lines.append(
            f"{c['symbol']:<15} {str(c['breakout_month']):<16} "
            f"{c['retest_low']:>8.2f} {c['retest_close']:>8.2f} "
            f"{c['retest_pct_from_breakout_low']:>13.2f}% "
            f"{c['days_since_breakout']:>13}d"
        )

    if not candidates:
        lines.append("No stocks in retest zone today.")

    return "\n".join(lines)

def send_email(subject, body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email credentials missing. Skipping email.")
        print("--- Email Body ---")
        print(body)
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")

if __name__ == "__main__":
    main()
