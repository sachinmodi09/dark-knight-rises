"""
scan_retest.py
Runs every market day at 3:45 PM IST via GitHub Actions.
For each active breakout stock (last 12 months), checks if today's price
is within 3% ABOVE breakout_day_low (daily candle low of breakout_date).
Every touch of zone is a valid retest — multiple entries per stock allowed.
Nifty 500 must be above its 50-day MA.
Inserts into retest_history and sends email.
"""

import os
import duckdb
import pandas as pd
import smtplib
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DB_PATH        = "data/market.db"
INDEX_SYMBOL   = "NIFTY500"
RETEST_ZONE_PCT = 0.03
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

def is_nifty500_above_50dma(con):
    df = con.execute("""
        SELECT close FROM index_daily_ohlc
        WHERE symbol = ?
        ORDER BY date DESC LIMIT 50
    """, [INDEX_SYMBOL]).fetchdf()

    if len(df) < 10:
        print("  Warning: insufficient index data.")
        return True

    latest_close = float(df["close"].iloc[0])
    ma_50        = float(df["close"].mean())
    above        = latest_close > ma_50
    print(f"  Nifty500 close={latest_close:.2f}, 50DMA={ma_50:.2f}, above={above}")
    return above

def send_email(subject, body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email credentials missing.")
        print(body)
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_RECEIVER
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")

def main():
    print(f"=== scan_retest.py started at {datetime.now()} ===")

    today  = date.today()
    cutoff = today - timedelta(days=365)

    con = duckdb.connect(DB_PATH)

    # Ensure table exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS retest_history (
            symbol                          VARCHAR,
            breakout_month                  DATE,
            retest_date                     DATE,
            retest_open                     DOUBLE,
            retest_high                     DOUBLE,
            retest_low                      DOUBLE,
            retest_close                    DOUBLE,
            retest_volume                   BIGINT,
            days_since_breakout             INTEGER,
            retest_pct_from_breakout_day_low DOUBLE,
            nifty500_above_50dma            BOOLEAN,
            perf_5d                         DOUBLE,
            perf_10d                        DOUBLE,
            perf_20d                        DOUBLE,
            perf_30d                        DOUBLE,
            PRIMARY KEY (symbol, retest_date)
        )
    """)

    # Check index regime
    nifty_above = is_nifty500_above_50dma(con)
    if not nifty_above:
        print("Nifty 500 below 50DMA. No signals today.")
        send_email(
            f"[Retest] No signals — Index below 50DMA ({today})",
            f"Nifty 500 is below 50DMA on {today}. No retest scanned."
        )
        con.close()
        return

    # Load active breakouts last 12 months with valid breakout_date and breakout_day_low
    breakouts = con.execute("""
        SELECT symbol, breakout_month, breakout_date, breakout_day_low
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_month >= ?
        AND breakout_date IS NOT NULL
        AND breakout_day_low IS NOT NULL
    """, [cutoff]).fetchdf()

    print(f"Active breakouts to scan: {len(breakouts)}")

    if breakouts.empty:
        send_email(f"[Retest] No active breakouts ({today})", "No active breakouts found.")
        con.close()
        return

    # Load today's data for all these stocks
    symbols = breakouts["symbol"].tolist()
    today_data = con.execute("""
        SELECT symbol, date, open, high, low, close, volume
        FROM daily_ohlc
        WHERE symbol IN ({})
        AND date = (SELECT MAX(date) FROM daily_ohlc d2 WHERE d2.symbol = daily_ohlc.symbol)
    """.format(",".join(["?" for _ in symbols])), symbols).fetchdf()

    today_map = today_data.set_index("symbol").to_dict("index")

    candidates = []

    for _, bo in breakouts.iterrows():
        sym        = bo["symbol"]
        bo_month   = pd.to_datetime(bo["breakout_month"]).date()
        bo_date    = pd.to_datetime(bo["breakout_date"]).date()
        bo_day_low = float(bo["breakout_day_low"])

        zone_low  = bo_day_low
        zone_high = bo_day_low * (1 + RETEST_ZONE_PCT)

        if sym not in today_map:
            continue

        td            = today_map[sym]
        current_close = float(td["close"])
        current_date  = td["date"]

        # Retest must be after breakout_date
        if pd.to_datetime(current_date).date() <= bo_date:
            continue

        # Must be within zone (above breakout_day_low, within 3%)
        if not (current_low <= zone_high and current_close > breakout_day_low):
            continue

        days_since   = (pd.to_datetime(current_date).date() - bo_date).days
        pct_from_low = round((current_close - bo_day_low) / bo_day_low * 100, 4)

        candidates.append({
            "symbol":                           sym,
            "breakout_month":                   bo_month,
            "retest_date":                      pd.to_datetime(current_date).date(),
            "retest_open":                      round(float(td["open"]), 4),
            "retest_high":                      round(float(td["high"]), 4),
            "retest_low":                       round(float(td["low"]), 4),
            "retest_close":                     round(current_close, 4),
            "retest_volume":                    int(td["volume"]),
            "days_since_breakout":              days_since,
            "retest_pct_from_breakout_day_low": pct_from_low,
            "nifty500_above_50dma":             nifty_above,
            "perf_5d":  None,
            "perf_10d": None,
            "perf_20d": None,
            "perf_30d": None
        })

    print(f"Retest candidates today: {len(candidates)}")

    if candidates:
        df_insert = pd.DataFrame(candidates)
        con.execute("""
            INSERT OR IGNORE INTO retest_history
            SELECT
                symbol, breakout_month, retest_date,
                retest_open, retest_high, retest_low, retest_close, retest_volume,
                days_since_breakout, retest_pct_from_breakout_day_low,
                nifty500_above_50dma,
                perf_5d, perf_10d, perf_20d, perf_30d
            FROM df_insert
        """)

    # Build and send email
    lines = [
        f"Retest Scan — {today}",
        f"Nifty 500 above 50DMA: {nifty_above}",
        f"Stocks in retest zone: {len(candidates)}",
        "",
        f"{'Symbol':<15} {'BO Month':<12} {'BO Day Low':>10} {'Price':>8} {'% from Low':>11} {'Days':>6}",
        "-" * 70
    ]
    for c in sorted(candidates, key=lambda x: x["retest_pct_from_breakout_day_low"]):
        lines.append(
            f"{c['symbol']:<15} {str(c['breakout_month']):<12} "
            f"{bo_day_low:>10.2f} {c['retest_close']:>8.2f} "
            f"{c['retest_pct_from_breakout_day_low']:>10.2f}% "
            f"{c['days_since_breakout']:>5}d"
        )

    if not candidates:
        lines.append("No stocks in retest zone today.")

    body    = "\n".join(lines)
    subject = f"[Retest Alert] {len(candidates)} stocks in zone — {today}"
    send_email(subject, body)

    con.close()
    print("=== Done ===")

if __name__ == "__main__":
    main()
