"""
scan_retest.py
Runs once daily after market close via GitHub Actions.
Stateless: reads breakout_monthly (fact) and today's daily_ohlc close (fact),
computes today's buy-zone candidates fresh, and emails. Nothing is written
to the database -- "is this in the buy zone" is a query against the facts,
re-evaluated every run, not a history to persist. See retest_common.py.
"""

import os
import duckdb
import pandas as pd
import smtplib
from datetime import datetime, date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import retest_common as rc

DB_PATH        = rc.DB_PATH
EMAIL_SENDER   = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

def send_email(subject, body):
    # EMAIL_RECEIVER may hold several comma-separated addresses; smtplib
    # treats a bare string as ONE recipient, so split it into a real list.
    recipients = [a.strip() for a in (EMAIL_RECEIVER or "").split(",") if a.strip()]
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not recipients:
        print("Email credentials missing.")
        print(body)
        return
    try:
        msg = MIMEMultipart()
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, recipients, msg.as_string())
        print(f"Email sent to {len(recipients)} recipient(s): {subject}")
    except Exception as e:
        print(f"Email failed: {e}")

def get_todays_prices(con, symbols):
    """Latest daily_ohlc close per symbol -- the official, once-daily price."""
    df = con.execute("""
        SELECT symbol, date, close
        FROM daily_ohlc
        WHERE symbol IN ({})
        AND date = (SELECT MAX(date) FROM daily_ohlc d2 WHERE d2.symbol = daily_ohlc.symbol)
    """.format(",".join(["?" for _ in symbols])), symbols).fetchdf()

    return {
        row["symbol"]: {"price": row["close"], "date": row["date"]}
        for _, row in df.iterrows()
    }

def main():
    print(f"=== scan_retest.py started at {datetime.now()} ===")
    today = date.today()

    con = duckdb.connect(DB_PATH)

    nifty_above = rc.is_nifty500_above_50dma(con)
    if not nifty_above:
        print("Nifty 500 below 50DMA. No signals today.")
        send_email(
            f"[Retest] No signals — Index below 50DMA ({today})",
            f"Nifty 500 is below 50DMA on {today}. No retest scanned."
        )
        con.close()
        return

    breakouts = rc.get_active_breakouts(con)
    preliminary_breakouts = rc.get_preliminary_breakouts(con)
    print(f"Active breakouts to scan: {len(breakouts)}")
    print(f"Preliminary (month not yet closed) breakouts to scan: {len(preliminary_breakouts)}")

    if breakouts.empty and preliminary_breakouts.empty:
        send_email(f"[Retest] No active breakouts ({today})", "No active breakouts found.")
        con.close()
        return

    all_symbols = list(set(breakouts["symbol"].tolist() + preliminary_breakouts["symbol"].tolist()))
    price_map = get_todays_prices(con, all_symbols)
    daily_lookup = rc.get_daily_since_breakout(con, pd.concat([breakouts, preliminary_breakouts], ignore_index=True))
    con.close()

    candidates = rc.compute_candidates(breakouts, price_map, today, daily_lookup, tier="confirmed")
    preliminary_candidates = rc.compute_candidates(preliminary_breakouts, price_map, today, daily_lookup, tier="preliminary")
    print(f"Retest candidates today: {len(candidates)} confirmed, {len(preliminary_candidates)} preliminary")

    subject = f"[Retest Alert] {len(candidates)} stocks in zone — {today}"
    if preliminary_candidates:
        subject += f" (+{len(preliminary_candidates)} preliminary)"
    body = rc.format_email_body(candidates, nifty_above, str(today), preliminary_candidates)
    send_email(subject, body)

    print("=== Done ===")

if __name__ == "__main__":
    main()
