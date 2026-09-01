"""
scan_fresh_breakouts.py
Runs once daily (Mon-Fri), after enrich_breakouts.py, via GitHub Actions.
Reports every breakout (confirmed month-closed, or preliminary current-
month) whose breakout_date is TODAY -- not a running list, just what's
fresh today, since the pipeline runs once daily and re-derives everything
from breakout_monthly + daily_ohlc fresh each time.

Shows ALL of today's breakouts, not just the "clean" ones -- CLEAR
candles (a decisive body_pct >= CLEAR_BODY_PCT with clearance_pct >=
CLEAR_CLEARANCE_PCT above the true prior high) are sorted to the top of
their own section; everything else follows below for completeness. See
retest_common.py module docstring (old version, pre score-removal) and
this session's chat history for how these two thresholds were derived:
body_pct is that day's own open->close %, clearance_pct is how much of
that candle's real body sits above the TRUE prior high (computed from
daily_ohlc's full history, not breakout_monthly's prev_ath -- prev_ath is
monthly-high based and can understate real resistance from an intra-month
spike that never became a new monthly high; see the GVT&D/NAVINFLUOR case
from this session: a spike high mid-month that the eventual close-based
breakout didn't actually clear yet).
"""

import os
import duckdb
import pandas as pd
import smtplib
from email.mime.text import MIMEText

DB_PATH = "data/market.db"
INDEX_SYMBOL = "NIFTY500"
CLEAR_BODY_PCT = 4.0
CLEAR_CLEARANCE_PCT = 20.0
CLEAR_VOL_RATIO = 1.5
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

def send_email(subject, body):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email credentials missing.")
        print(body)
        return
    try:
        msg = MIMEText(body, "plain")
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Email failed: {e}")

def is_nifty500_above_50dma(con):
    df = con.execute("""
        SELECT close FROM index_daily_ohlc
        WHERE symbol = ? ORDER BY date DESC LIMIT 50
    """, [INDEX_SYMBOL]).fetchdf()
    if len(df) < 10:
        return True
    latest_close = float(df["close"].iloc[0])
    ma_50 = float(df["close"].mean())
    return latest_close > ma_50

def get_drift_log_today(con, as_of):
    """Monthly OHLC corrections (Yahoo -> daily-consolidated) made by
    mark_breakouts.py's true_up_monthly_from_daily() during today's run --
    see monthly_drift_log in mark_breakouts.py for the schema/rationale."""
    return con.execute("""
        SELECT symbol, month_date, yahoo_open, yahoo_high, yahoo_low, yahoo_close,
               daily_open, daily_high, daily_low, daily_close, drifted_fields, max_drift_pct
        FROM monthly_drift_log
        WHERE run_date = ?
        ORDER BY max_drift_pct DESC
    """, [as_of]).fetchdf()

def get_breakouts_on(con, as_of):
    return con.execute("""
        SELECT symbol, breakout_month, breakout_date, breakout_day_open, breakout_day_high,
               breakout_day_low, breakout_day_close, consolidation_months
        FROM breakout_monthly
        WHERE breakout_date = ?
    """, [as_of]).fetchdf()

def compute_quality(con, breakouts, as_of):
    """Adds true_prior_high (max daily HIGH over all history strictly
    before breakout_date), body_pct, clearance_pct, and a tier label
    (confirmed vs preliminary, based on whether breakout_month has closed
    as of `as_of` -- NOT as of breakout_date, see note below)."""
    if breakouts.empty:
        return breakouts

    symbols = breakouts["symbol"].unique().tolist()
    daily = con.execute(f"""
        SELECT symbol, date, high, volume FROM daily_ohlc
        WHERE symbol IN ({",".join("?" for _ in symbols)})
        ORDER BY symbol, date
    """, symbols).fetchdf()
    daily["date"] = pd.to_datetime(daily["date"])

    rows = []
    for _, bo in breakouts.iterrows():
        sym = bo["symbol"]
        bo_date = pd.Timestamp(bo["breakout_date"])
        g = daily[daily["symbol"] == sym]
        before = g[g["date"] < bo_date]
        if not before.empty:
            peak_idx = before["high"].idxmax()
            true_prior_high = float(before.loc[peak_idx, "high"])
            true_prior_high_date = before.loc[peak_idx, "date"]
            duration_months = round((bo_date - true_prior_high_date).days / 30.44, 1)
        else:
            true_prior_high, true_prior_high_date, duration_months = None, None, None

        pre50 = before.tail(50)
        avg_vol_50 = float(pre50["volume"].mean()) if len(pre50) >= 10 else None
        bo_row = g[g["date"] == bo_date]
        bo_volume = float(bo_row["volume"].iloc[0]) if not bo_row.empty else None
        vol_ratio = bo_volume / avg_vol_50 if avg_vol_50 and avg_vol_50 > 0 else None

        o, c = float(bo["breakout_day_open"]), float(bo["breakout_day_close"])
        body_pct = (c - o) / o * 100 if o else None
        if true_prior_high is not None and c != o:
            clearance_pct = (c - true_prior_high) / (c - o) * 100
        else:
            clearance_pct = None

        is_clear = (
            body_pct is not None and clearance_pct is not None and vol_ratio is not None
            and body_pct >= CLEAR_BODY_PCT and clearance_pct >= CLEAR_CLEARANCE_PCT and vol_ratio >= CLEAR_VOL_RATIO
        )
        # "preliminary" means breakout_month hasn't closed YET as of as_of --
        # NOT as of breakout_date, which is always within breakout_month by
        # construction (enrich_breakouts.py searches for breakout_date inside
        # breakout_month), so comparing against breakout_date would always be
        # equal and always read "preliminary". Compare against the real
        # evaluation date instead.
        as_of_month_start = pd.Timestamp(as_of).to_period("M").start_time
        tier = "preliminary" if pd.Timestamp(bo["breakout_month"]).to_period("M").start_time >= as_of_month_start else "confirmed"

        rows.append({
            **bo.to_dict(),
            "true_prior_high": true_prior_high,
            "true_prior_high_date": true_prior_high_date,
            "duration_months": duration_months,
            "body_pct": round(body_pct, 2) if body_pct is not None else None,
            "clearance_pct": round(clearance_pct, 1) if clearance_pct is not None else None,
            "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "is_clear": is_clear,
            "tier": tier,
        })
    return pd.DataFrame(rows)

DRIFT_NOTES_MAX = 25

def format_drift_notes(drift_df):
    if drift_df.empty:
        return []
    total = len(drift_df)
    shown = drift_df.head(DRIFT_NOTES_MAX)
    title = f"--- ADDITIONAL NOTES: Monthly data corrected from daily ({total}) ---"
    if total > DRIFT_NOTES_MAX:
        title += f"  (showing top {DRIFT_NOTES_MAX} by drift %; full log in monthly_drift_log table)"
    lines = [title, ""]
    for i, (_, r) in enumerate(shown.iterrows(), 1):
        lines.extend([
            f"{i}. {r['symbol']}  --  {str(r['month_date'])[:7]}",
            f"   Yahoo monthly : O:{r['yahoo_open']:.2f} H:{r['yahoo_high']:.2f} "
            f"L:{r['yahoo_low']:.2f} C:{r['yahoo_close']:.2f}",
            f"   Daily-derived : O:{r['daily_open']:.2f} H:{r['daily_high']:.2f} "
            f"L:{r['daily_low']:.2f} C:{r['daily_close']:.2f}",
            f"   Drifted       : {r['drifted_fields']}  (threshold 0.5%)",
            f"   -> Replaced with daily-derived OHLC.",
            "",
        ])
    return lines

def format_email_body(df, as_of_label, nifty_above, drift_df=None):
    # A negative clearance_pct means today's close still hasn't cleared a
    # more recent high than the one breakout_monthly's prev_ath checked
    # (prev_ath is monthly-high based -- see module docstring) -- e.g.
    # CHENNPETRO's 2026-07-21 breakout closed at 1258.10, still below the
    # 1279.70 high set 2026-07-16. That's not a confirmed breakout yet, so
    # it's dropped from the report entirely rather than shown as "OTHER".
    excluded_count = int((df["clearance_pct"] < 0).sum()) if not df.empty else 0
    if not df.empty:
        df = df[(df["clearance_pct"].isna()) | (df["clearance_pct"] >= 0)]

    lines = [
        f"Fresh Breakouts - {as_of_label}",
        f"Nifty 500 Above 50 DMA : {nifty_above}",
        f"Entry : prior ATH (buy on retest down to that level, close must hold above BO day low).",
        f"Stop : BO day low (close basis). No target -- hold until stop.",
        f"CLEAR = body >= {CLEAR_BODY_PCT}% AND clearance >= {CLEAR_CLEARANCE_PCT}% AND volume >= {CLEAR_VOL_RATIO}x 50d avg.",
        f"Total breakouts today : {len(df)}"
        + (f"  ({excluded_count} excluded -- hasn't cleared a more recent high yet)" if excluded_count else ""),
        "",
    ]
    if df.empty:
        lines.append("No breakouts today.")
        if drift_df is not None:
            lines.append("")
            lines.extend(format_drift_notes(drift_df))
        return "\n".join(lines)

    clear = df[df["is_clear"]].sort_values("body_pct", ascending=False)
    rest = df[~df["is_clear"]].sort_values("body_pct", ascending=False)

    def block(sub, title):
        out = [f"--- {title} ({len(sub)}) ---", ""]
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            cp = f"{r['clearance_pct']:.1f}%" if r["clearance_pct"] is not None else "n/a"
            tph = f"{r['true_prior_high']:.2f}" if r["true_prior_high"] is not None else "n/a"
            tphd = str(r["true_prior_high_date"])[:10] if r["true_prior_high_date"] is not None else "n/a"
            dur = f"{r['duration_months']:.1f} months" if r["duration_months"] is not None else "n/a"
            vr = f"{r['vol_ratio']:.2f}x" if r["vol_ratio"] is not None else "n/a"
            out.extend([
                f"{i}. {r['symbol']}  --  {r['tier'].upper()}  --  {'CLEAR' if r['is_clear'] else 'not clear'}",
                f"   Breakout Date : {r['breakout_date']}",
                f"   Breakout Day  : O:{r['breakout_day_open']:.2f} H:{r['breakout_day_high']:.2f} "
                f"L:{r['breakout_day_low']:.2f} C:{r['breakout_day_close']:.2f}",
                f"   Prior ATH : {tph}  on {tphd}   ({dur} ago)  -- ENTRY reference",
                f"   Stop (BO day low, close basis) : {r['breakout_day_low']:.2f}",
                f"   Body : {r['body_pct']:.2f}%   Clearance above prior high : {cp}   Volume : {vr} of 50d avg",
                "",
            ])
        return out

    lines.extend(block(clear, "CLEAR"))
    lines.extend(block(rest, "OTHER"))
    if drift_df is not None and not drift_df.empty:
        lines.extend(format_drift_notes(drift_df))
    return "\n".join(lines)

def main(as_of=None):
    con = duckdb.connect(DB_PATH, read_only=True)
    if as_of is None:
        as_of = con.execute("SELECT MAX(date) FROM daily_ohlc").fetchone()[0]

    nifty_above = is_nifty500_above_50dma(con)
    breakouts = get_breakouts_on(con, as_of)
    df = compute_quality(con, breakouts, as_of)
    drift_df = get_drift_log_today(con, as_of)
    con.close()

    body = format_email_body(df, str(as_of), nifty_above, drift_df)

    valid = df[(df["clearance_pct"].isna()) | (df["clearance_pct"] >= 0)] if not df.empty else df
    n_total = len(valid)
    n_clear = int(valid["is_clear"].sum()) if n_total else 0
    subject = f"[Fresh Breakouts] {n_total} today, {n_clear} CLEAR -- {as_of}"
    send_email(subject, body)

    return df, body

if __name__ == "__main__":
    print(f"=== scan_fresh_breakouts.py started ===")
    df, body = main()
    print(body)
    print("=== Done ===")
