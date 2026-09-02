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
import html
import duckdb
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

DB_PATH = "data/market.db"
INDEX_SYMBOL = "NIFTY500"
CLEAR_BODY_PCT = 4.0
CLEAR_CLEARANCE_PCT = 20.0
CLEAR_VOL_RATIO = 1.5
EMAIL_SENDER = os.environ.get("EMAIL_SENDER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER")

def send_email(subject, body, html_body=None):
    """Sends multipart/alternative when an HTML body is supplied, so clients
    that can render it get the styled version and everything else falls back
    to the plain text one."""
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("Email credentials missing.")
        print(body)
        return
    try:
        if html_body:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
        else:
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
    """Monthly OHLC corrections (Yahoo -> daily-consolidated), scoped only to
    the current month, made by mark_breakouts.py's
    sync_current_month_from_daily() during today's run -- see
    monthly_drift_log in mark_breakouts.py for the schema/rationale."""
    return con.execute("""
        SELECT symbol, month_date, yahoo_open, yahoo_high, yahoo_low, yahoo_close,
               daily_open, daily_high, daily_low, daily_close, drifted_fields, max_drift_pct
        FROM monthly_drift_log
        WHERE run_date = ?
        ORDER BY max_drift_pct DESC
    """, [as_of]).fetchdf()

def get_breakouts_on(con, as_of):
    df = con.execute("""
        SELECT symbol, breakout_month, breakout_date, breakout_day_open, breakout_day_high,
               breakout_day_low, breakout_day_close, consolidation_months
        FROM breakout_monthly
        WHERE breakout_date = ?
    """, [as_of]).fetchdf()
    df["is_repeat"] = False
    return df

def get_repeat_breakouts_on(con, as_of):
    """A monthly breakout only ever gets ONE breakout_date (the first day
    the close cleared the prior ATH) -- so a stock that broke out weakly on
    day 1 and then pushed to a much stronger new high on day 20 of the same
    month was never re-reported; the second, often cleaner, move was
    silently dropped. Catch that: for any still-active breakout already
    alerted earlier this month, treat today as alert-worthy again if
    today's close sets a fresh high above the running max close since the
    original breakout_date -- i.e. the stock is pushing to new highs again,
    not just sitting above the old level. compute_quality() then runs the
    same true-prior-high/clearance/volume checks on it as any other
    breakout, so it lands in CLEAR or OTHER on its own merits."""
    df = con.execute("""
        WITH active_this_month AS (
            SELECT symbol, breakout_month, breakout_date, consolidation_months
            FROM breakout_monthly
            WHERE status = 'active' AND breakout_date IS NOT NULL
              AND breakout_date < ?
              AND strftime(breakout_month, '%Y-%m') = strftime(?, '%Y-%m')
        ),
        running_max AS (
            SELECT a.symbol, a.breakout_month, a.breakout_date, a.consolidation_months,
                   MAX(d.close) AS max_close_so_far
            FROM active_this_month a
            JOIN daily_ohlc d ON d.symbol = a.symbol AND d.date >= a.breakout_date AND d.date < ?
            GROUP BY a.symbol, a.breakout_month, a.breakout_date, a.consolidation_months
        ),
        today AS (
            SELECT symbol, open, high, low, close FROM daily_ohlc WHERE date = ?
        )
        SELECT r.symbol, r.breakout_month, ? AS breakout_date,
               t.open AS breakout_day_open, t.high AS breakout_day_high,
               t.low AS breakout_day_low, t.close AS breakout_day_close,
               r.consolidation_months
        FROM running_max r
        JOIN today t ON t.symbol = r.symbol
        WHERE t.close > r.max_close_so_far
    """, [as_of, as_of, as_of, as_of, as_of]).fetchdf()
    df["is_repeat"] = True
    return df

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
            "breakout_day_volume": bo_volume,
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

def init_alert_log_table(con):
    """Audit trail of every alert this script has ever emailed -- so 'how is
    the system performing' can be answered by querying this table directly
    instead of digging through past emails. Carries every field the email
    itself shows, so a row here fully reconstructs what was sent.
    alert_seq counts how many times a symbol has been alerted within the
    same breakout_month (1 = first alert, 2 = a later fresh-high day in the
    same month, etc.); is_repeat flags those later ones."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS alert_log (
            symbol               VARCHAR,
            alert_date           DATE,
            breakout_month       DATE,
            alert_seq            INTEGER,
            is_repeat            BOOLEAN,
            open                 DOUBLE,
            high                 DOUBLE,
            low                  DOUBLE,
            close                DOUBLE,
            volume               BIGINT,
            true_prior_high      DOUBLE,
            true_prior_high_date DATE,
            duration_months      DOUBLE,
            consolidation_months INTEGER,
            body_pct             DOUBLE,
            clearance_pct        DOUBLE,
            vol_ratio            DOUBLE,
            is_clear             BOOLEAN,
            tier                 VARCHAR,
            sent_at              TIMESTAMP,
            PRIMARY KEY (symbol, alert_date)
        )
    """)

def attach_alert_seq(con, df):
    """alert_seq = how many alerts this symbol has already had within the
    same breakout_month, +1. Counted from alert_log itself, so it keeps
    incrementing correctly across days without extra bookkeeping."""
    if df.empty:
        df["alert_seq"] = pd.Series(dtype=int)
        return df
    seqs = []
    for _, r in df.iterrows():
        prior = con.execute("""
            SELECT COUNT(*) FROM alert_log
            WHERE symbol = ? AND breakout_month = ? AND alert_date < ?
        """, [r["symbol"], r["breakout_month"], r["breakout_date"]]).fetchone()[0]
        seq = prior + 1
        # is_repeat means this stock already broke out earlier in the same
        # month, so an earlier alert definitely happened -- even if alert_log
        # has no row for it (the table was added mid-month, so alerts sent
        # before it existed were never recorded). Without this floor those
        # rows come back as seq 1 and the email drops the repeat flag
        # entirely, which is exactly what happened on the 2026-09-02 run.
        if bool(r.get("is_repeat", False)) and seq < 2:
            seq = 2
        seqs.append(seq)
    df = df.copy()
    df["alert_seq"] = seqs
    return df

def _num(v):
    return None if v is None or pd.isna(v) else float(v)

def log_alerts(con, df):
    """Record exactly the rows that made it into the email (post-exclusion)
    into alert_log -- keyed on (symbol, alert_date), so a re-run on the same
    day refreshes rather than duplicates."""
    if df.empty:
        return
    now = pd.Timestamp.now()
    for _, r in df.iterrows():
        tph_date = r["true_prior_high_date"]
        con.execute("""
            INSERT OR REPLACE INTO alert_log VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            r["symbol"], r["breakout_date"], r["breakout_month"],
            int(r["alert_seq"]), bool(r.get("is_repeat", False)),
            _num(r["breakout_day_open"]), _num(r["breakout_day_high"]),
            _num(r["breakout_day_low"]), _num(r["breakout_day_close"]),
            int(r["breakout_day_volume"]) if pd.notna(r["breakout_day_volume"]) else None,
            _num(r["true_prior_high"]),
            None if tph_date is None or pd.isna(tph_date) else pd.Timestamp(tph_date).date(),
            _num(r["duration_months"]),
            int(r["consolidation_months"]) if pd.notna(r["consolidation_months"]) else None,
            _num(r["body_pct"]), _num(r["clearance_pct"]), _num(r["vol_ratio"]),
            bool(r["is_clear"]), r["tier"], now,
        ])

def split_valid_and_excluded(df):
    """Drop anything that hasn't actually cleared the TRUE prior high --
    e.g. CHENNPETRO's 2026-07-21 breakout closed at 1258.10, still below
    the 1279.70 high set 2026-07-16. Not a confirmed breakout, so it's not
    shown as "OTHER" and not logged as sent.

    The test is on the price itself (close > true_prior_high), NOT on the
    sign of clearance_pct. clearance_pct divides by the candle body
    (close - open), so on a RED candle both numerator and denominator go
    negative and the ratio comes back spuriously POSITIVE -- sailing
    straight through a `clearance_pct >= 0` filter. Real cases from the
    2026-09-02 run: SAKAR closed 985.80 against a 999.00 prior high and
    scored +183.3%; CORDSCABLE closed 315.50 against 316.00 and scored
    +11.5%. Both were reported as breakouts despite closing BELOW the
    level they were supposed to have broken."""
    if df.empty:
        return df, 0
    cleared = df["true_prior_high"].isna() | (df["breakout_day_close"] > df["true_prior_high"])
    non_negative = df["clearance_pct"].isna() | (df["clearance_pct"] >= 0)
    keep = cleared & non_negative
    excluded_count = int((~keep).sum())
    return df[keep], excluded_count

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

def format_email_body(df, as_of_label, nifty_above, drift_df=None, excluded_count=0):
    """`df` is expected to already be the valid (post-exclusion) set from
    split_valid_and_excluded(), with `excluded_count` passed alongside it --
    so exactly what gets emailed is also exactly what gets logged to
    alert_log."""
    repeat_count = int(df["alert_seq"].gt(1).sum()) if not df.empty and "alert_seq" in df else 0

    lines = [
        f"Fresh Breakouts - {as_of_label}",
        f"Nifty 500 Above 50 DMA : {nifty_above}",
        f"Entry : prior ATH (buy on retest down to that level, close must hold above BO day low).",
        f"Stop : BO day low (close basis). No target -- hold until stop.",
        f"CLEAR = body >= {CLEAR_BODY_PCT}% AND clearance >= {CLEAR_CLEARANCE_PCT}% AND volume >= {CLEAR_VOL_RATIO}x 50d avg.",
        f"Total breakouts today : {len(df)}"
        + (f" ({repeat_count} repeat)" if repeat_count else "")
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
            seq = int(r["alert_seq"]) if "alert_seq" in r and pd.notna(r["alert_seq"]) else 1
            seq_tag = f"  --  ALERT #{seq} THIS MONTH" if seq > 1 else ""
            out.extend([
                f"{i}. {r['symbol']}  --  {r['tier'].upper()}  --  {'CLEAR' if r['is_clear'] else 'not clear'}{seq_tag}",
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

# --- HTML email ------------------------------------------------------------
# Everything is inline-styled and table-based on purpose: Gmail and most
# mobile clients strip <style> blocks and ignore external CSS, so any styling
# that isn't on the element itself simply disappears.

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif"
MONO = "'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace"
INK, MUTED, LINE = "#16181d", "#6b7280", "#dfe3e8"
GREEN, GREEN_BG = "#0b7a43", "#e6f4ec"
AMBER, AMBER_BG = "#92400e", "#fef3c7"
BLUE, BLUE_BG = "#1d4ed8", "#dbeafe"
RED, RED_BG = "#b42318", "#fee4e2"
SLATE, SLATE_BG = "#475467", "#f2f4f7"

def _f(v, dp=2, dash="n/a"):
    return dash if v is None or pd.isna(v) else f"{float(v):,.{dp}f}"

def _badge(text, fg, bg):
    return (f'<span style="display:inline-block;padding:3px 9px;border-radius:11px;'
            f'font-size:11px;font-weight:700;letter-spacing:.3px;color:{fg};'
            f'background:{bg};font-family:{FONT};">{html.escape(str(text))}</span>')

def _stat(label, value, color):
    return (f'<td align="center" style="padding:10px 6px;">'
            f'<div style="font:700 22px {MONO};color:{color};">{html.escape(str(value))}</div>'
            f'<div style="font:600 10px {FONT};color:{MUTED};letter-spacing:.6px;'
            f'text-transform:uppercase;padding-top:3px;">{html.escape(label)}</div></td>')

def _ohlc_cell(label, value):
    return (f'<td align="center" style="padding:7px 4px;border-right:1px solid {LINE};">'
            f'<div style="font:600 10px {FONT};color:{MUTED};">{label}</div>'
            f'<div style="font:600 14px {MONO};color:{INK};padding-top:2px;">{value}</div></td>')

def _kv(label, value):
    return (f'<tr>'
            f'<td style="font:400 12px {FONT};color:{MUTED};padding:3px 10px 3px 0;'
            f'white-space:nowrap;">{label}</td>'
            f'<td style="font:600 13px {MONO};color:{INK};padding:3px 0;">{value}</td></tr>')

def _stock_card(idx, r):
    is_clear = bool(r["is_clear"])
    accent = GREEN if is_clear else SLATE
    seq = int(r["alert_seq"]) if "alert_seq" in r and pd.notna(r["alert_seq"]) else 1

    badges = [_badge("CLEAR", GREEN, GREEN_BG) if is_clear else _badge("NOT CLEAR", SLATE, SLATE_BG),
              _badge(str(r["tier"]).upper(), BLUE, BLUE_BG)]
    if seq > 1:
        badges.append(_badge(f"ALERT #{seq} THIS MONTH", AMBER, AMBER_BG))

    tph, tphd = _f(r["true_prior_high"]), str(r["true_prior_high_date"])[:10]
    dur = "n/a" if pd.isna(r["duration_months"]) else f"{float(r['duration_months']):.1f} months ago"
    cp = "n/a" if pd.isna(r["clearance_pct"]) else f"{float(r['clearance_pct']):.1f}%"
    vr = "n/a" if pd.isna(r["vol_ratio"]) else f"{float(r['vol_ratio']):.2f}x"

    return f"""
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#ffffff;border:1px solid {LINE};border-left:4px solid {accent};border-radius:6px;margin-bottom:10px;">
  <tr><td style="padding:13px 15px 11px;">
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%"><tr>
      <td style="font:700 17px {FONT};color:{INK};">
        <span style="color:{MUTED};font-weight:400;font-size:13px;">{idx}.</span>
        &nbsp;{html.escape(str(r['symbol']))}
      </td>
      <td align="right">{'&nbsp;'.join(badges)}</td>
    </tr></table>
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin-top:11px;border:1px solid {LINE};border-radius:5px;background:#fafbfc;">
      <tr>
        {_ohlc_cell('OPEN', _f(r['breakout_day_open']))}
        {_ohlc_cell('HIGH', _f(r['breakout_day_high']))}
        {_ohlc_cell('LOW', _f(r['breakout_day_low']))}
        <td align="center" style="padding:7px 4px;">
          <div style="font:600 10px {FONT};color:{MUTED};">CLOSE</div>
          <div style="font:700 14px {MONO};color:{accent};padding-top:2px;">{_f(r['breakout_day_close'])}</div>
        </td>
      </tr>
    </table>
    <table role="presentation" cellpadding="0" cellspacing="0" style="margin-top:11px;">
      {_kv('Breakout date', str(r['breakout_date'])[:10])}
      {_kv('Entry &mdash; prior ATH', f'{tph} <span style="font:400 11px {FONT};color:{MUTED};">on {tphd} &middot; {dur}</span>')}
      {_kv('Stop &mdash; BO day low', f"<span style='color:{RED};'>{_f(r['breakout_day_low'])}</span>")}
      {_kv('Body / Clearance / Volume', f"{_f(r['body_pct'])}% &nbsp;&middot;&nbsp; {cp} &nbsp;&middot;&nbsp; {vr} of 50d avg")}
    </table>
  </td></tr>
</table>"""

def _section(title, sub, rows, color):
    if not len(rows):
        return ""
    cards = "".join(_stock_card(i, r) for i, (_, r) in enumerate(rows.iterrows(), 1))
    return f"""
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin:20px 0 9px;">
  <tr><td style="border-left:3px solid {color};padding-left:9px;">
    <div style="font:700 14px {FONT};color:{INK};letter-spacing:.3px;">{title}
      <span style="color:{MUTED};font-weight:400;">({len(rows)})</span></div>
    <div style="font:400 11px {FONT};color:{MUTED};padding-top:2px;">{sub}</div>
  </td></tr>
</table>{cards}"""

def _drift_html(drift_df):
    if drift_df is None or drift_df.empty:
        return ""
    total, shown = len(drift_df), drift_df.head(DRIFT_NOTES_MAX)
    note = (f" &middot; showing top {DRIFT_NOTES_MAX} by drift; full log in "
            f"<code>monthly_drift_log</code>" if total > DRIFT_NOTES_MAX else "")
    body = "".join(
        f'<tr style="background:{"#ffffff" if i % 2 else "#fafbfc"};">'
        f'<td style="padding:6px 9px;font:600 12px {FONT};color:{INK};border-top:1px solid {LINE};">'
        f'{html.escape(str(r["symbol"]))}<div style="font:400 10px {FONT};color:{MUTED};">{str(r["month_date"])[:7]}</div></td>'
        f'<td style="padding:6px 9px;font:400 11px {MONO};color:{MUTED};border-top:1px solid {LINE};">'
        f'{_f(r["yahoo_open"])} / {_f(r["yahoo_high"])} / {_f(r["yahoo_low"])} / {_f(r["yahoo_close"])}</td>'
        f'<td style="padding:6px 9px;font:400 11px {MONO};color:{INK};border-top:1px solid {LINE};">'
        f'{_f(r["daily_open"])} / {_f(r["daily_high"])} / {_f(r["daily_low"])} / {_f(r["daily_close"])}</td>'
        f'<td style="padding:6px 9px;font:600 11px {MONO};color:{AMBER};border-top:1px solid {LINE};">'
        f'{html.escape(str(r["drifted_fields"]))}</td></tr>'
        for i, (_, r) in enumerate(shown.iterrows()))
    return f"""
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="margin:22px 0 9px;">
  <tr><td style="border-left:3px solid {AMBER};padding-left:9px;">
    <div style="font:700 14px {FONT};color:{INK};">Additional notes &mdash; monthly data corrected from daily
      <span style="color:{MUTED};font-weight:400;">({total})</span></div>
    <div style="font:400 11px {FONT};color:{MUTED};padding-top:2px;">Current month only; historical months untouched{note}</div>
  </td></tr>
</table>
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#ffffff;border:1px solid {LINE};border-radius:6px;">
  <tr style="background:{SLATE_BG};">
    <th align="left" style="padding:7px 9px;font:700 10px {FONT};color:{SLATE};letter-spacing:.5px;">STOCK</th>
    <th align="left" style="padding:7px 9px;font:700 10px {FONT};color:{SLATE};letter-spacing:.5px;">YAHOO O/H/L/C</th>
    <th align="left" style="padding:7px 9px;font:700 10px {FONT};color:{SLATE};letter-spacing:.5px;">DAILY-DERIVED O/H/L/C</th>
    <th align="left" style="padding:7px 9px;font:700 10px {FONT};color:{SLATE};letter-spacing:.5px;">DRIFT</th>
  </tr>{body}
</table>"""

def format_email_html(df, as_of_label, nifty_above, drift_df=None, excluded_count=0):
    n_total = len(df)
    n_clear = int(df["is_clear"].sum()) if n_total else 0
    n_repeat = int(df["alert_seq"].gt(1).sum()) if n_total and "alert_seq" in df else 0

    nifty_txt = "Nifty 500 above 50 DMA" if nifty_above else "Nifty 500 below 50 DMA"
    nifty_badge = _badge(nifty_txt, GREEN, GREEN_BG) if nifty_above else _badge(nifty_txt, RED, RED_BG)

    if n_total:
        clear = df[df["is_clear"]].sort_values("body_pct", ascending=False)
        rest = df[~df["is_clear"]].sort_values("body_pct", ascending=False)
        sections = (
            _section("CLEAR", f"Body &ge; {CLEAR_BODY_PCT}% &middot; clearance &ge; {CLEAR_CLEARANCE_PCT}% "
                              f"&middot; volume &ge; {CLEAR_VOL_RATIO}x 50d average", clear, GREEN)
            + _section("OTHER", "Broke out, but did not meet every CLEAR threshold", rest, SLATE))
    else:
        sections = (f'<table role="presentation" width="100%" style="background:#ffffff;border:1px solid {LINE};'
                    f'border-radius:6px;margin-top:16px;"><tr><td align="center" style="padding:34px 16px;'
                    f'font:400 14px {FONT};color:{MUTED};">No breakouts today.</td></tr></table>')

    excl = (f'<div style="font:400 11px {FONT};color:{MUTED};padding-top:9px;">'
            f'{excluded_count} excluded &mdash; close had not cleared a more recent high.</div>'
            if excluded_count else "")

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#eef1f5;">
<table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background:#eef1f5;">
 <tr><td align="center" style="padding:22px 12px;">
  <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:680px;">

   <tr><td style="background:#16181d;border-radius:8px 8px 0 0;padding:19px 20px;">
     <div style="font:700 20px {FONT};color:#ffffff;letter-spacing:.2px;">Fresh Breakouts</div>
     <div style="font:400 13px {FONT};color:#9ba3af;padding-top:3px;">{html.escape(as_of_label)}</div>
     <div style="padding-top:11px;">{nifty_badge}</div>
   </td></tr>

   <tr><td style="background:#ffffff;border-left:1px solid {LINE};border-right:1px solid {LINE};padding:4px 8px;">
     <table role="presentation" cellpadding="0" cellspacing="0" width="100%"><tr>
       {_stat('Total', n_total, INK)}
       {_stat('Clear', n_clear, GREEN)}
       {_stat('Repeat', n_repeat, AMBER if n_repeat else MUTED)}
       {_stat('Excluded', excluded_count, MUTED)}
     </tr></table>
   </td></tr>

   <tr><td style="background:{SLATE_BG};border:1px solid {LINE};border-radius:0 0 8px 8px;padding:11px 15px;">
     <div style="font:400 12px {FONT};color:{SLATE};line-height:1.55;">
       <b style="color:{INK};">Entry</b> prior ATH &mdash; buy on retest to that level, close must hold above BO day low.<br>
       <b style="color:{INK};">Stop</b> BO day low on a close basis. No target &mdash; hold until stop.
     </div>{excl}
   </td></tr>

   <tr><td>{sections}{_drift_html(drift_df)}</td></tr>

   <tr><td style="padding:18px 4px 4px;font:400 11px {FONT};color:#9098a5;line-height:1.5;">
     Generated by the dark-knight-rises daily pipeline. Every alert above is recorded in the
     <code>alert_log</code> table for later review.
   </td></tr>

  </table>
 </td></tr>
</table>
</body></html>"""

def main(as_of=None):
    # Writable now (was read_only): this script owns alert_log.
    con = duckdb.connect(DB_PATH)
    init_alert_log_table(con)
    if as_of is None:
        as_of = con.execute("SELECT MAX(date) FROM daily_ohlc").fetchone()[0]

    nifty_above = is_nifty500_above_50dma(con)
    # First-time breakouts (breakout_date == today) plus any stock already
    # alerted earlier this month that is pushing to a fresh new high today.
    breakouts = pd.concat(
        [get_breakouts_on(con, as_of), get_repeat_breakouts_on(con, as_of)],
        ignore_index=True,
    )
    df = compute_quality(con, breakouts, as_of)
    valid, excluded_count = split_valid_and_excluded(df)
    valid = attach_alert_seq(con, valid)
    drift_df = get_drift_log_today(con, as_of)

    body = format_email_body(valid, str(as_of), nifty_above, drift_df, excluded_count)
    html_body = format_email_html(valid, str(as_of), nifty_above, drift_df, excluded_count)

    log_alerts(con, valid)
    con.close()

    n_total = len(valid)
    n_clear = int(valid["is_clear"].sum()) if n_total else 0
    n_repeat = int(valid["alert_seq"].gt(1).sum()) if n_total else 0
    subject = (f"[Fresh Breakouts] {n_total} today, {n_clear} CLEAR"
               + (f", {n_repeat} repeat" if n_repeat else "")
               + f" -- {as_of}")
    send_email(subject, body, html_body)

    return valid, body

if __name__ == "__main__":
    print(f"=== scan_fresh_breakouts.py started ===")
    df, body = main()
    print(body)
    print("=== Done ===")
