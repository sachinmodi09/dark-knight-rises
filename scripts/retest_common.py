"""
retest_common.py
Shared, stateless retest-zone logic used by both:
  - scan_retest.py       (once daily, after close, using daily_ohlc)
  - scan_retest_live.py  (hourly during market hours, using live quotes)

Nothing here writes to the database. breakout_monthly is read as the
factual record (a breakout's existence, and its breakout-day OHLC, don't
change); whether a stock is currently "in the buy zone" is a query against
that fact using today's price, not something to persist -- the definition
of "buy zone" is a tunable strategy choice, re-evaluated fresh every time.

Reporting window: 0% to REPORT_ZONE_PCT above breakout_day_low. Anything at
or below breakout_day_low (support broken) is excluded entirely.
RECOMMENDED_ENTRY_PCT is highlighted separately: backtesting (see
scripts/backtest_risk_reward.py) found entries within a few % of that point,
paired with a stop ~2% below breakout_day_low and a 3x target, gave the best
expectancy of anything tested for a 5-10 day hold.
"""

import duckdb
import pandas as pd
from datetime import date, timedelta

DB_PATH = "data/market.db"
INDEX_SYMBOL = "NIFTY500"
REPORT_ZONE_PCT = 5.0        # show everything from 0% up to this, above breakout_day_low
RECOMMENDED_ENTRY_PCT = 3.0  # flagged as the statistically best entry (see backtest_risk_reward.py)

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
    ma_50 = float(df["close"].mean())
    above = latest_close > ma_50
    print(f"  Nifty500 close={latest_close:.2f}, 50DMA={ma_50:.2f}, above={above}")
    return above

def get_active_breakouts(con):
    """
    Active breakouts (last 12 months) with a known breakout day -- the factual
    basis for scanning.

    Two guards against premature/unconfirmed breakouts:

    1. The breakout's month must have fully closed. breakout_month for the
       current, still-in-progress calendar month is a monthly candle built
       from however many trading days have happened so far -- its close can
       still move a lot before the month actually ends, so a "breakout"
       against it isn't confirmed yet. Only monthly candles from a month
       that has actually finished are trustworthy.
    2. Confirmed departure: at least one close after breakout_date must have
       exceeded breakout_day_high. Without this, a breakout that's only 1-3
       days old trivially sits "near the low" simply because it hasn't moved
       yet -- that's not a retest, it's a stock that never left the starting
       line. A real retest needs the round trip: breakout, rally away, pull back.
    """
    today = date.today()
    cutoff = today - timedelta(days=365)
    current_month_start = today.replace(day=1)
    return con.execute("""
        SELECT
            b.symbol, b.breakout_month, b.breakout_date,
            b.breakout_day_open, b.breakout_day_high, b.breakout_day_low, b.breakout_day_close,
            b.consolidation_months
        FROM breakout_monthly b
        WHERE b.status = 'active'
        AND b.breakout_month >= ?
        AND b.breakout_month < ?
        AND b.breakout_date IS NOT NULL
        AND b.breakout_date <= ?
        AND b.breakout_day_low IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM daily_ohlc d
            WHERE d.symbol = b.symbol
            AND d.date > b.breakout_date
            AND d.close > b.breakout_day_high
        )
    """, [cutoff, current_month_start, today]).fetchdf()

def get_preliminary_breakouts(con):
    """
    Breakouts from the CURRENT, still-in-progress calendar month -- the ones
    get_active_breakouts() deliberately excludes. Backtesting the last 12
    months showed 56% of breakouts retest to within 3% of breakout_day_low
    before their month even closes, and a third of those hit +10% within 10
    days (avg 5.7 days) -- waiting for month-end confirmation costs the
    fastest, cheapest entries. The breakout trigger itself (close > the
    PRIOR, already-confirmed month's ATH) doesn't depend on the current
    month finishing -- only consolidation_months/prev_ath_month do, and
    those aren't used here.

    These are lower-confidence than get_active_breakouts() results (the
    month could still end up being a down month overall, and mark_breakouts.py
    hasn't run for it yet in the normal monthly cadence -- this only finds
    something if a breakout_monthly row for the current month already
    exists), so report them in a clearly separate, labeled section, never
    merged into the confirmed list.
    """
    today = date.today()
    current_month_start = today.replace(day=1)
    return con.execute("""
        SELECT
            b.symbol, b.breakout_month, b.breakout_date,
            b.breakout_day_open, b.breakout_day_high, b.breakout_day_low, b.breakout_day_close,
            b.consolidation_months
        FROM breakout_monthly b
        WHERE b.status = 'active'
        AND b.breakout_month >= ?
        AND b.breakout_date IS NOT NULL
        AND b.breakout_date <= ?
        AND b.breakout_day_low IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM daily_ohlc d
            WHERE d.symbol = b.symbol
            AND d.date > b.breakout_date
            AND d.close > b.breakout_day_high
        )
    """, [current_month_start, today]).fetchdf()

def compute_candidates(breakouts, price_map, as_of_date):
    """
    price_map: {symbol: {"price": float, "date": date}} -- current/latest price per
    symbol, from wherever the caller sourced it (daily_ohlc close, or a live quote).

    Returns candidates with pct_from_low in (0, REPORT_ZONE_PCT], sorted ascending.
    Excludes anything at or below breakout_day_low (support broken/untested).
    """
    candidates = []
    for _, bo in breakouts.iterrows():
        sym = bo["symbol"]
        if sym not in price_map:
            continue

        bo_date = pd.to_datetime(bo["breakout_date"]).date()
        if as_of_date <= bo_date:
            continue

        bo_day_low = float(bo["breakout_day_low"])
        current_price = float(price_map[sym]["price"])

        if current_price <= bo_day_low:
            continue  # support broken (or untested) -- not a valid retest

        pct_from_low = (current_price - bo_day_low) / bo_day_low * 100
        if pct_from_low > REPORT_ZONE_PCT:
            continue

        candidates.append({
            "symbol": sym,
            "breakout_month": pd.to_datetime(bo["breakout_month"]).date(),
            "breakout_date": bo_date,
            "current_price": round(current_price, 2),
            "pct_from_low": round(pct_from_low, 2),
            "days_since_breakout": (as_of_date - bo_date).days,
            "breakout_open": float(bo["breakout_day_open"]),
            "breakout_high": float(bo["breakout_day_high"]),
            "breakout_low": bo_day_low,
            "breakout_close": float(bo["breakout_day_close"]),
            "consolidation_months": int(bo["consolidation_months"]),
            "recommended": abs(pct_from_low - RECOMMENDED_ENTRY_PCT) <= 1.0,
        })

    # Highest preference: longest consolidation first (a breakout from a longer
    # base is the stronger signal); pct_from_low only breaks ties within that.
    return sorted(candidates, key=lambda c: (-c["consolidation_months"], c["pct_from_low"]))

def _format_candidate_lines(candidates):
    lines = []
    for i, c in enumerate(candidates, 1):
        star = " *" if c["recommended"] else ""
        lines.extend([
            f"{i}. {c['symbol']}{star}  --  Consolidation : {c['consolidation_months']} months",
            f"   Current Price : {c['current_price']:.2f}",
            f"   Breakout      : "
            f"O:{c['breakout_open']:.2f} H:{c['breakout_high']:.2f} "
            f"L:{c['breakout_low']:.2f} C:{c['breakout_close']:.2f}",
            f"   From BO Low   : +{c['pct_from_low']:.2f}%",
            f"   Days Since BO : {c['days_since_breakout']}",
            "",
        ])
    return lines

def format_email_body(candidates, nifty_above, as_of_label, preliminary_candidates=None):
    lines = [
        f"Retest Scan - {as_of_label}",
        f"Nifty 500 Above 50 DMA : {nifty_above}",
        f"Reporting window : 0% to {REPORT_ZONE_PCT:.0f}% above breakout day low "
        f"(* = near the recommended {RECOMMENDED_ENTRY_PCT:.0f}% entry)",
        f"Sorted by consolidation period, longest first",
        f"Candidates : {len(candidates)}",
        "",
    ]
    lines.extend(_format_candidate_lines(candidates))

    if preliminary_candidates:
        lines.extend([
            "=" * 50,
            "PRELIMINARY -- breakout month not yet closed",
            "Lower confidence: the month could still end down overall.",
            f"Candidates : {len(preliminary_candidates)}",
            "",
        ])
        lines.extend(_format_candidate_lines(preliminary_candidates))

    return "\n".join(lines)
