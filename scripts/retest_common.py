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
    """Active breakouts (last 12 months) with a known breakout day -- the factual basis for scanning."""
    today = date.today()
    cutoff = today - timedelta(days=365)
    return con.execute("""
        SELECT
            symbol, breakout_month, breakout_date,
            breakout_day_open, breakout_day_high, breakout_day_low, breakout_day_close,
            consolidation_months
        FROM breakout_monthly
        WHERE status = 'active'
        AND breakout_month >= ?
        AND breakout_date IS NOT NULL
        AND breakout_date <= ?
        AND breakout_day_low IS NOT NULL
    """, [cutoff, today]).fetchdf()

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

def format_email_body(candidates, nifty_above, as_of_label):
    lines = [
        f"Retest Scan - {as_of_label}",
        f"Nifty 500 Above 50 DMA : {nifty_above}",
        f"Reporting window : 0% to {REPORT_ZONE_PCT:.0f}% above breakout day low "
        f"(* = near the recommended {RECOMMENDED_ENTRY_PCT:.0f}% entry)",
        f"Sorted by consolidation period, longest first",
        f"Candidates : {len(candidates)}",
        "",
    ]

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
    return "\n".join(lines)
