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

Two-tier scoring (validated on 2 years of historical signals, out-of-sample
tested with a time-based train/test split -- see local backtest research):
  - CONFIRMED candidates (breakout month has actually closed) get a 0-6
    score from breakout_strength + days_since_breakout + max_rally_pct +
    reversal_strong. breakout_strength (the breakout MONTH's close vs prior
    ATH) is the single strongest, most monotonic predictor found -- but it
    only means what it's supposed to mean once the month has actually
    closed, since it's implicitly measuring whether the stock held its
    gains through month-end, which can't be known while the month is still
    forming.
  - PRELIMINARY candidates (breakout month still in progress) get a 0-3
    score from ONLY days_since_breakout + max_rally_pct + reversal_strong.
    breakout_strength is excluded entirely, not estimated -- every
    daily-based substitute tested (breakout-day close, peak daily close
    before pullback) either carried no signal or measured the opposite
    thing (rally tightness, already captured by max_rally_pct). There is no
    reliable shortcut for "did it hold up through month-end" before the
    month has ended.

  reversal_strong: on the day price first touched the retest zone (the
  actual limit-order fill day -- low <= zone ceiling, regardless of that
  day's close), did the close end up back ABOVE the zone ceiling (a strong
  same-day bounce, buyers stepped in) rather than staying down near it (a
  weak bounce)? Validated on 1,062 historical signals: strong bounces hit
  +10% within 15 days 51.6% of the time (avg 5.9 days) vs 35.5% for weak
  bounces (avg 7.1 days), with a much lower stop-out rate too (42.3% vs
  60.7%). This is a same-day, daily-only signal -- no monthly dependency --
  so it applies equally to both tiers.

  These two scores are NOT on the same scale and must never be compared
  directly -- a preliminary "2" is not equivalent to a confirmed "2".
  volume_ratio was tested and deliberately excluded from both scores: it
  showed some signal in isolation but made the combined score less stable
  out-of-sample than leaving it out.

Alerts fire whenever a stock sets a new deepest pullback (today's low below
every prior post-departure low) -- not on every day it merely still sits
inside the 0-5% reporting band. This keeps re-alerting as a stock falls
closer to the breakout low (a genuinely better entry each time), while a
stock that lingers flat or bounces within a range it has already touched
goes quiet, since there's nothing new to act on.
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
            b.consolidation_months, b.breakout_strength
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
    month finishing.

    These are lower-confidence than get_active_breakouts() results (the
    month could still end up being a down month overall), so report them in
    a clearly separate, labeled section, never merged into the confirmed
    list, and scored on a different (lower-max) scale -- see module docstring.
    """
    today = date.today()
    current_month_start = today.replace(day=1)
    return con.execute("""
        SELECT
            b.symbol, b.breakout_month, b.breakout_date,
            b.breakout_day_open, b.breakout_day_high, b.breakout_day_low, b.breakout_day_close,
            b.consolidation_months, b.breakout_strength
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

def get_daily_since_breakout(con, breakouts):
    """
    Daily OHLC-lite (low, close) history per symbol, from each symbol's
    breakout_date through today -- needed to compute days-to-departure, the
    rally peak before entry, and the zone-touch reversal signal for scoring.
    One bulk query, not one per symbol.
    """
    if breakouts.empty:
        return {}
    symbols = breakouts["symbol"].unique().tolist()
    df = con.execute("""
        SELECT symbol, date, low, close FROM daily_ohlc
        WHERE symbol IN ({})
        ORDER BY symbol, date
    """.format(",".join(["?" for _ in symbols])), symbols).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    return {sym: g.reset_index(drop=True) for sym, g in df.groupby("symbol")}

def _score_features(bo, daily_lookup, as_of_date):
    """
    Shared feature computation:
      - days_since_breakout: days from breakout_date to now.
      - max_rally_pct: peak close reached above breakout_day_high, before now.
      - reversal_strong: on the day price first touched the retest zone (low
        <= zone ceiling), did that day's close end up back ABOVE the zone
        ceiling (a strong same-day bounce) rather than staying down near it?
        None if the zone hasn't been touched yet.
      - is_new_low: does TODAY's low set a new deepest pullback since
        departure (lower than every prior post-departure day), or is this
        the first day back inside the reporting band? Alerting "once, on
        first touch" was tried and is wrong -- if a stock only pokes the
        zone at 4.8% and keeps falling toward the actual breakout low, that
        deeper pullback is a BETTER entry (tighter stop, better reward) and
        must still get alerted. A stock bouncing around a level it has
        already touched has nothing new to offer, so only a fresh, deeper
        low re-triggers the alert -- this is what actually fixes repetition
        without silencing genuinely improving setups.
    Returns (days_since_breakout, max_rally_pct, reversal_strong, is_new_low),
    or (None, None, None, None) if there's not enough daily data to compute them.
    """
    sym = bo["symbol"]
    daily = daily_lookup.get(sym)
    if daily is None:
        return None, None, None, None

    bo_date = pd.to_datetime(bo["breakout_date"])
    bo_low = float(bo["breakout_day_low"])
    bo_high = float(bo["breakout_day_high"])
    after = daily[daily["date"] > bo_date]
    after = after[after["date"] <= pd.Timestamp(as_of_date)]
    if after.empty:
        return None, None, None, None

    departure_mask = after["close"] > bo_high
    if not departure_mask.any():
        return None, None, None, None  # shouldn't happen given the confirmed-departure SQL guard, but be safe

    days_since_breakout = (pd.to_datetime(as_of_date) - bo_date).days
    rally_peak = after.loc[departure_mask, "close"].max()
    max_rally_pct = (rally_peak - bo_high) / bo_high * 100

    departure_idx = departure_mask.idxmax()
    post_departure = after.loc[departure_idx:]
    zone_limit = bo_low * (1 + RECOMMENDED_ENTRY_PCT / 100)
    zone_touch_mask = (post_departure["low"] <= zone_limit) & (post_departure["close"] > bo_low)
    reversal_strong = None
    if zone_touch_mask.any():
        touch_idx = zone_touch_mask.idxmax()
        reversal_strong = bool(post_departure.loc[touch_idx, "close"] > zone_limit)

    today_idx = after.index[-1]
    prior_lows = post_departure[post_departure.index < today_idx]["low"]
    today_low = post_departure.loc[today_idx, "low"] if today_idx in post_departure.index else None
    if today_low is None:
        is_new_low = False  # today isn't even post-departure yet
    elif prior_lows.empty:
        is_new_low = True  # first day since departure -- nothing to compare against
    else:
        is_new_low = bool(today_low < prior_lows.min())

    return days_since_breakout, max_rally_pct, reversal_strong, is_new_low

def score_confirmed(breakout_strength, days_since_breakout, max_rally_pct, reversal_strong):
    """0-6. Validated out-of-sample: high scores here saw ~65-71% win rate vs ~25-33% at the bottom."""
    s = 0
    if breakout_strength >= 20: s += 3
    elif breakout_strength >= 10: s += 2
    elif breakout_strength >= 6: s += 1
    if days_since_breakout <= 6: s += 1
    if max_rally_pct <= 5: s += 1
    if reversal_strong: s += 1
    return s

def score_preliminary(days_since_breakout, max_rally_pct, reversal_strong):
    """0-3. Deliberately excludes breakout_strength -- see module docstring. Not comparable to score_confirmed."""
    s = 0
    if days_since_breakout <= 6: s += 1
    if max_rally_pct <= 5: s += 1
    if reversal_strong: s += 1
    return s

def compute_candidates(breakouts, price_map, as_of_date, daily_lookup=None, tier="confirmed"):
    """
    price_map: {symbol: {"price": float, "date": date}} -- current/latest price per
    symbol, from wherever the caller sourced it (daily_ohlc close, or a live quote).
    daily_lookup: {symbol: DataFrame[date, close]} from get_daily_since_breakout(),
    used to compute the score. If omitted, candidates are returned unscored
    (score=None) rather than guessing.
    tier: "confirmed" or "preliminary" -- selects which score function applies.

    Returns candidates with pct_from_low in (0, REPORT_ZONE_PCT], sorted by
    score descending (then consolidation, then closeness to the recommended
    entry). Excludes anything at or below breakout_day_low (support broken).
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

        # Trade plan, all derived from the one fact (breakout_day_low):
        #   entry  = the recommended limit price, RECOMMENDED_ENTRY_PCT above BO low
        #   stop   = BO day low itself (support broken -> exit)
        #   target = 10% above the entry price
        entry_price = bo_day_low * (1 + RECOMMENDED_ENTRY_PCT / 100)
        target_price = entry_price * 1.10
        stop_loss = bo_day_low

        score = None
        if daily_lookup is not None:
            days_since_breakout, max_rally_pct, reversal_strong, is_new_low = \
                _score_features(bo, daily_lookup, as_of_date)
            if days_since_breakout is not None:
                # Only alert when today is a NEW deepest pullback since
                # departure -- not just any day the stock happens to still be
                # sitting inside the reporting band. A stock bouncing around
                # a level already touched has nothing new to act on, but a
                # stock still falling toward the breakout low is offering a
                # genuinely better entry each time, and must keep alerting.
                if not is_new_low:
                    continue
                if tier == "confirmed":
                    score = score_confirmed(float(bo["breakout_strength"]), days_since_breakout, max_rally_pct, reversal_strong)
                else:
                    score = score_preliminary(days_since_breakout, max_rally_pct, reversal_strong)

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
            "entry_price": round(entry_price, 2),
            "target_price": round(target_price, 2),
            "stop_loss": round(stop_loss, 2),
            "recommended": abs(pct_from_low - RECOMMENDED_ENTRY_PCT) <= 1.0,
            "score": score,
            "tier": tier,
        })

    # Highest preference: score descending (the validated predictor), then
    # consolidation as a tiebreaker, then closeness to the recommended entry.
    return sorted(
        candidates,
        key=lambda c: (-(c["score"] if c["score"] is not None else -1), -c["consolidation_months"], c["pct_from_low"])
    )

def _format_candidate_lines(candidates, max_score):
    lines = []
    for i, c in enumerate(candidates, 1):
        star = " *" if c["recommended"] else ""
        score_str = f"{c['score']}/{max_score}" if c["score"] is not None else "n/a"
        lines.extend([
            f"{i}. {c['symbol']}{star}  --  Score : {score_str}  --  Consolidation : {c['consolidation_months']} months",
            f"   Current Price : {c['current_price']:.2f}",
            f"   Breakout      : "
            f"O:{c['breakout_open']:.2f} H:{c['breakout_high']:.2f} "
            f"L:{c['breakout_low']:.2f} C:{c['breakout_close']:.2f}",
            f"   From BO Low   : +{c['pct_from_low']:.2f}%",
            f"   Entry (3% from BO Low) : {c['entry_price']:.2f}",
            f"   Target (10% above entry) : {c['target_price']:.2f}",
            f"   Stop Loss (BO Low)     : {c['stop_loss']:.2f}",
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
        f"CONFIRMED sorted by score (0-6, validated), consolidation as tiebreaker",
        f"Candidates : {len(candidates)}",
        "",
    ]
    lines.extend(_format_candidate_lines(candidates, max_score=6))

    if preliminary_candidates:
        lines.extend([
            "=" * 50,
            "PRELIMINARY -- breakout month not yet closed",
            "Lower confidence: the month could still end down overall.",
            "Scored 0-3 (NOT comparable to the confirmed 0-6 score above --",
            "breakout_strength can't be measured until the month closes).",
            f"Candidates : {len(preliminary_candidates)}",
            "",
        ])
        lines.extend(_format_candidate_lines(preliminary_candidates, max_score=3))

    return "\n".join(lines)
