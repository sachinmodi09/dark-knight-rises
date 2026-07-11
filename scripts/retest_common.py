"""
retest_common.py
Shared, stateless retest logic used by both:
  - scan_retest.py       (once daily, after close, using daily_ohlc)
  - scan_retest_live.py  (hourly during market hours, using live quotes)

Nothing here writes to the database. breakout_monthly is read as the
factual record (a breakout's existence, and its breakout-day OHLC, don't
change); whether a stock is currently retesting is a query against that
fact using today's price, not something to persist -- the definition of
"retest" is a tunable strategy choice, re-evaluated fresh every time.

Entry model: the breakout day's OPEN price, not a fixed % above the low.
Validated on 258 historical signals (retest_days/retest_depth/EMA study,
3-month forward window): this reference, held with a stop at breakout day
low (close basis) and NO fixed target, produced +1699% summed return
across 258 trades vs at best +1087% for the best trailing-stop variant
tested -- these are volatile small/midcaps that need room to breathe, and
any target or trail tight enough to lock in gains also cuts the rare big
winners short (one signal alone ran +92% held vs +4% under a 15%/10%
trailing stop). So: no target line, no trailing stop -- hold until the
close breaks the breakout day low, or exit is a manual decision.

Scoring: same two-tier design as before (CONFIRMED vs PRELIMINARY, see
below), but the four scored features changed. The original score's
days_since_breakout<=6 threshold was recalibrated as part of this: the
258-trade study found winners average 29.6 days from breakout to retest
(losers 50.4) -- both far past a 6-day window, so that old cutoff almost
never fired and wasn't testing what actually separates winners. The
features below are what held up with real statistical significance
(Welch's t-test, winners n=53 vs stopped-out n=205):
  - retest_depth_pct: how far price fell from its post-breakout rally
    peak to the breakout-day-open touch. Winners 11.25% vs losers 15.24%
    (p=0.005) -- a shallow pullback beats a deep one.
  - retest_days: trading days from the rally peak to the touch. Winners
    10.6 vs losers 22.4 (p=0.0002) -- a quick bounce beats a slow, weak one.
  - dist_ema20_pct / dist_ema50_pct: touch-day close vs its own 20- and
    50-day EMA. Winners sit above both (+0.87%, +5.81%); losers are
    already below the 20 EMA on average (-2.46%) and barely above the 50
    (+1.10%) (p=0.0008, p=0.002) -- the winners' trend hadn't cracked yet.
  Volume-based features (pullback volume, bounce volume) and candle-shape
  (bullish reversal candle) were also tested and did NOT reach
  significance (p>0.1) -- deliberately left out, not an oversight.

  CONFIRMED (breakout month closed): breakout_strength (0-3, unchanged --
  still the single strongest, most monotonic predictor, but only means
  what it should once the month has actually closed) + the 4 features
  above (0-4) = 0-7.
  PRELIMINARY (month still forming): the 4 features only = 0-4.
  breakout_strength is excluded from PRELIMINARY, not estimated -- no
  reliable daily substitute for "did it hold up through month-end" was
  ever found (see git history). These two scores are NOT on the same
  scale and must never be compared directly.

Alerts fire whenever a stock sets a new deepest pullback (today's low
below every prior post-departure low) -- not on every day it merely still
sits above breakout day low. This re-alerts as a stock falls closer to
the breakout low (a genuinely better entry each time) while staying quiet
for a stock that lingers flat or bounces within a range already touched.
"""

import duckdb
import pandas as pd
from datetime import date, timedelta

DB_PATH = "data/market.db"
INDEX_SYMBOL = "NIFTY500"
RETEST_DEPTH_MAX_PCT = 12.0   # winners avg 11.25%, losers avg 15.24%
RETEST_DAYS_MAX = 15          # winners avg 10.6 days, losers avg 22.4 days
EMA50_MIN_PCT = 3.0           # winners avg +5.81% above 50 EMA, losers avg +1.10%

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
    months showed most breakouts retest well before their month even closes
    -- waiting for month-end confirmation costs the fastest, cheapest
    entries. The breakout trigger itself (close > the PRIOR, already-
    confirmed month's ATH) doesn't depend on the current month finishing.

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
    Full daily OHLCV history per symbol (not date-bounded to "since
    breakout") -- the 20/50-day EMA needs real pre-breakout price history
    to be meaningful, not a series that starts cold on the breakout date.
    One bulk query, not one per symbol.
    """
    if breakouts.empty:
        return {}
    symbols = breakouts["symbol"].unique().tolist()
    df = con.execute("""
        SELECT symbol, date, high, low, close FROM daily_ohlc
        WHERE symbol IN ({})
        ORDER BY symbol, date
    """.format(",".join(["?" for _ in symbols])), symbols).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    result = {}
    for sym, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        g["ema20"] = g["close"].ewm(span=20, adjust=False).mean()
        g["ema50"] = g["close"].ewm(span=50, adjust=False).mean()
        result[sym] = g
    return result

def _score_features(bo, daily_lookup, as_of_date):
    """
    Shared feature computation, all keyed off the breakout-day-OPEN touch
    (see module docstring for validation):
      - retest_depth_pct: % fall from the post-breakout rally peak close
        to the touch day's low.
      - retest_days: trading days from the rally peak to the touch day.
      - dist_ema20_pct / dist_ema50_pct: touch-day close vs its 20/50-day EMA.
      - is_new_low: does TODAY's low set a new deepest pullback since
        departure (lower than every prior post-departure day)? Alerting
        "once, on first touch" was tried and is wrong -- a stock still
        falling toward the breakout low is offering a genuinely better
        entry each time it does, and must keep alerting; a stock bouncing
        around a level already touched has nothing new to offer.
    Returns (retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct,
    is_new_low), or (None, None, None, None, None) if there's not enough
    daily data to compute them.
    """
    sym = bo["symbol"]
    daily = daily_lookup.get(sym)
    if daily is None:
        return None, None, None, None, None

    bo_date = pd.to_datetime(bo["breakout_date"])
    bo_open = float(bo["breakout_day_open"])
    bo_low = float(bo["breakout_day_low"])
    bo_high = float(bo["breakout_day_high"])
    after = daily[daily["date"] > bo_date]
    after = after[after["date"] <= pd.Timestamp(as_of_date)]
    if after.empty:
        return None, None, None, None, None

    departure_mask = after["close"] > bo_high
    if not departure_mask.any():
        return None, None, None, None, None  # shouldn't happen given the confirmed-departure SQL guard, but be safe

    departure_idx = departure_mask.idxmax()
    post_departure = after.loc[departure_idx:]
    # Touch search excludes the departure day itself: a day that both
    # confirms departure AND dips back to bo_open (a same-day gap-down-then-
    # rally) isn't a genuine post-rally pullback, just departure-day noise.
    after_departure_day = post_departure[post_departure.index > departure_idx]
    touch_mask = (after_departure_day["low"] <= bo_open) & (after_departure_day["close"] > bo_low)
    if not touch_mask.any():
        return None, None, None, None, None  # hasn't touched the breakout-day open yet

    touch_idx = touch_mask.idxmax()
    rally_seg = post_departure.loc[:touch_idx]
    peak_idx = rally_seg["close"].idxmax()
    rally_peak_close = rally_seg.loc[peak_idx, "close"]
    touch_row = post_departure.loc[touch_idx]

    retest_depth_pct = (rally_peak_close - touch_row["low"]) / rally_peak_close * 100
    retest_days = after.index.get_loc(touch_idx) - after.index.get_loc(peak_idx)
    dist_ema20_pct = (touch_row["close"] - touch_row["ema20"]) / touch_row["ema20"] * 100
    dist_ema50_pct = (touch_row["close"] - touch_row["ema50"]) / touch_row["ema50"] * 100

    today_idx = after.index[-1]
    prior_lows = post_departure[post_departure.index < today_idx]["low"]
    today_low = post_departure.loc[today_idx, "low"] if today_idx in post_departure.index else None
    if today_low is None:
        is_new_low = False  # today isn't even post-departure yet
    elif prior_lows.empty:
        is_new_low = True  # first day since departure -- nothing to compare against
    else:
        is_new_low = bool(today_low < prior_lows.min())

    return round(retest_depth_pct, 2), retest_days, round(dist_ema20_pct, 2), round(dist_ema50_pct, 2), is_new_low

def score_confirmed(breakout_strength, retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct):
    """0-7. See module docstring for the significance testing behind each cutoff."""
    s = 0
    if breakout_strength >= 20: s += 3
    elif breakout_strength >= 10: s += 2
    elif breakout_strength >= 6: s += 1
    if retest_depth_pct <= RETEST_DEPTH_MAX_PCT: s += 1
    if retest_days <= RETEST_DAYS_MAX: s += 1
    if dist_ema20_pct >= 0: s += 1
    if dist_ema50_pct >= EMA50_MIN_PCT: s += 1
    return s

def score_preliminary(retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct):
    """0-4. Deliberately excludes breakout_strength -- see module docstring. Not comparable to score_confirmed."""
    s = 0
    if retest_depth_pct <= RETEST_DEPTH_MAX_PCT: s += 1
    if retest_days <= RETEST_DAYS_MAX: s += 1
    if dist_ema20_pct >= 0: s += 1
    if dist_ema50_pct >= EMA50_MIN_PCT: s += 1
    return s

def compute_candidates(breakouts, price_map, as_of_date, daily_lookup=None, tier="confirmed"):
    """
    price_map: {symbol: {"price": float, "date": date}} -- current/latest price per
    symbol, from wherever the caller sourced it (daily_ohlc close, or a live quote).
    daily_lookup: {symbol: DataFrame[date, high, low, close, ema20, ema50]} from
    get_daily_since_breakout(), used to compute the score. If omitted, candidates
    are returned unscored (score=None) rather than guessing.
    tier: "confirmed" or "preliminary" -- selects which score function applies.

    A candidate only appears the day its low sets a fresh, deeper pullback
    toward breakout_day_open (see _score_features/is_new_low) -- not on
    every day it merely still trades above breakout_day_low. Excludes
    anything at or below breakout_day_low entirely (support broken).
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

        # Trade plan: entry at the breakout day's open (validated reference,
        # see module docstring), stop at breakout day low on a CLOSE basis.
        # Deliberately no target line -- see module docstring on why a
        # fixed target/trailing stop underperformed a plain hold-to-stop.
        entry_price = float(bo["breakout_day_open"])
        stop_loss = bo_day_low

        score = None
        if daily_lookup is not None:
            retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct, is_new_low = \
                _score_features(bo, daily_lookup, as_of_date)
            if retest_depth_pct is None:
                continue  # hasn't touched breakout_day_open yet -- not a candidate at all
            # Only alert when today is a NEW deepest pullback since departure
            # -- not just any day the stock happens to still be above
            # breakout_day_low. A stock bouncing around a level already
            # touched has nothing new to act on, but a stock still falling
            # toward the breakout low is offering a genuinely better entry
            # each time, and must keep alerting.
            if not is_new_low:
                continue
            if tier == "confirmed":
                score = score_confirmed(float(bo["breakout_strength"]), retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct)
            else:
                score = score_preliminary(retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct)
        else:
            retest_depth_pct = retest_days = dist_ema20_pct = dist_ema50_pct = None

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
            "stop_loss": round(stop_loss, 2),
            "retest_depth_pct": retest_depth_pct,
            "retest_days": retest_days,
            "dist_ema20_pct": dist_ema20_pct,
            "dist_ema50_pct": dist_ema50_pct,
            "score": score,
            "tier": tier,
        })

    # Highest preference: score descending (the validated predictor), then
    # consolidation as a tiebreaker, then closeness to the entry price.
    return sorted(
        candidates,
        key=lambda c: (-(c["score"] if c["score"] is not None else -1), -c["consolidation_months"], c["pct_from_low"])
    )

def _format_candidate_lines(candidates, max_score):
    lines = []
    for i, c in enumerate(candidates, 1):
        score_str = f"{c['score']}/{max_score}" if c["score"] is not None else "n/a"
        lines.extend([
            f"{i}. {c['symbol']}  --  Score : {score_str}  --  Consolidation : {c['consolidation_months']} months",
            f"   Current Price : {c['current_price']:.2f}",
            f"   Breakout      : "
            f"O:{c['breakout_open']:.2f} H:{c['breakout_high']:.2f} "
            f"L:{c['breakout_low']:.2f} C:{c['breakout_close']:.2f}",
            f"   From BO Low   : +{c['pct_from_low']:.2f}%",
            f"   Entry (BO Open) : {c['entry_price']:.2f}",
            f"   Stop Loss (BO Low, close basis) : {c['stop_loss']:.2f}",
            f"   Retest Depth : {c['retest_depth_pct']}%   Retest Days : {c['retest_days']}   "
            f"vs 20EMA : {c['dist_ema20_pct']}%   vs 50EMA : {c['dist_ema50_pct']}%"
            if c["retest_depth_pct"] is not None else "   (unscored -- no daily history available)",
            f"   Days Since BO : {c['days_since_breakout']}",
            "",
        ])
    return lines

def format_email_body(candidates, nifty_above, as_of_label, preliminary_candidates=None):
    lines = [
        f"Retest Scan - {as_of_label}",
        f"Nifty 500 Above 50 DMA : {nifty_above}",
        f"Entry : breakout day OPEN.  Stop : breakout day LOW (close basis).  No fixed target -- hold until stop.",
        f"CONFIRMED sorted by score (0-7, validated), consolidation as tiebreaker",
        f"Candidates : {len(candidates)}",
        "",
    ]
    lines.extend(_format_candidate_lines(candidates, max_score=7))

    if preliminary_candidates:
        lines.extend([
            "=" * 50,
            "PRELIMINARY -- breakout month not yet closed",
            "Lower confidence: the month could still end down overall.",
            "Scored 0-4 (NOT comparable to the confirmed 0-7 score above --",
            "breakout_strength can't be measured until the month closes).",
            f"Candidates : {len(preliminary_candidates)}",
            "",
        ])
        lines.extend(_format_candidate_lines(preliminary_candidates, max_score=4))

    return "\n".join(lines)
