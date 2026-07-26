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
    peak to the approach day's low. Winners 11.25% vs losers 15.24%
    (p=0.005) -- a shallow pullback beats a deep one.
  - retest_days: trading days from the rally peak to the approach. Winners
    10.6 vs losers 22.4 (p=0.0002) -- a quick bounce beats a slow, weak one.
  - dist_ema20_pct / dist_ema50_pct: approach-day close vs its own 20- and
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

Alerts fire when price gets within APPROACH_PCT of breakout_day_open (not
only on the exact touch), whenever that's a new closest approach since
the breakout itself -- not on every day it merely still sits above
breakout day low. The pipeline runs once, after close; if it only
alerted on the exact touch, you couldn't react until the next trading
day, and by then price often already moved on (checked empirically: 3 of
10 real signals in one window never came back to the exact entry after
the earliest day you could have placed the order). Alerting on approach
means the resting limit order goes in BEFORE the touch happens, so
whenever it does -- same day or later -- it just fills. This re-alerts
as a stock falls closer (a genuinely better entry each time) while
staying quiet for a stock that lingers flat or bounces within a range it
has already gotten this close to.

No requirement that the stock first rally above breakout_day_high before
it's eligible ("confirmed departure") -- a stock is watched from the day
after its breakout onward. An earlier version required that rally-first
round trip and it cost a real entry: MANAPPURAM dipped within 1% of its
breakout_day_open just 3 days after breaking out, but wasn't eligible to
alert on it under that rule, and by the time it finally satisfied
"confirmed departure" the stock had already run ~8% further. The
is_new_low freshness check (see _score_features) is what actually
prevents repetition, not a rally-first precondition.
"""

import duckdb
import numpy as np
import pandas as pd
from datetime import date, timedelta

DB_PATH = "data/market.db"
INDEX_SYMBOL = "NIFTY500"
RETEST_DEPTH_MAX_PCT = 12.0   # winners avg 11.25%, losers avg 15.24%
RETEST_DAYS_MAX = 15          # winners avg 10.6 days, losers avg 22.4 days
EMA50_MIN_PCT = 3.0           # winners avg +5.81% above 50 EMA, losers avg +1.10%
APPROACH_PCT = 1.0            # alert when price is within this % above breakout_day_open,
                               # not only on the exact touch -- see _score_features docstring.
                               # Tested 1/2/3/5%: 1% keeps volume near the old ~1/day pace
                               # (15 alerts / 10 days) while still giving the resting limit
                               # order days of lead time before the actual touch, instead of
                               # the hard next-day-only window an exact-touch trigger forces.

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

    One guard: the breakout's month must have fully closed. breakout_month
    for the current, still-in-progress calendar month is a monthly candle
    built from however many trading days have happened so far -- its close
    can still move a lot before the month actually ends, so a "breakout"
    against it isn't confirmed yet. Only monthly candles from a month that
    has actually finished are trustworthy.

    Deliberately does NOT require the stock to have rallied above
    breakout_day_high before it's eligible to watch -- the model is simply
    "breakout happened, watch for price to return to breakout_day_open,"
    not "breakout happened, then wait for a fresh high, then watch." An
    earlier version required that "confirmed departure" and it cost a real,
    concrete entry: MANAPPURAM dipped to within 1% of its breakout_day_open
    on 2026-07-09, 3 days after breaking out, but wasn't eligible to alert
    on it because it hadn't yet closed above its breakout_day_high -- that
    only happened on 07-21, by which point the stock had already run to 352
    (the 07-09 entry was ~330). Nothing here prevents a fresh, 1-day-old
    breakout from qualifying immediately if it dips back that quickly --
    that's intended, not a gap: _score_features' is_new_low check still
    requires each alert to be a genuinely deeper approach than any before
    it, so a stock sitting flat near its own breakout day doesn't repeat.
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
    """, [cutoff, current_month_start, today]).fetchdf()

def get_preliminary_breakouts(con):
    """
    Breakouts from the CURRENT, still-in-progress calendar month -- the ones
    get_active_breakouts() deliberately excludes. Backtesting the last 12
    months showed most breakouts retest well before their month even closes
    -- waiting for month-end confirmation costs the fastest, cheapest
    entries. The breakout trigger itself (close > the PRIOR, already-
    confirmed month's ATH) doesn't depend on the current month finishing.

    Same as get_active_breakouts(): no "confirmed departure" precondition
    required -- see that function's docstring for why.

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
    Shared feature computation, keyed off the breakout-day-OPEN approach
    (see module docstring for validation of the underlying entry model):
      - retest_depth_pct: % fall from the highest close reached since
        breakout to TODAY's low.
      - retest_days: trading days from that peak to today.
      - dist_ema20_pct / dist_ema50_pct: today's close vs its 20/50-day EMA.
    All four are computed relative to TODAY specifically -- not whenever
    the stock first got close. A stock can approach breakout_day_open
    once, drift, then fall to an even deeper low weeks later; that deeper
    day is a genuinely better (and different) entry, with its own
    depth/speed/trend numbers, not a repeat of the original approach's stats.

    No requirement that the stock first rallied above breakout_day_high --
    see get_active_breakouts' docstring for why that was removed. A stock
    can qualify as soon as the day after breakout if it dips back that
    quickly; the peak used for retest_depth/retest_days is simply the
    highest close reached since breakout up to today, whatever that is
    (even if today IS that peak, giving depth=0/days=0 for a stock still
    climbing that hasn't pulled back at all).

    Trigger is APPROACH_PCT above breakout_day_open, not an exact touch.
    The alert only reaches you after close (the pipeline runs once, end
    of day) -- if it only fired on the day price actually touched the
    entry, you couldn't act until the NEXT day at the earliest, and by
    then price has often already moved on (checked empirically: 3 of 10
    real signals over one 10-day window never came back down to the exact
    touch price after the earliest day you could have reacted). Alerting
    on approach instead means you place the resting limit order BEFORE
    the touch happens, so whenever it does -- that same day or later --
    it just fills, with no lag.

    Returns (retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct)
    only when TODAY itself is a fresh approach (a new closest point to
    breakout_day_open since the breakout) -- otherwise (None, None, None,
    None): either the stock hasn't gotten close yet, or today isn't a new
    closest approach and there's nothing fresh to report.
    """
    sym = bo["symbol"]
    daily = daily_lookup.get(sym)
    if daily is None:
        return None, None, None, None, None

    bo_date = pd.to_datetime(bo["breakout_date"])
    bo_open = float(bo["breakout_day_open"])
    bo_low = float(bo["breakout_day_low"])
    approach_limit = bo_open * (1 + APPROACH_PCT / 100)
    after = daily[daily["date"] > bo_date]
    after = after[after["date"] <= pd.Timestamp(as_of_date)]
    if after.empty:
        return None, None, None, None, None

    today_idx = after.index[-1]
    today_row = after.loc[today_idx]
    if not (today_row["low"] <= approach_limit and today_row["close"] > bo_low):
        return None, None, None, None, None  # today isn't near breakout_day_open at all

    # Fresh only if today is a NEW closest approach since breakout
    # (excluding today itself from the comparison).
    prior = after[after.index != today_idx]
    if not prior.empty and today_row["low"] >= prior["low"].min():
        return None, None, None, None, None  # already gotten this close or closer before -- nothing new

    rally_seg = after.loc[:today_idx]
    peak_idx = rally_seg["close"].idxmax()
    rally_peak_close = rally_seg.loc[peak_idx, "close"]

    retest_depth_pct = (rally_peak_close - today_row["low"]) / rally_peak_close * 100
    retest_days = after.index.get_loc(today_idx) - after.index.get_loc(peak_idx)
    dist_ema20_pct = (today_row["close"] - today_row["ema20"]) / today_row["ema20"] * 100
    dist_ema50_pct = (today_row["close"] - today_row["ema50"]) / today_row["ema50"] * 100

    # today_row["low"] is returned too -- it's what actually triggered this
    # alert (the approach check is low-based), while "current price" shown
    # to the user is the close, which can end up meaningfully higher on a
    # day with a strong intraday recovery. Without showing the low, a
    # stock that dipped near entry then rallied hard looks like a display
    # bug ("why does this show as approaching when the price is way above
    # entry?") instead of the genuinely bullish signal it actually is.
    return round(retest_depth_pct, 2), retest_days, round(dist_ema20_pct, 2), round(dist_ema50_pct, 2), round(float(today_row["low"]), 2)

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

        # Trading days elapsed, not calendar days -- (as_of_date - bo_date).days
        # counts weekends as if the market were open, inflating the figure
        # (e.g. a Thursday breakout checked the following Sunday reads as "3
        # days" when only 1 trading day -- Friday -- actually happened). Count
        # real rows in the daily history when available; fall back to a
        # weekday-only approximation (still excludes weekends, just not
        # market holidays) when daily_lookup wasn't supplied.
        if daily_lookup is not None and sym in daily_lookup:
            sym_daily = daily_lookup[sym]
            trading_days_since_breakout = int((
                (sym_daily["date"] > pd.Timestamp(bo_date)) &
                (sym_daily["date"] <= pd.Timestamp(as_of_date))
            ).sum())
        else:
            trading_days_since_breakout = int(np.busday_count(bo_date, as_of_date))

        # Trade plan: entry at the breakout day's open (validated reference,
        # see module docstring), stop at breakout day low on a CLOSE basis.
        # Deliberately no target line -- see module docstring on why a
        # fixed target/trailing stop underperformed a plain hold-to-stop.
        entry_price = float(bo["breakout_day_open"])
        stop_loss = bo_day_low

        score = None
        if daily_lookup is not None:
            retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct, today_low = \
                _score_features(bo, daily_lookup, as_of_date)
            if retest_depth_pct is None:
                # Either hasn't touched breakout_day_open yet, or today isn't
                # a fresh (new-deepest-low) touch -- see _score_features
                # docstring. Either way, nothing to alert on today.
                continue
            if tier == "confirmed":
                score = score_confirmed(float(bo["breakout_strength"]), retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct)
            else:
                score = score_preliminary(retest_depth_pct, retest_days, dist_ema20_pct, dist_ema50_pct)
        else:
            retest_depth_pct = retest_days = dist_ema20_pct = dist_ema50_pct = today_low = None

        candidates.append({
            "symbol": sym,
            "breakout_month": pd.to_datetime(bo["breakout_month"]).date(),
            "breakout_date": bo_date,
            "current_price": round(current_price, 2),
            "today_low": today_low,
            "pct_from_low": round(pct_from_low, 2),
            "days_since_breakout": trading_days_since_breakout,
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
            f"   Current Price : {c['current_price']:.2f}"
            + (f"   (today's low: {c['today_low']:.2f} -- this is what triggered the alert)" if c["today_low"] is not None else ""),
            f"   Breakout Date : {c['breakout_date']}",
            f"   Breakout      : "
            f"O:{c['breakout_open']:.2f} H:{c['breakout_high']:.2f} "
            f"L:{c['breakout_low']:.2f} C:{c['breakout_close']:.2f}",
            f"   From BO Low   : +{c['pct_from_low']:.2f}%",
            f"   Entry (BO Open) : {c['entry_price']:.2f}",
            f"   Stop Loss (BO Low, close basis) : {c['stop_loss']:.2f}",
            f"   Retest Depth : {c['retest_depth_pct']}%   Retest Days : {c['retest_days']}   "
            f"vs 20EMA : {c['dist_ema20_pct']}%   vs 50EMA : {c['dist_ema50_pct']}%"
            if c["retest_depth_pct"] is not None else "   (unscored -- no daily history available)",
            f"   Trading Days Since BO : {c['days_since_breakout']}",
            "",
        ])
    return lines

def format_email_body(candidates, nifty_above, as_of_label, preliminary_candidates=None):
    lines = [
        f"Retest Scan - {as_of_label}",
        f"Nifty 500 Above 50 DMA : {nifty_above}",
        f"Entry : breakout day OPEN (place a RESTING limit order here, may not fill today).",
        f"Stop : breakout day LOW (close basis).  No fixed target -- hold until stop.",
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
