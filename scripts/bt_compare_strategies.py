"""
bt_compare_strategies.py
Step 3: the actual decision -- trade the 5/10% pops, or hold for a multiple?

Runs four things over the same 109 tradeable CLEAR breakouts:

  A  HOLD with stop, no target. Same entry and stop as the trading plan, but
     nothing is sold into strength -- the position runs until it closes below
     the breakout day low. This is the "wait for 2x-3x" strategy.
  B  What the 29 never-retested signals went on to do, which decides whether
     insisting on a retest entry is quietly filtering out the winners.
  C  Buying the breakout CLOSE instead of waiting for a retest -- fills every
     signal, at a worse price.
  D  Capital actually required: peak simultaneous open positions, since the
     trading plan recycles money and "Rs 1 lakh x 80 trades" is not Rs 80 lakh.

Horizon honesty: daily data ends 2026-09-04, so a 2025-09 breakout has ~12
months of forward bars while a 2026-08 one has ~1 month. Hold results are
therefore reported by cohort age, never pooled into a single "average
return" that would be dominated by trades that have barely started.
"""

import duckdb
import pandas as pd

DB_PATH = "data/market.db"
MAX_HOLD_DAYS = 400
CAPITAL_PER_TRADE = 100_000


def load(con):
    trades = con.execute("""
        SELECT * FROM bt_clear_breakouts WHERE NOT stop_above_entry ORDER BY breakout_date
    """).fetchdf()
    bars = con.execute("""
        SELECT symbol, date, high, low, close FROM daily_ohlc
        WHERE symbol IN (SELECT DISTINCT symbol FROM bt_clear_breakouts)
          AND NOT (volume = 0 AND open = high AND high = low AND low = close)
        ORDER BY symbol, date
    """).fetchdf()
    bars["date"] = pd.to_datetime(bars["date"])
    return trades, {s: g.reset_index(drop=True) for s, g in bars.groupby("symbol")}


def hold_with_stop(trades, by_symbol):
    """A: enter on retest, exit only on a close below the breakout day low."""
    rows = []
    for _, t in trades.iterrows():
        g = by_symbol.get(t["symbol"])
        if g is None:
            continue
        bo = pd.Timestamp(t["breakout_date"])
        after = g[g["date"] > bo]
        fills = after[(after["date"] <= bo + pd.Timedelta(days=90)) &
                      (after["low"] <= t["entry_price"])]
        if fills.empty:
            continue
        entry_date = fills.iloc[0]["date"]
        held = after[after["date"] >= entry_date].head(MAX_HOLD_DAYS)
        if held.empty:
            continue
        e = t["entry_price"]

        stopped = held[held["close"] < t["stop_price"]]
        if not stopped.empty:
            exit_row = stopped.iloc[0]
            outcome, exit_px, exit_dt = "stopped", exit_row["close"], exit_row["date"]
            run = held[held["date"] <= exit_dt]
        else:
            outcome, exit_px, exit_dt = "still_held", held.iloc[-1]["close"], held.iloc[-1]["date"]
            run = held

        rows.append({
            "symbol": t["symbol"], "month": pd.Timestamp(t["breakout_month"]).strftime("%Y-%m"),
            "entry_date": entry_date, "outcome": outcome,
            "return_pct": (exit_px - e) / e * 100,
            "max_gain_pct": (run["high"].max() - e) / e * 100,
            "days_held": (pd.Timestamp(exit_dt) - pd.Timestamp(entry_date)).days,
            "bars_forward": len(after),
        })
    return pd.DataFrame(rows)


def missed(trades, by_symbol):
    """B: signals whose retest never came -- what did they go on to do?"""
    rows = []
    for _, t in trades.iterrows():
        g = by_symbol.get(t["symbol"])
        if g is None:
            continue
        bo = pd.Timestamp(t["breakout_date"])
        after = g[g["date"] > bo]
        win = after[after["date"] <= bo + pd.Timedelta(days=90)]
        if win.empty or (win["low"] <= t["entry_price"]).any():
            continue  # it did retest -> not a missed trade
        fwd = after.head(MAX_HOLD_DAYS)
        ref = t["bo_close"]  # you'd have had to pay at least the breakout close
        rows.append({
            "symbol": t["symbol"], "month": pd.Timestamp(t["breakout_month"]).strftime("%Y-%m"),
            "gain_from_bo_close_pct": (fwd["high"].max() - ref) / ref * 100,
            "return_to_date_pct": (fwd.iloc[-1]["close"] - ref) / ref * 100,
            "bars_forward": len(fwd),
        })
    return pd.DataFrame(rows)


def buy_the_close(trades, by_symbol, target_pct):
    """C: skip the retest, buy the breakout close, same stop."""
    rows = []
    for _, t in trades.iterrows():
        g = by_symbol.get(t["symbol"])
        if g is None:
            continue
        bo = pd.Timestamp(t["breakout_date"])
        held = g[g["date"] > bo].head(MAX_HOLD_DAYS)
        if held.empty:
            continue
        e = t["bo_close"]
        target = e * (1 + target_pct / 100)
        outcome, exit_px = "open", held.iloc[-1]["close"]
        for _, b in held.iterrows():
            if b["close"] < t["stop_price"]:
                outcome, exit_px = "stop", b["close"]
                break
            if b["high"] >= target:
                outcome, exit_px = "target", target
                break
        rows.append({"symbol": t["symbol"], "outcome": outcome,
                     "return_pct": (exit_px - e) / e * 100})
    return pd.DataFrame(rows)


def concurrency(res):
    """D: peak simultaneous open positions -> real capital needed."""
    ev = []
    for _, r in res.iterrows():
        ev.append((pd.Timestamp(r["entry_date"]), 1))
        ev.append((pd.Timestamp(r["entry_date"]) + pd.Timedelta(days=int(r["days_held"])), -1))
    ev.sort()
    cur = peak = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    trades, by_symbol = load(con)
    con.close()

    h = hold_with_stop(trades, by_symbol)
    print("=" * 70)
    print("  A.  HOLD WITH STOP  (enter on retest, no target, exit only on stop)")
    print("=" * 70)
    print(f"  trades: {len(h)}")
    print(f"    stopped out : {(h.outcome=='stopped').sum():>3}  ({100*(h.outcome=='stopped').mean():.0f}%)")
    print(f"    still open  : {(h.outcome=='still_held').sum():>3}  ({100*(h.outcome=='still_held').mean():.0f}%)")
    print(f"\n  return per trade: mean {h.return_pct.mean():+.1f}%   median {h.return_pct.median():+.1f}%")
    print(f"  best {h.return_pct.max():+.0f}%   worst {h.return_pct.min():+.0f}%")
    print(f"  total P&L at Rs {CAPITAL_PER_TRADE:,}/trade: Rs {(h.return_pct/100*CAPITAL_PER_TRADE).sum():,.0f}")
    print(f"\n  peak unrealised gain reached (max_gain) : median {h.max_gain_pct.median():.1f}%")
    for thr in (10, 25, 50, 100):
        n = (h.max_gain_pct >= thr).sum()
        print(f"    ever traded +{thr}% above entry : {n:>3} of {len(h)}  ({100*n/len(h):.0f}%)")
    print(f"\n  how much of the peak was given back by holding to the stop:")
    st = h[h.outcome == "stopped"]
    if len(st):
        print(f"    stopped trades: peak was +{st.max_gain_pct.median():.1f}% (median), "
              f"exited at {st.return_pct.median():+.1f}%")
    print("\n  by cohort age (older cohorts have more forward data):")
    h["age"] = pd.cut(h.bars_forward, [0, 60, 120, 200, 1000],
                      labels=["<3m fwd", "3-6m fwd", "6-10m fwd", ">10m fwd"])
    print(h.groupby("age", observed=True).agg(
        trades=("return_pct", "size"), med_return=("return_pct", "median"),
        med_peak=("max_gain_pct", "median")).round(1).to_string())

    m = missed(trades, by_symbol)
    print("\n" + "=" * 70)
    print("  B.  THE SIGNALS THE RETEST ENTRY MISSED")
    print("=" * 70)
    print(f"  never retested: {len(m)} signals")
    print(f"  had you paid the breakout CLOSE instead, peak gain available:")
    print(f"    median {m.gain_from_bo_close_pct.median():+.1f}%   mean {m.gain_from_bo_close_pct.mean():+.1f}%")
    for thr in (5, 10, 25, 50):
        n = (m.gain_from_bo_close_pct >= thr).sum()
        print(f"    reached +{thr}% : {n:>3} of {len(m)}  ({100*n/len(m):.0f}%)")

    print("\n" + "=" * 70)
    print("  C.  BUY THE BREAKOUT CLOSE INSTEAD OF WAITING FOR A RETEST")
    print("=" * 70)
    for tp in (5, 10):
        c = buy_the_close(trades, by_symbol, tp)
        won = (c.outcome == "target").sum()
        lost = (c.outcome == "stop").sum()
        pnl = (c.return_pct / 100 * CAPITAL_PER_TRADE).sum()
        print(f"  target {tp:>2}%: {len(c)} trades (100% filled)   "
              f"target {won} ({100*won/len(c):.0f}%)  stop {lost} ({100*lost/len(c):.0f}%)  "
              f"P&L Rs {pnl:,.0f}")

    print("\n" + "=" * 70)
    print("  D.  CAPITAL ACTUALLY REQUIRED")
    print("=" * 70)
    for tp in (5, 10):
        try:
            r = pd.read_pickle(f"/tmp/bt_result_{tp}.pkl")
            r = r[r.outcome.isin(["target", "stop", "stop_ambiguous", "open"])].dropna(subset=["days_held"])
            peak = concurrency(r)
            print(f"  target {tp:>2}%: peak {peak} positions open at once "
                  f"-> Rs {peak*CAPITAL_PER_TRADE:,} of capital, not Rs {len(r)*CAPITAL_PER_TRADE:,}")
        except FileNotFoundError:
            print(f"  target {tp}%: run bt_run_strategy.py first")


if __name__ == "__main__":
    main()
