"""
bt_run_strategy.py
Step 2 of the trading-vs-holding backtest.

Takes bt_clear_breakouts (built by bt_build_clear_sample.py) and races each
trade to a target against the stop, so the two strategies can be compared on
evidence rather than intuition.

Trade mechanics, exactly as the alert email specifies them:
  entry  = prior ATH, bought on a retest -- filled the first day the LOW
           trades down to that level after the breakout day
  stop   = breakout day low, "close basis" -- exits only on a CLOSE below it
  target = entry * (1 + target_pct), filled on an intraday HIGH touch
           (a resting limit order would have been taken)

Two deliberate modelling choices, both stated so the numbers can be read
honestly:

1. Same-bar ambiguity. Daily OHLC cannot say whether the high or the close
   came first, so a bar that touches the target AND closes below the stop is
   scored a STOP. That is the pessimistic reading; the count is reported so
   the size of the assumption is visible.

2. Zero-volume placeholder bars (market holidays -- 1.3% of daily_ohlc, all
   four prices equal) are skipped, so they neither trigger exits nor inflate
   the holding-period day counts.

Trades whose stop sits above their entry (gap-ups; stop_above_entry) are
excluded: price would have to break the stop to reach the entry, so the
strategy as specified cannot take them.
"""

import sys
from datetime import date

import duckdb
import pandas as pd

DB_PATH = "data/market.db"
RETEST_WINDOW_DAYS = 90      # give the retest this long to happen
MAX_HOLD_DAYS = 180          # then race target vs stop for this long
CAPITAL_PER_TRADE = 100_000  # Rs 1 lakh per trade


def load_bars(con):
    """All daily bars for the traded symbols, holidays removed."""
    df = con.execute("""
        SELECT d.symbol, d.date, d.high, d.low, d.close
        FROM daily_ohlc d
        WHERE d.symbol IN (SELECT DISTINCT symbol FROM bt_clear_breakouts)
          AND NOT (d.volume = 0 AND d.open = d.high AND d.high = d.low AND d.low = d.close)
        ORDER BY d.symbol, d.date
    """).fetchdf()
    # Normalise to Timestamp once here; duckdb hands back datetime64 while the
    # trade rows carry python dates, and comparing the two raises.
    df["date"] = pd.to_datetime(df["date"])
    return df


def simulate(trades, bars, target_pct, allow_same_bar_target=False):
    """allow_same_bar_target=False is the honest default. The entry fills on
    the bar's LOW and the target on its HIGH, but a daily bar cannot say which
    came first: if the high printed before the dip, the fill happened near the
    low and the target was NOT available that day. So the target is only
    checked from the bar AFTER entry. Stops are exempt -- a close is by
    definition the day's last price, so a same-bar close below the stop is
    real. Left switchable because the difference is large (41% of 5% winners
    'won' on their entry bar) and worth being able to show as an upper bound."""
    by_symbol = {s: g.reset_index(drop=True) for s, g in bars.groupby("symbol")}
    out = []
    for _, t in trades.iterrows():
        g = by_symbol.get(t["symbol"])
        rec = {"symbol": t["symbol"], "breakout_date": t["breakout_date"],
               "month": pd.Timestamp(t["breakout_month"]).strftime("%Y-%m"),
               "entry_price": t["entry_price"], "stop_price": t["stop_price"],
               "risk_pct": t["risk_pct"]}
        if g is None:
            out.append({**rec, "outcome": "no_data"})
            continue

        bo = pd.Timestamp(t["breakout_date"])
        after = g[g["date"] > bo]
        window = after[after["date"] <= bo + pd.Timedelta(days=RETEST_WINDOW_DAYS)]

        fills = window[window["low"] <= t["entry_price"]]
        if fills.empty:
            # Never came back to the entry -- no trade. Record what it did
            # anyway, because those are often the biggest movers.
            missed = after.head(MAX_HOLD_DAYS)
            rec["missed_max_gain_pct"] = (
                (missed["high"].max() - t["entry_price"]) / t["entry_price"] * 100
                if not missed.empty else None)
            out.append({**rec, "outcome": "no_fill"})
            continue

        entry_row = fills.iloc[0]
        entry_date = entry_row["date"]
        target = t["entry_price"] * (1 + target_pct / 100)

        held = after[(after["date"] >= entry_date)].head(MAX_HOLD_DAYS)
        outcome, exit_date, exit_price = "open", None, None
        for i, (_, b) in enumerate(held.iterrows()):
            on_entry_bar = (i == 0)
            hit_t = b["high"] >= target and (allow_same_bar_target or not on_entry_bar)
            hit_s = b["close"] < t["stop_price"]
            if hit_t and hit_s:
                outcome, exit_date, exit_price = "stop_ambiguous", b["date"], t["stop_price"]
                break
            if hit_t:
                outcome, exit_date, exit_price = "target", b["date"], target
                break
            if hit_s:
                outcome, exit_date, exit_price = "stop", b["date"], b["close"]
                break

        if outcome == "open" and not held.empty:
            exit_price = held.iloc[-1]["close"]
            exit_date = held.iloc[-1]["date"]

        rec.update({
            "outcome": outcome,
            "entry_date": entry_date,
            "days_to_entry": (pd.Timestamp(entry_date) - bo).days,
            "exit_date": exit_date,
            "days_held": (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days if exit_date else None,
            "return_pct": (exit_price - t["entry_price"]) / t["entry_price"] * 100 if exit_price else None,
            "max_gain_pct": (held["high"].max() - t["entry_price"]) / t["entry_price"] * 100 if not held.empty else None,
            "max_drawdown_pct": (held["low"].min() - t["entry_price"]) / t["entry_price"] * 100 if not held.empty else None,
            "bars_available": len(held),
        })
        out.append(rec)
    return pd.DataFrame(out)


def report(res, target_pct):
    n = len(res)
    filled = res[~res["outcome"].isin(["no_fill", "no_data"])]
    won = filled[filled["outcome"] == "target"]
    lost = filled[filled["outcome"].isin(["stop", "stop_ambiguous"])]
    still = filled[filled["outcome"] == "open"]

    print(f"\n{'='*66}")
    print(f"  TARGET {target_pct:.0f}%   (stop = breakout day low, close basis)")
    print(f"{'='*66}")
    print(f"  signals                    {n}")
    print(f"  filled on retest           {len(filled)}  ({100*len(filled)/n:.0f}%)")
    print(f"  never retested (no trade)  {n-len(filled)}  ({100*(n-len(filled))/n:.0f}%)")
    if not len(filled):
        return None
    print(f"\n  of the {len(filled)} trades actually taken:")
    print(f"    hit +{target_pct:.0f}% FIRST        {len(won):>3}   ({100*len(won)/len(filled):.0f}%)")
    print(f"    stopped out FIRST       {len(lost):>3}   ({100*len(lost)/len(filled):.0f}%)")
    print(f"    still open at data end  {len(still):>3}   ({100*len(still)/len(filled):.0f}%)")
    amb = int((filled["outcome"] == "stop_ambiguous").sum())
    if amb:
        print(f"    (of the stops, {amb} touched target the same bar -- scored as stops)")

    decided = pd.concat([won, lost])
    if len(decided):
        wr = 100 * len(won) / len(decided)
        print(f"\n  win rate on DECIDED trades  {wr:.0f}%  ({len(won)}/{len(decided)})")
        breakeven = 100 * filled["risk_pct"].mean() / (target_pct + filled["risk_pct"].mean())
        print(f"  breakeven win rate needed   {breakeven:.0f}%   "
              f"(avg risk {filled['risk_pct'].mean():.2f}% vs {target_pct:.0f}% reward)")
        print(f"  -> {'PROFITABLE' if wr > breakeven else 'LOSS-MAKING'} by {abs(wr-breakeven):.0f} points")

    if len(won):
        print(f"\n  days to hit target:  median {won['days_held'].median():.0f}   mean {won['days_held'].mean():.0f}")
    if len(lost):
        print(f"  days to stop out:    median {lost['days_held'].median():.0f}   mean {lost['days_held'].mean():.0f}")
    print(f"  days waiting for entry: median {filled['days_to_entry'].median():.0f}")

    # Money, at Rs 1 lakh a trade
    pnl = (filled["return_pct"] / 100 * CAPITAL_PER_TRADE)
    print(f"\n  P&L at Rs {CAPITAL_PER_TRADE:,}/trade over 12 months:")
    print(f"    total       Rs {pnl.sum():>12,.0f}   across {len(filled)} trades")
    print(f"    per trade   Rs {pnl.mean():>12,.0f}   (median Rs {pnl.median():,.0f})")
    print(f"    capital deployed Rs {len(filled)*CAPITAL_PER_TRADE:,} (if never reused)")
    return filled


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    trades = con.execute("""
        SELECT * FROM bt_clear_breakouts WHERE NOT stop_above_entry
        ORDER BY breakout_date
    """).fetchdf()
    bars = load_bars(con)
    con.close()
    print(f"tradeable CLEAR breakouts: {len(trades)}   daily bars loaded: {len(bars):,}")

    results = {}
    for tp in (5, 10):
        res = simulate(trades, bars, tp, allow_same_bar_target=False)
        report(res, tp)
        results[tp] = res
        res.to_pickle(f"/tmp/bt_result_{tp}.pkl")

        opt = simulate(trades, bars, tp, allow_same_bar_target=True)
        f_o = opt[~opt["outcome"].isin(["no_fill", "no_data"])]
        w_o = int((f_o["outcome"] == "target").sum())
        print(f"\n  [upper bound if same-bar target fills were allowed: "
              f"{w_o} wins instead of {int((results[tp]['outcome']=='target').sum())}]")
    return results


if __name__ == "__main__":
    main()
