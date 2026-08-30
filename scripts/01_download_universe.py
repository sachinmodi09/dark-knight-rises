"""
01_download_universe.py  --  one-time / occasional bootstrap.

Pulls your repo, builds a top-750-by-market-cap universe, and scrapes
Screener.in fundamentals (annual + quarterly, including the Net Profit
sub-rows) for every one of them.

RUN THIS IN GOOGLE COLAB OR ON YOUR OWN MACHINE -- not in a Claude session
and not in GitHub Actions. It needs open internet to reach screener.in.

    # ---- Colab cell 1: get the repo -------------------------------------
    !git clone --depth 1 https://github.com/sachinmodi09/dark-knight-rises.git
    %cd dark-knight-rises
    !ls -la data/            # you should see stocks.csv and market.db

    # ---- Colab cell 2: install + run ------------------------------------
    !pip install -q requests beautifulsoup4 lxml pandas
    !python scripts/01_download_universe.py --build-universe --top 750
    !python scripts/01_download_universe.py --selftest
    !python scripts/01_download_universe.py --scrape --limit 5   # check 5 first
    !python scripts/01_download_universe.py --scrape             # then all 750

    # ---- Colab cell 3: keep the output ----------------------------------
    from google.colab import files
    files.download("data/universe_750.csv")
    files.download("data/fundamentals.csv")

If you are NOT in the repo directory, point at stocks.csv explicitly:
    !python scripts/01_download_universe.py --build-universe --stocks path/to/stocks.csv

OUTPUTS
  data/universe_750.csv   symbol, market_cap, sector, rank_mcap
  data/fundamentals.csv   symbol, basis, statement, period,
                     sales, sales_yoy_pct, opm_pct, net_profit, eps,
                     profit_excl_excep, profit_yoy_pct, profit_basis
                     ...the two _yoy_pct columns are year-on-year growth:
                     annual vs the prior year, quarterly vs the SAME quarter
                     a year earlier. Blank when the prior period is missing
                     or the prior value was a loss. profit_basis names which
                     profit line the growth was measured on -- Screener's
                     it was measured on -- normally profit excl exceptional
                     items, falling back to net profit for companies that
                     report no exceptional-items line at all.
  screener_cache/    raw HTML for every page fetched -- your audit trail.
                     Re-runs read from here, so a second run costs 0 requests.

NOTE ON THE DATA
  Annual figures go back to Mar 2015. Quarterly only covers the last ~13
  quarters -- Screener keeps a rolling window, so quarterly cannot support a
  long backtest. Use annual for anything historical, and remember an annual
  result is not public until ~75 days after the year ends.

BEFORE RUNNING AT SCALE
  Check https://www.screener.in/robots.txt and Screener's terms. There is a
  1.5s delay between requests; leave it alone.
"""

import argparse, os, re, sys, time
import pandas as pd

CACHE_DIR, DELAY_SECONDS, TIMEOUT = "screener_cache", 1.5, 30
# all paths are relative to the REPO ROOT -- run from there:
#     python scripts/01_download_universe.py --build-universe
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; personal-research-script/1.0)",
           "Accept-Language": "en-US,en;q=0.9", "X-Requested-With": "XMLHttpRequest"}

FIELDS = {
    "sales":             ["sales", "revenue"],
    "opm_pct":           ["opm %", "financing margin %"],
    "net_profit":        ["net profit"],
    "eps":               ["eps in rs"],
    "profit_excl_excep": ["profit excl excep", "profit excl. excep",
                          "profit excluding exceptional items"],
}
# Dropped on request: expenses, operating_profit, other_income, interest,
# depreciation, pbt, tax_pct, dividend_payout_pct, profit_from_associates,
# minority_share, exceptional_items, profit_for_pe, profit_for_eps,
# profit_growth_pct. To bring any back, add its row-label alias here --
# the parser already reads every row on the page, it just does not write
# them out. exceptional_items is recoverable anyway:
#     exceptional_items = net_profit - profit_excl_excep

# read by hand off the live pages 2026-08-28; if these fail, the parser is broken
SELFTEST = {
    "CRAFTSMAN": {"Mar 2026": {"sales": 8069, "net_profit": 384},
                  "Mar 2025": {"sales": 5690, "net_profit": 201}},
    "HFCL":      {"Mar 2026": {"net_profit": 329}, "Mar 2024": {"net_profit": 338}},
    "HDFCBANK":  {"Mar 2026": {"profit_excl_excep": 78758}},
}


def _num(t):
    t = (t or "").strip().replace(",", "").replace("−", "-").replace("%", "").strip()
    if not t or t in {"-", "--"}:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _clean(raw):
    return re.sub(r"\s+", " ", raw.replace("+", "").replace("−", "").rstrip("- ").strip()).lower()


def fetch_html(symbol, session, use_cache=True):
    import requests
    os.makedirs(CACHE_DIR, exist_ok=True)
    for basis in ("consolidated", "standalone"):
        path = os.path.join(CACHE_DIR, f"{symbol}__{basis}.html")
        if use_cache and os.path.exists(path):
            html = open(path, encoding="utf-8").read()
        else:
            url = (f"https://www.screener.in/company/{symbol}/consolidated/"
                   if basis == "consolidated" else
                   f"https://www.screener.in/company/{symbol}/")
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
            time.sleep(DELAY_SECONDS)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            html = r.text
            open(path, "w", encoding="utf-8").write(html)
        if 'id="profit-loss"' in html:
            return html, basis
    return None, None


def _company_id(html):
    for pat in (r'data-company-id="(\d+)"', r'/api/company/(\d+)/',
                r'company_id["\']?\s*[:=]\s*["\']?(\d+)'):
        m = re.search(pat, html or "")
        if m:
            return m.group(1)
    return None


def _table(soup, sec_id, heading):
    sec = soup.find("section", id=sec_id)
    if sec is None:
        for s in soup.find_all("section"):
            h = s.find(["h2", "h3"])
            if h and heading.lower() in h.get_text(strip=True).lower():
                sec = s
                break
    return sec.find("table", class_=re.compile("data-table")) if sec else None


def _parse(tbl):
    if tbl is None:
        return {}, []
    periods = [c.get_text(strip=True) for c in tbl.find("thead").find_all("th")[1:]]
    out, labels = {p: {} for p in periods}, []
    for tr in tbl.find_all("tr"):            # every row, hidden or not
        cells = tr.find_all("td")
        if not cells:
            continue
        lab = _clean(cells[0].get_text(" ", strip=True))
        if not lab:
            continue
        labels.append(lab)
        for p, cell in zip(periods, cells[1:]):
            out[p][lab] = _num(cell.get_text(strip=True))
    return out, labels


def _schedule(session, cid, consolidated, section="profit-loss"):
    """Net Profit sub-rows, if they are not already in the page HTML.

    section: "profit-loss" for the annual table, "quarters" for quarterly.
    """
    if not cid:
        return {}
    base = f"https://www.screener.in/api/company/{cid}/schedules/"
    for params in ({"parent": "Net Profit", "section": section,
                    "consolidated": "true" if consolidated else ""},
                   {"parent": "Net Profit", "section": section}):
        try:
            r = session.get(base, params=params, headers=HEADERS, timeout=TIMEOUT)
            time.sleep(0.5)
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue
        flat = {}
        for label, series in (data.items() if isinstance(data, dict) else []):
            if not isinstance(series, dict):
                continue
            k = _clean(str(label))
            for per, val in series.items():
                flat.setdefault(per, {})[k] = _num(str(val))
        if flat:
            return flat
    return {}


def get_one(symbol, session, use_cache=True):
    from bs4 import BeautifulSoup
    html, basis = fetch_html(symbol, session, use_cache)
    if html is None:
        return [], [], "no page"
    soup = BeautifulSoup(html, "lxml")
    ann, al = _parse(_table(soup, "profit-loss", "Profit & Loss"))
    qtr, ql = _parse(_table(soup, "quarters", "Quarterly Results"))

    # The excl-exceptional line is a collapsed sub-row under Net Profit. It is
    # usually already in the page HTML (_parse walks every <tr>, hidden or not)
    # in BOTH tables. When it is not, Screener serves it from the schedules
    # API -- and that has to be asked for per section: "profit-loss" for the
    # annual table, "quarters" for the quarterly one. Asking only for the
    # annual section leaves the quarters short.
    src, cid, con = "html", _company_id(html), basis == "consolidated"
    have = lambda labels: any(n in labels for n in
                              ("profit excl excep", "exceptional items at"))
    filled = []
    for tbl, labels, section, tag in ((ann, al, "profit-loss", "annual"),
                                      (qtr, ql, "quarters",    "quarterly")):
        if have(labels):
            continue
        extra = _schedule(session, cid, con, section)
        if extra:
            for per, vals in extra.items():
                tbl.setdefault(per, {}).update(vals)
            labels += sorted({k for v in extra.values() for k in v})
            filled.append(tag)
    if filled:
        src = "html + schedule API (" + "+".join(filled) + ")"
    elif not (have(al) and have(ql)):
        src = "html only"

    def rows(tbl, stmt):
        out = []
        for period, vals in tbl.items():
            if period.lower() in {"", "raw pdf"}:
                continue
            rec = dict(symbol=symbol, basis=basis, statement=stmt, period=period)
            for col, aliases in FIELDS.items():
                rec[col] = next((vals[a] for a in aliases if a in vals), None)
            out.append(rec)
        return out
    return add_growth(rows(ann, "annual") + rows(qtr, "quarterly")), al, src



# ------------------------------------------------------------------ growth
# Year-on-year growth, for the two lines you asked for: sales, and profit
# excluding exceptional items.
#
#   annual    Mar 2026 vs Mar 2025
#   quarterly Jun 2026 vs Jun 2025   (same quarter a year earlier)
#
# Year-on-year, not sequential, for the quarters. Sequential would compare
# Jun against Mar and read seasonality as growth -- most Indian companies
# post a big Q4 and a soft Q1, so QoQ shows a "collapse" every June that
# means nothing. YoY compares like with like.
#
# The comparison is keyed on the actual period, not on row position: if
# Screener is missing a year, that row's growth is left blank rather than
# silently comparing across a two-year gap.
#
# Blank (not zero, not a number) when:
#   the prior period is missing · the prior value is missing
#   the prior value is <= 0  -- growth off a loss is not a percentage.
#     Going from -50 to +200 is not "+500%"; it is a turnaround, and any
#     number you print there is misleading. Read the absolute columns.
# TTM never gets a growth number: it overlaps the latest full year.

_MON = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# Which profit line the growth is measured on, in order of preference.
# Both the annual and the quarterly table carry "Profit excl Excep" as a
# collapsed sub-row under Net Profit, so that is used wherever it is present.
# Some companies have no exceptional items at all and so no such row; for them
# the growth falls back to net profit. profit_basis records which line was
# actually used, per row. The basis is chosen per comparison: both periods must
# carry the same line, or the pair is skipped. Growth is never measured across
# two different profit definitions.
PROFIT_COLS = ["profit_excl_excep", "net_profit"]

COLUMNS = ["symbol", "basis", "statement", "period",
           "sales", "sales_yoy_pct", "opm_pct", "net_profit", "eps",
           "profit_excl_excep", "profit_yoy_pct", "profit_basis"]


def _ok(v):
    return v is not None and v == v          # False for None and for NaN


def _pkey(period):
    """'Mar 2026' -> 24315 (months since year 0). None for TTM / junk."""
    p = (period or "").strip().split()
    if len(p) == 2 and p[0][:3].lower() in _MON and p[1].isdigit():
        return int(p[1]) * 12 + _MON[p[0][:3].lower()]
    return None


def add_growth(rows):
    """Fill the yoy columns in place and return rows in COLUMNS order."""
    for r in rows:
        for out in ("sales_yoy_pct", "profit_yoy_pct", "profit_basis"):
            r.setdefault(out, None)
    index = {}
    for r in rows:
        k = _pkey(r["period"])
        if k is not None:
            index[(r["symbol"], r["basis"], r["statement"], k)] = r
    for r in rows:
        k = _pkey(r["period"])
        if k is None:                      # TTM
            continue
        prev = index.get((r["symbol"], r["basis"], r["statement"], k - 12))
        if prev is None:
            continue

        a, b = r.get("sales"), prev.get("sales")
        if _ok(a) and _ok(b) and b > 0:
            r["sales_yoy_pct"] = round((a / b - 1) * 100, 1)

        # pick the profit line FIRST -- both periods must carry it -- and only
        # then decide whether a percentage is meaningful. Never fall through to
        # net profit just because the excl-excep base was a loss: that would
        # quietly change what the number means.
        for col in PROFIT_COLS:
            a, b = r.get(col), prev.get(col)
            if not (_ok(a) and _ok(b)):
                continue
            r["profit_basis"] = col
            if b > 0:
                r["profit_yoy_pct"] = round((a / b - 1) * 100, 1)
            break
    return [{c: r.get(c) for c in COLUMNS} for r in rows]


# ---------------------------------------------------------------- commands
def build_universe(stocks_path, top, out):
    df = pd.read_csv(stocks_path)
    df = df.dropna(subset=["symbol", "market_cap"]).drop_duplicates("symbol")
    df = df.sort_values("market_cap", ascending=False).head(top).reset_index(drop=True)
    df["rank_mcap"] = df.index + 1
    df.to_csv(out, index=False)
    print(f"{len(df)} symbols -> {out}")
    print(f"  largest  : {df.symbol.iloc[0]:<12} {df.market_cap.iloc[0]:>12,.0f} cr")
    print(f"  smallest : {df.symbol.iloc[-1]:<12} {df.market_cap.iloc[-1]:>12,.0f} cr")
    return df



def growth_selftest():
    """Deterministic check of the growth maths. No network, always runs."""
    t = [
        dict(symbol="T", basis="c", statement="annual", period="Mar 2024", sales=100, profit_excl_excep=-50),
        dict(symbol="T", basis="c", statement="annual", period="Mar 2025", sales=150, profit_excl_excep=20),
        dict(symbol="T", basis="c", statement="annual", period="Mar 2026", sales=120, profit_excl_excep=10),
        dict(symbol="T", basis="c", statement="annual", period="TTM",      sales=130, profit_excl_excep=15),
        dict(symbol="G", basis="c", statement="annual", period="Mar 2022", sales=100, profit_excl_excep=10),
        dict(symbol="G", basis="c", statement="annual", period="Mar 2024", sales=200, profit_excl_excep=20),
        dict(symbol="Q", basis="c", statement="quarterly", period="Jun 2026", sales=110, profit_excl_excep=11),
        dict(symbol="Q", basis="c", statement="quarterly", period="Mar 2026", sales=300, profit_excl_excep=30),
        dict(symbol="Q", basis="c", statement="quarterly", period="Jun 2025", sales=100, profit_excl_excep=10),
        dict(symbol="Q", basis="c", statement="quarterly", period="Mar 2025", sales=None, profit_excl_excep=None),
        # quarterly tables carry no excl-excep line -> must fall back to net profit
        dict(symbol="N", basis="c", statement="quarterly", period="Jun 2025", sales=100, net_profit=10),
        dict(symbol="N", basis="c", statement="quarterly", period="Jun 2026", sales=130, net_profit=14),
        # excl-excep present in both, but the prior year was a loss -> blank,
        # and it must NOT silently fall back to the net-profit line
        dict(symbol="L", basis="c", statement="annual", period="Mar 2025", sales=100,
             profit_excl_excep=-5, net_profit=20),
        dict(symbol="L", basis="c", statement="annual", period="Mar 2026", sales=120,
             profit_excl_excep=8, net_profit=30),
    ]
    want = {
        ("T", "Mar 2024"): (None, None),      # first year, nothing to compare
        ("T", "Mar 2025"): (50.0, None),      # prior profit was a LOSS -> blank, not +140%
        ("T", "Mar 2026"): (-20.0, -50.0),    # sales can fall; growth goes negative
        ("T", "TTM"):      (None, None),      # TTM overlaps the last year -> never
        ("G", "Mar 2022"): (None, None),
        ("G", "Mar 2024"): (None, None),      # Mar 2023 missing -> blank, not a 2-year jump
        ("Q", "Jun 2026"): (10.0, 10.0),      # vs Jun 2025, NOT vs Mar 2026
        ("Q", "Mar 2026"): (None, None),      # Mar 2025 row exists but is empty
        ("Q", "Jun 2025"): (None, None),
        ("N", "Jun 2026"): (30.0, 40.0, "net_profit"),          # fallback used
        ("L", "Mar 2026"): (20.0, None, "profit_excl_excep"),   # loss base, no fallback
    }
    got = {(r["symbol"], r["period"]): ((r["sales_yoy_pct"], r["profit_yoy_pct"], r["profit_basis"])
                                        if r["symbol"] in ("N", "L") else
                                        (r["sales_yoy_pct"], r["profit_yoy_pct"]))
           for r in add_growth([dict(x) for x in t])}
    ok = True
    print("Growth self-test (no network):")
    for k in sorted(want):
        good = got[k] == want[k]
        ok &= good
        print(f"  {k[0]:<3}{k[1]:<10} want {str(want[k]):<34} got {str(got[k]):<34}"
              f"{'ok' if good else 'FAIL'}")
    print("  " + ("PASS\n" if ok else "FAIL - growth maths is wrong, do not trust the columns\n"))
    return ok


def selftest():
    import requests
    ok = growth_selftest()
    s = requests.Session()
    print("Parser self-test against values read by hand from screener.in on 2026-08-28.\n")
    for sym, expect in SELFTEST.items():
        rows, _, src = get_one(sym, s)
        idx = {r["period"]: r for r in rows if r["statement"] == "annual"}
        for period, fields in expect.items():
            for field, want in fields.items():
                got = idx.get(period, {}).get(field)
                good = got == want
                ok &= good
                print(f"  {sym:<11}{period:<10}{field:<19}want {want:>9}  got {str(got):>9}"
                      f"  {'ok' if good else 'FAIL'}   [{src}]")
    print("\n" + ("PASS - parser reproduces every hand-checked value."
                  if ok else "FAIL - do NOT use the output. Screener's markup changed."))
    return ok


def scrape(universe_csv, out, limit=None):
    import requests
    syms = pd.read_csv(universe_csv).symbol.dropna().tolist()
    if limit:
        syms = syms[:limit]
    s, rows, failed = requests.Session(), [], []
    t0 = time.time()
    for n, sym in enumerate(syms, 1):
        try:
            r, _, src = get_one(sym, s)
        except Exception as e:
            print(f"  [{n}/{len(syms)}] {sym:<12} ERROR {e}")
            failed.append(sym)
            continue
        if not r:
            failed.append(sym)
            continue
        rows += r
        if n <= 5 or n % 25 == 0 or n == len(syms):
            na = sum(x["statement"] == "annual" for x in r)
            el = time.time() - t0
            eta = el / n * (len(syms) - n) / 60
            print(f"  [{n}/{len(syms)}] {sym:<12} {na}a+{len(r)-na}q  [{src}]  eta {eta:.0f}m")
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df):,} rows / {df.symbol.nunique()} symbols -> {out}")
    if failed:
        print(f"{len(failed)} failed (NSE symbol != Screener slug, or delisted):")
        print("  " + ", ".join(failed[:40]) + (" ..." if len(failed) > 40 else ""))
        pd.Series(failed, name="symbol").to_csv("data/fundamentals_failed.csv", index=False)
        print("  full list -> data/fundamentals_failed.csv")
    miss = [c for c in ("sales", "opm_pct", "net_profit", "eps", "profit_excl_excep")
            if c in df and df[df.statement == "annual"][c].isna().all()]
    if miss:
        print(f"!! never captured for any company: {miss} -- check the page layout")

    print("\ncoverage of the excl-exceptional profit line:")
    for stmt in ("annual", "quarterly"):
        s = df[df.statement == stmt]
        if s.empty:
            continue
        n = s.symbol.nunique()
        got = s[s.profit_excl_excep.notna()].symbol.nunique()
        print(f"  {stmt:<10} {got:>4}/{n} companies have it  "
              f"({s.profit_excl_excep.notna().mean()*100:.0f}% of rows)")
    if "profit_basis" in df:
        for stmt in ("annual", "quarterly"):
            s = df[(df.statement == stmt) & df.profit_basis.notna()]
            if len(s):
                vc = s.profit_basis.value_counts()
                print(f"  {stmt:<10} growth measured on: "
                      + ", ".join(f"{k} {v}" for k, v in vc.items()))
    print("  If quarterly coverage is near zero the schedules API call for the")
    print("  'quarters' section is failing -- send me this output.")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-universe", action="store_true")
    ap.add_argument("--stocks", default="data/stocks.csv")
    ap.add_argument("--top", type=int, default=750)
    ap.add_argument("--universe", default="data/universe_750.csv")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scrape", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="data/fundamentals.csv")
    a, _ = ap.parse_known_args()          # ignore Colab's own -f kernel.json

    did = False
    if a.build_universe:
        if not os.path.exists(a.stocks):
            print(f"{a.stocks} not found. Clone the repo first:\n"
                  "  git clone --depth 1 https://github.com/sachinmodi09/dark-knight-rises.git")
            return 1
        build_universe(a.stocks, a.top, a.universe); did = True
    if a.selftest:
        did = True
        if not selftest():
            return 1
    if a.scrape:
        if not os.path.exists(a.universe):
            print(f"{a.universe} not found -- run --build-universe first.")
            return 1
        scrape(a.universe, a.out, a.limit); did = True
    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
