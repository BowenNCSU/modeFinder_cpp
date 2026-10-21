"""A pool of every strategy variant tried so far, to run on any ticker.

A variant is fit/buy/sell/rule/filter, for example
``kernel_vol/q25/q99/timall/traj_guard``:

  fit      kernel | kernel_vol (bar volumes as weights) | bern (Bernstein,
           support [lowest - 1/1560, highest + 1/1560] of each window)
  buy      a fitted quantile (q01 ... q99, see PROBS) or mode: buy when the
           close is below it
  sell     a fitted quantile, mode, or max (the highest value of the 1560
           bars before t): sell when the close is above it
  rule     plain | tim20 | timall | either (entry gates on the plain
           strategy's time in market) | cap | stop | capstop | capstop_rebuy
           (rules on the strategy's own path)
  filter   none | band_guard | band_literal | traj_guard | traj_literal
           (from the Bernstein mode of the close, see improvements.py)

Every variant uses a 1560-bar window, runs on the high, open and close on
their own and averages the three, decides at a bar's close and fills at the
next open, as in simulate.py.  POOL lists the 312 variants of the earlier
runs (see GROUPS), and ``strategy_pool.csv`` holds them with their results
on round 3's 2,000 tickers.  Any other combination of the values above runs
as well.

    python strategy_pool.py --list --limit 20          # best of the pool
    python strategy_pool.py VOO AMZN                   # TOP3 on two tickers
    python strategy_pool.py META --variants all --out meta.csv
    python strategy_pool.py SPY --variants 'kernel/q25/*/timall/none'
    python strategy_pool.py --write-pool               # rebuild the CSV

From Python:

    import strategy_pool as sp
    rows = sp.run_tickers(["VOO"], sp.TOP3)

Fits are cached in ``cache/pool`` here (ignored by git), keyed by the data
file's size and modification time.  A new ticker takes under a minute for
all fits, run in parallel (the Bernstein ones are the slow part).
"""

import argparse
import csv
import fnmatch
import glob
import json
import os
import sys
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import improvements as imp  # noqa: E402
import simulate as sim  # noqa: E402
import thresholds as th  # noqa: E402

PROBS = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)
FITS = {"kernel": "kernel", "kernel_vol": "kernel, volume weights",
        "bern": "Bernstein"}
RULES = {
    "plain": "no rule",
    "tim20": "tim_20d > 85.3%, entry only",
    "timall": "tim_all < 50%, entry only",
    "either": "tim_20d > 85.3% or tim_all < 50%, entry only",
    "cap": "cap: out while held > 50% of the last 1560 bars",
    "stop": "stop: sell 6% below the buy price",
    "capstop": "cap + stop",
    "capstop_rebuy": "cap + stop, rebuy at once",
}
FILTERS = imp.FILTERS
RUN = os.path.join(HERE, "results", "run_20261019_top3_2000")
POOL_CSV = os.path.join(HERE, "strategy_pool.csv")
CACHE = os.path.join(HERE, "cache", "pool")

Variant = namedtuple("Variant", "fit buy sell rule filter")


def parse(vid):
    """Variant from 'fit/buy/sell/rule/filter', with every part checked."""
    parts = vid.split("/")
    if len(parts) != 5:
        raise ValueError(f"{vid!r}: expected fit/buy/sell/rule/filter")
    v = Variant(*parts)
    if v.fit not in FITS or v.rule not in RULES or v.filter not in FILTERS:
        raise ValueError(f"{vid!r}: fit, rule or filter not in {list(FITS)}, "
                         f"{list(RULES)}, {list(FILTERS)}")
    for side, x in (("buy", v.buy), ("sell", v.sell)):
        if not (x == "mode" or (x == "max" and side == "sell") or column(x) is not None):
            raise ValueError(f"{vid!r}: {side} {x!r} must be mode, "
                             + ("max or " if side == "sell" else "")
                             + "one of " + ", ".join(qname(p) for p in PROBS))
    return v


def vid(v):
    return "/".join(v)


def qname(p):
    return f"q{round(p * 100):02d}"


def column(x):
    """Column of x in the cached fits: a PROBS quantile, or the mode last."""
    if x == "mode":
        return len(PROBS)
    names = [qname(p) for p in PROBS]
    return names.index(x) if x in names else None


def describe(v):
    return (f"{FITS[v.fit]} · buy < {v.buy} · sell > "
            f"{'max of last 1560' if v.sell == 'max' else v.sell} · "
            f"{RULES[v.rule]} · {FILTERS[v.filter]}")


# --------------------------------------------------------------------------
# The pool

ROUND2 = ["mode/mode", "q50/q50", "q40/q60", "q25/q75", "q10/q90", "q05/q95",
          "q01/q99", "q10/q75", "q25/q90"]
STOP_RULES = ["plain", "cap", "stop", "capstop", "capstop_rebuy"]
GROUPS = {
    "round2": [f"kernel/{r}/plain/none" for r in ROUND2],
    "stop_rules": [f"kernel/q05/q95/{r}/none" for r in STOP_RULES]
                  + ["kernel/q25/q90/capstop/none"],
    "improvements": [f"{f}/q25/{s}/{r}/{x}" for f, s, r, x in imp.variants()],
}
GROUP_NAMES = {
    "round2": "round 2 rules (kernel, 1560 bars)",
    "stop_rules": "stop-rules run",
    "improvements": "improvements run",
}
POOL = list(dict.fromkeys(v for g in GROUPS.values() for v in g))
TOP3 = ["kernel_vol/q25/max/timall/traj_guard",
        "kernel_vol/q25/q99/timall/traj_guard",
        "kernel_vol/q25/q99/timall/band_literal"]


def select(specs):
    """Variant ids from 'top3', 'all', a group name, an id or a glob pattern."""
    out = []
    for s in specs:
        if s == "top3":
            out += TOP3
        elif s == "all":
            out += POOL
        elif s in GROUPS:
            out += GROUPS[s]
        elif any(ch in s for ch in "*?["):
            hits = fnmatch.filter(POOL, s)
            if not hits:
                raise ValueError(f"{s!r} matches no variant of the pool")
            out += hits
        else:
            out.append(vid(parse(s)))
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------
# Fits

def fit_rolling(values, source, vol=None):
    """f[t] = the PROBS quantiles and the mode of values[t-1559 .. t],
    carried forward over failed fits."""
    n, W = len(values), sim.WINDOW
    q = np.full((n, len(PROBS) + 1), np.nan)
    for t in range(W - 1, n):
        fit = imp.fit_window(values[t - W + 1:t + 1], source,
                             None if vol is None else vol[t - W + 1:t + 1])
        if fit is None:
            continue
        q[t, :-1] = fit.quantile(PROBS)
        q[t, -1] = fit.mode
    return sim.ffill(q)


def file_key(path):
    st = os.stat(path)
    return f"{st.st_size}_{int(st.st_mtime)}"


def fit_path(cache, ticker, path, part):
    return os.path.join(cache, part, f"{ticker}_{file_key(path)}.npy")


def fit_job(job):
    """One cached fit: part is '<source>_<field>' or 'jack' (the close's
    leave-one-day-out Bernstein modes)."""
    ticker, path, part, cache = job
    out = fit_path(cache, ticker, path, part)
    if os.path.exists(out):
        return ticker, part, 0.0
    t0 = time.time()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bars = sim.load_bars(path)
    if part == "jack":
        ends, jack = imp.jackknife_modes(bars["close"], bars["ts"])
        res = np.column_stack((ends, jack))
    else:
        source, field = part.rsplit("_", 1)
        vol = np.nan_to_num(bars["volume"], nan=0.0) if source == "kernel_vol" else None
        res = fit_rolling(bars[field], source, vol)
    tmp = out[:-4] + ".tmp.npy"
    np.save(tmp, res)
    os.replace(tmp, out)
    return ticker, part, time.time() - t0


def parts_needed(variants):
    parts = {f"{v.fit}_{f}" for v in variants for f in sim.FIELDS}
    if any(v.filter != "none" for v in variants):
        parts |= {"bern_close", "jack"}
    return sorted(parts)


# --------------------------------------------------------------------------
# Backtests

def path_events(o, c, buy, sell, start, cap, stop, rebuy, no_buy, no_sell):
    """Events of the rules on a strategy's own path (cap, stop), with the
    filter's blocks on buys and on signal sells.  Matches
    stop_rules.rule_events and improvements.capstop_events."""
    n, w = len(c), sim.WINDOW
    cl, op, bl, sl = c.tolist(), o.tolist(), buy.tolist(), sell.tolist()
    nb, ns = no_buy.tolist(), no_sell.tolist()
    held = [False] * n
    out = [0] * n
    pos, in_window, entry, lock = False, 0, 0.0, 0.0
    for t in range(start, n - 1):
        held[t] = pos
        in_window += pos
        if t >= w:
            in_window -= held[t - w]
        over = cap and t - start + 1 >= w and in_window > imp.TIM_MAX * w
        x = cl[t]
        if pos:
            if x > sl[t] and not ns[t]:
                pos = False
            elif over:
                pos = False
            elif stop and x < entry * (1 - imp.STOP):
                pos = False
                if not rebuy:
                    lock = entry * (1 - imp.STOP)
        else:
            if lock and x >= lock:
                lock = 0.0
            if x < bl[t] and not over and not lock and not nb[t]:
                pos = True
                entry = op[t + 1]
        out[t] = 1 if pos else -1
    return np.array(out, np.int8)


def backtest_job(job):
    """Every requested variant on one ticker: per-field and mean returns,
    second-half return, time in market and trades, plus buy and hold."""
    ticker, path, ids, cache, cost, mid_day = job
    variants = [parse(i) for i in ids]
    bars = sim.load_bars(path)
    ts, o, c = bars["ts"], bars["open"], bars["close"]
    n, start = len(c), sim.WINDOW - 1
    fits = {p: np.load(fit_path(cache, ticker, path, p)) for p in parts_needed(variants)}
    masks = None
    if "jack" in fits:
        masks = imp.trend_masks(fits["bern_close"][:, -1], fits["jack"])
    pmax = {f: imp.prev_max(bars[f]) for f in sim.FIELDS}
    z = np.zeros(n, bool)
    gates = {}

    def lines(v, field):
        q = fits[f"{v.fit}_{field}"]
        buy = q[:, column(v.buy)]
        sell = pmax[field] if v.sell == "max" else q[:, column(v.sell)]
        return buy, sell

    def gate(v, field):
        key = (v.fit, v.buy, v.sell, field)
        if key not in gates:
            buy, sell = lines(v, field)
            ev0 = imp.gate_events(c, buy, sell, start, ~z, z, z)
            eq0, expo0, _, _ = sim.equity_from_events(o, c, ev0, cost)
            f = th.combo_features(bars, np.column_stack((buy, sell)), start, expo0, eq0)
            with np.errstate(invalid="ignore"):
                g = {"tim20": f["tim_20d"] > imp.TIM20_MIN,
                     "timall": f["tim_all"] < imp.TIMALL_MAX}
            g["either"] = g["tim20"] | g["timall"]
            gates[key] = g
        return gates[key]

    eq, _, _, _ = sim.buy_and_hold(bars, start, cost)
    days, bh = sim.daily_equity(ts, eq, start)
    mid = int(np.searchsorted(days, mid_day)) if mid_day else len(days) // 2
    mid = min(mid, len(days) - 1)
    rows = [{"ticker": ticker, "id": "buy_and_hold", "description": "buy and hold",
             "return": eq[-1] / sim.CAPITAL - 1, "second_half_return": bh[-1] / bh[mid] - 1,
             "time_in_market": 1.0, "trades": 1.0,
             **{f"return_{f}": eq[-1] / sim.CAPITAL - 1 for f in sim.FIELDS}}]
    for v in variants:
        nb, ns = (z, z) if v.filter == "none" else imp.blocks(v.filter, masks, n)
        per, curves = {}, []
        for field in sim.FIELDS:
            buy, sell = lines(v, field)
            if v.rule in ("plain", "tim20", "timall", "either"):
                g = ~z if v.rule == "plain" else gate(v, field)[v.rule]
                ev = imp.gate_events(c, buy, sell, start, g, nb, ns)
            else:
                ev = path_events(o, c, buy, sell, start, cap="cap" in v.rule,
                                 stop="stop" in v.rule, rebuy=v.rule.endswith("rebuy"),
                                 no_buy=nb, no_sell=ns)
            eq, expo, trades, _ = sim.equity_from_events(o, c, ev, cost)
            per[field] = (eq[-1] / sim.CAPITAL - 1, expo[start:].mean(), trades[start:].sum())
            curves.append(sim.daily_equity(ts, eq, start)[1])
        curve = np.mean(curves, axis=0)
        mean = np.mean(list(per.values()), axis=0)
        row = {"ticker": ticker, "id": vid(v), "description": describe(v),
               "return": mean[0], "second_half_return": curve[-1] / curve[mid] - 1,
               "time_in_market": mean[1], "trades": mean[2]}
        row.update({f"return_{f}": per[f][0] for f in sim.FIELDS})
        rows.append(row)
    return rows


def find_file(ticker, data):
    if os.path.isfile(ticker):
        return os.path.basename(ticker).split(".")[0].upper(), ticker
    hits = sorted(glob.glob(os.path.join(data, "**", ticker.lower() + ".us.txt"),
                            recursive=True))
    if not hits:
        raise FileNotFoundError(f"no {ticker.lower()}.us.txt under {data}")
    if len(hits) > 1:
        raise ValueError(f"{ticker}: several files, pass one as a path: {hits}")
    return ticker.upper(), hits[0]


def default_data():
    try:
        return json.load(open(os.path.join(RUN, "config.json")))["data"]
    except (OSError, KeyError, ValueError):
        return None


def run_tickers(tickers, ids=TOP3, data=None, cache=CACHE, cost_bps=2.0,
                mid_day=20260731, jobs=None, log=print):
    """Rows of backtest_job for each ticker, fitting what is not cached.
    mid_day starts the second half (its close); None takes the middle day."""
    data = data or default_data()
    files = [find_file(t, data) for t in tickers]
    variants = [parse(i) for i in ids]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        fit_jobs = [(t, p, part, cache) for t, p in files for part in parts_needed(variants)]
        for ticker, part, secs in pool.map(fit_job, fit_jobs):
            if secs and log:
                log(f"  fitted {ticker} {part} ({secs:.0f} s, {time.time() - t0:.0f} s total)")
        results = pool.map(backtest_job, [(t, p, list(ids), cache, cost_bps / 1e4, mid_day)
                                          for t, p in files])
        return [row for rows in results for row in rows]


# --------------------------------------------------------------------------
# The pool file

def write_pool(path=POOL_CSV):
    """strategy_pool.csv: every POOL variant with its group, description and
    results on round 3's 2,000 tickers (blank where it was not run there)."""
    known = {}
    with open(RUN + "_improvements/portfolio.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["source"] != "-":
                known[f"{r['source']}/q25/{r['sell']}/{r['strategy']}/{r['filter']}"] = r
    names = {"q05/q95": "plain", "q05/q95 + tim": "cap", "q05/q95 + stop": "stop",
             "q05/q95 + tim + stop": "capstop",
             "q05/q95 + tim + stop, rebuy at once": "capstop_rebuy"}
    with open(RUN + "_stop_rules/portfolio.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["field"] == "mean" and r["strategy"] in names:
                known[f"kernel/q05/q95/{names[r['strategy']]}/none"] = r
    cols = ["return", "sharpe", "max_drawdown", "second_half_return", "profitable",
            "time_in_market", "trades_per_ticker"]
    rows = []
    for i in POOL:
        v = parse(i)
        r = known.get(i, {})
        rows.append({"id": i, "group": next(g for g in GROUPS if i in GROUPS[g]),
                     **v._asdict(), "description": describe(v), "top3": int(i in TOP3),
                     **{f"r3_{k}": r.get(k, "") for k in cols}})
    rows.sort(key=lambda r: -float(r["r3_return"]) if r["r3_return"] != "" else np.inf)
    th.write_csv(path, rows)
    return path


def read_pool(path=POOL_CSV):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def pct(x):
    return "" if x in ("", None) else f"{float(x):.2%}"


def print_rows(rows):
    print(f"{'ticker':7s} {'variant':42s} {'return':>8s} {'2nd half':>9s} "
          f"{'in mkt':>7s} {'trades':>7s}")
    for r in rows:
        print(f"{r['ticker']:7s} {r['id']:42s} {r['return']:8.2%} "
              f"{r['second_half_return']:9.2%} {r['time_in_market']:7.0%} {r['trades']:7.1f}")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("tickers", nargs="*", help="tickers or paths to data files")
    p.add_argument("--variants", nargs="+", default=None,
                   help="top3 (default for tickers; --list shows all), all, a group "
                        "(%s), ids or glob patterns"
                        % ", ".join(GROUPS))
    p.add_argument("--list", action="store_true",
                   help="print the pool ranked by its return on round 3's tickers")
    p.add_argument("--limit", type=int, default=None, help="rows for --list")
    p.add_argument("--write-pool", action="store_true", help="rebuild strategy_pool.csv")
    p.add_argument("--data", default=None, help="data root (default: round 3's)")
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--mid-day", type=int, default=20260731,
                   help="the second half starts at this day's close; 0 for the middle day")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--out", default=None, help="write the results to this CSV")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.write_pool:
        print(f"wrote {write_pool()} ({len(POOL)} variants)")
    if args.list:
        rows = read_pool()
        ids = set(select(args.variants)) if args.variants else None
        rows = [r for r in rows if ids is None or r["id"] in ids][:args.limit]
        print(f"{'variant':42s} {'return':>8s} {'2nd half':>9s} {'sharpe':>7s} {'group':>13s}")
        for r in rows:
            sharpe = f"{float(r['r3_sharpe']):.2f}" if r["r3_sharpe"] else ""
            print(f"{r['id']:42s} {pct(r['r3_return']):>8s} "
                  f"{pct(r['r3_second_half_return']):>9s} {sharpe:>7s} {r['group']:>13s}")
    if args.tickers:
        rows = run_tickers(args.tickers, select(args.variants or ["top3"]), args.data, args.cache,
                           args.cost_bps, args.mid_day or None, args.jobs)
        print_rows(rows)
        if args.out:
            th.write_csv(args.out, rows)
            print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
