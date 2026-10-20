"""Rolling mode / quantile trading simulation with modefinder.

For a random sample of stocks and ETFs with 5-minute bars, a rolling window
of the last W bars of one price field (open, close, high or low) is fitted
with ``modefinder.ModeFinder`` (Bernstein or kernel smoothing) at every bar:
the oldest bar leaves the window and the newest enters.  The fitted mode or
quantiles predict the distribution of the coming prices, and a long-only
strategy starting with $10,000 per ticker

    buys  (all in)  when the current close is below the buy threshold and
    sells (all out) when the current close is above the sell threshold,

with thresholds mode/mode, median/median or a lower/upper quantile pair.
Orders are decided at a bar's close and filled at the next bar's open.
Each parameter combination is compared with buy and hold over the data's
whole history.  Results and a markdown report go to --out.

    python simulations/stock_mode_quantile/simulate.py \\
        --data "/Users/bowenliu/Downloads/data/5 min" --stocks 10 --etfs 5

Requires numpy and modefinder (``pip install .`` at the repository root).
"""

import argparse
import csv
import datetime as dt
import itertools
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import modefinder

FIELDS = ("open", "close", "high", "low")
PROBS = (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99)
# rule name -> (buy threshold, sell threshold); "mode" or a probability.
# Buy below the lower quantile, sell above the upper one, hold in between.
RULES = {
    "mode": ("mode", "mode"),
    "median": (0.50, 0.50),
    "q40/q60": (0.40, 0.60),
    "q25/q75": (0.25, 0.75),
    "q10/q90": (0.10, 0.90),
    "q05/q95": (0.05, 0.95),
    "q01/q99": (0.01, 0.99),
    "q10/q75": (0.10, 0.75),     # buy deep, sell early
    "q25/q90": (0.25, 0.90),     # buy early, let it run
}
CAPITAL = 10_000.0


# --------------------------------------------------------------------------
# Data

def find_files(root):
    """All price files under root, with the asset kind guessed from the path
    (Stooq's bulk download keeps e.g. "nasdaq etfs" and "nyse stocks" apart)."""
    out = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.startswith(".") or not name.lower().endswith((".txt", ".csv")):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root).lower()
            kind = "etf" if "etf" in rel else "stock" if "stock" in rel else "unknown"
            ticker = name.rsplit(".", 1)[0]
            if ticker.lower().endswith(".us"):
                ticker = ticker[:-3]
            out.append({"path": path, "ticker": ticker.upper(), "kind": kind})
    out.sort(key=lambda f: f["path"])
    return out


def _digits(s):
    return "".join(ch for ch in s if ch.isdigit())


def load_bars(path):
    """Read a CSV of bars into arrays ts (int yyyymmddhhmmss), open, high,
    low, close, volume, sorted by time.  Understands Stooq's
    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,... files and plain headers such as
    Date,Time,Open,High,Low,Close,Volume or Datetime,Open,..."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return None
        cols = [h.strip().strip("<>").strip().lower() for h in header]

        def col(*names):
            for nm in names:
                if nm in cols:
                    return cols.index(nm)
            return None

        i_date = col("date", "day")
        i_time = col("time")
        i_dt = col("datetime", "timestamp", "date_time")
        i_o, i_h = col("open", "o"), col("high", "h", "max")
        i_l, i_c = col("low", "l", "min"), col("close", "c", "last")
        i_v = col("vol", "volume", "v")
        if None in (i_o, i_h, i_l, i_c) or (i_dt is None and i_date is None):
            return None
        ts, o, h, lo, c, v = [], [], [], [], [], []
        for row in reader:
            try:
                if i_dt is not None:
                    stamp = _digits(row[i_dt])
                else:
                    stamp = _digits(row[i_date])
                    if i_time is not None:
                        stamp += _digits(row[i_time]).zfill(6)
                stamp = (stamp + "000000")[:14]
                vals = (float(row[i_o]), float(row[i_h]), float(row[i_l]),
                        float(row[i_c]))
            except (ValueError, IndexError):
                continue
            ts.append(int(stamp))
            o.append(vals[0]); h.append(vals[1]); lo.append(vals[2]); c.append(vals[3])
            v.append(float(row[i_v]) if i_v is not None and row[i_v] else np.nan)
    if not ts:
        return None
    ts = np.array(ts, dtype=np.int64)
    bars = {"ts": ts, "open": np.array(o), "high": np.array(h),
            "low": np.array(lo), "close": np.array(c), "volume": np.array(v)}
    order = np.argsort(ts, kind="stable")
    keep = np.ones(len(ts), bool)
    keep[1:] = np.diff(ts[order]) != 0          # drop duplicated stamps
    order = order[keep]
    bars = {k: a[order] for k, a in bars.items()}
    good = ((bars["open"] > 0) & (bars["high"] > 0) & (bars["low"] > 0)
            & (bars["close"] > 0))
    return {k: a[good] for k, a in bars.items()}


def eligible(bars, min_bars, min_price, min_dollar_volume):
    if bars is None or len(bars["ts"]) < min_bars:
        return False
    if np.median(bars["close"]) < min_price:
        return False
    vol = bars["volume"]
    if min_dollar_volume > 0 and np.isfinite(vol).any():
        if np.nanmedian(vol * bars["close"]) < min_dollar_volume:
            return False
    return True


def select(files, args):
    """Randomly pick eligible stocks and ETFs (or any files if the kinds are
    unknown), loading candidates in random order until enough are found."""
    rng = random.Random(args.seed)
    wanted = {"stock": args.stocks, "etf": args.etfs}
    if not any(f["kind"] in wanted for f in files):
        wanted = {"unknown": args.stocks + args.etfs}
    chosen = []
    for kind, count in wanted.items():
        pool = [f for f in files if f["kind"] == kind]
        rng.shuffle(pool)
        for f in pool:
            if sum(c["kind"] == kind for c in chosen) >= count:
                break
            bars = load_bars(f["path"])
            if eligible(bars, args.min_bars, args.min_price, args.min_dollar_volume):
                chosen.append(dict(f, bars=len(bars["ts"]),
                                   first=int(bars["ts"][0]), last=int(bars["ts"][-1]),
                                   median_close=float(np.median(bars["close"]))))
    return chosen


# --------------------------------------------------------------------------
# Rolling fits

def ffill(a):
    """Forward-fill NaN along axis 0."""
    valid = ~np.isnan(a if a.ndim == 1 else a[:, 0])
    idx = np.where(valid, np.arange(len(valid)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def rolling_fit(values, window, method, step, pad):
    """mode[t] and quantiles q[t, PROBS] of values[t-window+1 .. t], refitted
    every step bars and carried forward in between (and over failed fits)."""
    n = len(values)
    mode = np.full(n, np.nan)
    q = np.full((n, len(PROBS)), np.nan)
    for t in range(window - 1, n, step):
        w = values[t - window + 1:t + 1]
        lo, hi = w.min(), w.max()
        if not hi > lo:
            continue
        try:
            if method == "Bernstein":
                # the Bernstein support must be given for prices; pad the
                # window's range so that the fit is not cut at its extremes
                r = pad * (hi - lo)
                fit = modefinder.ModeFinder(w, fix_lower=lo - r, fix_upper=hi + r)
            else:
                fit = modefinder.ModeFinder(w, smooth_option="kernel")
        except (ValueError, ArithmeticError):
            continue
        mode[t] = fit.mode
        q[t] = fit.quantile(PROBS)
    return ffill(mode), ffill(q)


# --------------------------------------------------------------------------
# Backtest

def bars_per_day(ts):
    _, counts = np.unique(ts // 1_000_000, return_counts=True)
    return float(np.median(counts))


def run_strategy(bars, buy_thr, sell_thr, start, cost):
    """Equity after each bar of the all-in / all-out long strategy.

    The decision at bar t compares close[t] with the thresholds fitted on
    bars up to t and is filled at open[t + 1].  Prices follow the open of
    each bar, and the last bar's close marks the final equity.  cost is the
    fraction of equity paid per buy or sell."""
    o, c = bars["open"], bars["close"]
    n = len(c)
    ok = ~np.isnan(buy_thr) & ~np.isnan(sell_thr)
    ev = np.zeros(n, np.int8)
    ev[ok & (c < buy_thr)] = 1
    ev[ok & (c > sell_thr)] = -1     # buy_thr <= sell_thr: never both
    ev[:start] = 0
    ev[-1] = 0
    return equity_from_events(o, c, ev, cost)


def equity_from_events(o, c, ev, cost):
    n = len(c)
    last = np.where(ev != 0, np.arange(n), -1)
    np.maximum.accumulate(last, out=last)
    pos = np.where(last >= 0, ev[np.maximum(last, 0)] == 1, False).astype(float)
    path = np.concatenate((o, c[-1:]))          # open[0..n-1], close[n-1]
    r = path[1:] / path[:-1] - 1                # interval s: path[s] -> path[s+1]
    expo = np.concatenate(([0.0], pos[:-1]))    # held over s if decided at s-1
    trades = np.abs(np.diff(np.concatenate(([0.0], expo))))
    gross = np.cumprod(1 + expo * r)
    equity = CAPITAL * np.cumprod((1 + expo * r) * (1 - cost * trades))
    return equity, expo, trades, gross


def buy_and_hold(bars, start, cost):
    n = len(bars["close"])
    ev = np.zeros(n, np.int8)
    ev[start] = 1
    return equity_from_events(bars["open"], bars["close"], ev, cost)


def metrics(equity, expo, trades, gross, start, cost, bpd):
    eq = np.concatenate(([CAPITAL], equity[start:]))
    rets = eq[1:] / eq[:-1] - 1
    sd = rets.std()
    sharpe = rets.mean() / sd * np.sqrt(252 * bpd) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float((eq / peak - 1).min())
    total = eq[-1] / CAPITAL - 1
    days = (len(eq) - 1) / bpd
    years = days / 252
    ann = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    # round trips: entry where exposure 0 -> 1, exit where 1 -> 0
    e = expo[start:]
    g = np.concatenate(([1.0], gross[start:]))
    d = np.diff(np.concatenate(([0.0], e, [0.0])))
    entries, exits = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    rt = g[exits] / g[entries] * (1 - cost) ** 2 - 1
    return {
        "final_value": float(eq[-1]),
        "total_return": float(total),
        "annual_return": float(ann),
        "max_drawdown": max_dd,
        "sharpe": float(sharpe),
        "trades": int(trades[start:].sum()),
        "round_trips": int(len(entries)),
        "win_rate": float((rt > 0).mean()) if len(rt) else np.nan,
        "avg_trip_return": float(rt.mean()) if len(rt) else np.nan,
        "time_in_market": float(e.mean()),
    }


def daily_equity(ts, equity, start):
    """Equity at the last bar of each trading day from start on."""
    days = ts[start:] // 1_000_000
    eq = equity[start:]
    last = np.flatnonzero(np.diff(np.concatenate((days, [-1]))) != 0)
    return days[last], eq[last]


# --------------------------------------------------------------------------
# One ticker

def fit_task(path, field, window, method, step, pad):
    """Worker: the rolling fit of one (ticker, field, window, method)."""
    t0 = time.time()
    bars = load_bars(path)
    mode, q = rolling_fit(bars[field], window, method, step, pad)
    return mode, q, time.time() - t0


def evaluate_ticker(info, fits, args):
    """Backtest every rule and cost on the rolling fits of one ticker;
    fits maps (field, window, method) to (mode, q)."""
    bars = load_bars(info["path"])
    ts, close = bars["ts"], bars["close"]
    n = len(close)
    start = max(args.windows) - 1          # same trading period for every window
    bpd = bars_per_day(ts)
    horizons = [h for h in args.horizons if h < n - start]
    rows, calib, daily = [], [], {}

    for cost in args.costs:
        eq, expo, trades, gross = buy_and_hold(bars, start, cost)
        m = metrics(eq, expo, trades, gross, start, cost, bpd)
        rows.append(dict(ticker=info["ticker"], kind=info["kind"], field="-",
                         window=0, method="buy&hold", rule="buy&hold",
                         cost_bps=cost * 1e4, **m))
        daily[("buy&hold", cost)] = daily_equity(ts, eq, start)

    for field, window, method in itertools.product(args.fields, args.windows,
                                                   args.methods):
        mode, q = fits[(field, window, method)]
        thr = {"mode": mode}
        thr.update({p: q[:, i] for i, p in enumerate(PROBS)})
        for rule in args.rules:
            b, s = RULES[rule]
            for cost in args.costs:
                eq, expo, trades, gross = run_strategy(bars, thr[b], thr[s],
                                                       start, cost)
                m = metrics(eq, expo, trades, gross, start, cost, bpd)
                rows.append(dict(ticker=info["ticker"], kind=info["kind"],
                                 field=field, window=window, method=method,
                                 rule=rule, cost_bps=cost * 1e4, **m))
                daily[(field, window, method, rule, cost)] = daily_equity(ts, eq, start)
        # calibration: how often the future close falls below each quantile
        for h in horizons:
            fut = close[start + h:]
            cur = q[start:n - h]
            for i, p in enumerate(PROBS):
                ok = ~np.isnan(cur[:, i])
                calib.append(dict(ticker=info["ticker"], field=field,
                                  window=window, method=method, horizon=h,
                                  prob=p, coverage=float((fut[ok] <= cur[ok, i]).mean())))
    return {"ticker": info["ticker"], "rows": rows, "calib": calib,
            "daily": daily, "bars": n,
            "bars_per_day": bpd, "start_ts": int(ts[start]), "end_ts": int(ts[-1])}


# --------------------------------------------------------------------------
# Portfolio and output

def portfolio_rows(results, args):
    """Equal split of $10,000 over the tickers, each part run on its own and
    marked at each day's close; a part is cash before its ticker's first
    trading day and keeps its last value after its ticker's data ends."""
    all_days = np.unique(np.concatenate(
        [r["daily"][("buy&hold", args.costs[0])][0] for r in results]))
    out = []
    for key in results[0]["daily"]:
        curves = []
        for r in results:
            days, eq = r["daily"][key]
            idx = np.searchsorted(days, all_days, side="right") - 1
            curve = np.where(idx >= 0, eq[np.maximum(idx, 0)], CAPITAL)
            curves.append(curve)
        port = np.mean(curves, axis=0)
        port = np.concatenate(([CAPITAL], port))
        rets = port[1:] / port[:-1] - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
        years = (len(port) - 1) / 252
        total = port[-1] / CAPITAL - 1
        if key[0] == "buy&hold":
            field, window, method, rule, cost = "-", 0, "buy&hold", "buy&hold", key[1]
        else:
            field, window, method, rule, cost = key
        sub = [row for r in results for row in r["rows"]
               if (row["field"], row["window"], row["method"], row["rule"],
                   row["cost_bps"]) == (field, window, method, rule, cost * 1e4)]
        bh = {row["ticker"]: row["total_return"] for r in results for row in r["rows"]
              if row["method"] == "buy&hold" and row["cost_bps"] == cost * 1e4}
        out.append(dict(
            field=field, window=window, method=method, rule=rule,
            cost_bps=cost * 1e4, final_value=float(port[-1]),
            total_return=float(total),
            annual_return=float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan,
            max_drawdown=float((port / np.maximum.accumulate(port) - 1).min()),
            sharpe=float(sharpe),
            trades_per_ticker=float(np.mean([s["trades"] for s in sub])),
            time_in_market=float(np.mean([s["time_in_market"] for s in sub])),
            win_rate=float(np.nanmean([s["win_rate"] for s in sub])),
            beat_bh=float(np.mean([s["total_return"] > bh[s["ticker"]] for s in sub])),
        ))
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="/Users/bowenliu/Downloads/data/5 min")
    p.add_argument("--out", default=None,
                   help="output directory (default results/run_<timestamp>)")
    p.add_argument("--stocks", type=int, default=10)
    p.add_argument("--etfs", type=int, default=5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--tickers", nargs="*", default=None,
                   help="use these tickers instead of a random selection")
    p.add_argument("--fields", nargs="+", default=list(FIELDS), choices=FIELDS)
    p.add_argument("--windows", nargs="+", type=int, default=[78, 390, 1560],
                   help="rolling window lengths in bars (78 bars = 1 day)")
    p.add_argument("--methods", nargs="+", default=["Bernstein", "kernel"],
                   choices=["Bernstein", "kernel"])
    p.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    p.add_argument("--costs-bps", nargs="+", type=float, default=[0.0, 2.0],
                   help="cost per buy or sell, in basis points of equity")
    p.add_argument("--step", type=int, default=1,
                   help="refit every step bars (1: at every new bar)")
    p.add_argument("--pad", type=float, default=0.1,
                   help="Bernstein support: window range widened by pad * range")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 78],
                   help="forecast horizons (bars) checked for quantile coverage")
    p.add_argument("--min-bars", type=int, default=5000)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-volume", type=float, default=100_000.0,
                   help="minimum median dollar volume per bar")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    args.costs = [c / 1e4 for c in args.costs_bps]
    if min(args.windows) <= 38:
        p.error("windows must be longer than 38 bars (modefinder's minimum)")
    if args.out is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(here, "results", f"run_{stamp}")
    args.min_bars = max(args.min_bars, max(args.windows) + 100)
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    files = find_files(args.data)
    if not files:
        sys.exit(f"no .txt/.csv files under {args.data}")
    print(f"{len(files)} files under {args.data}")
    if args.tickers:
        want = {t.upper() for t in args.tickers}
        chosen = [dict(f) for f in files if f["ticker"] in want]
    else:
        chosen = select(files, args)
    if not chosen:
        sys.exit("no eligible tickers")
    print("selected:", ", ".join(f"{c['ticker']} ({c['kind']})" for c in chosen))

    # one task per (ticker, field, window, method), submitted ticker by
    # ticker so that each ticker is backtested, and its fits freed, as soon
    # as they are all done
    combos = list(itertools.product(args.fields, args.windows, args.methods))
    combos.sort(key=lambda k: -k[1])
    by_name = {c["ticker"]: c for c in chosen}
    fits = {c["ticker"]: {} for c in chosen}
    results = []
    t0 = time.time()
    total = len(chosen) * len(combos)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(fit_task, c["path"], *k, args.step, args.pad):
                (c["ticker"], k) for c in chosen for k in combos}
        for done, fut in enumerate(as_completed(futs), 1):
            ticker, k = futs[fut]
            mode, q, secs = fut.result()
            fits[ticker][k] = (mode, q)
            if done % max(1, total // 20) == 0 or done == total:
                print(f"  fits {done}/{total} ({time.time() - t0:.0f} s; last "
                      f"{ticker} {'/'.join(map(str, k))} {secs:.0f} s)", flush=True)
            if len(fits[ticker]) == len(combos):
                results.append(evaluate_ticker(by_name[ticker],
                                               fits.pop(ticker), args))
    results.sort(key=lambda r: r["ticker"])
    by_ticker = {r["ticker"]: r for r in results}

    sel_rows = []
    for c in chosen:
        r = by_ticker[c["ticker"]]
        sel_rows.append(dict(ticker=c["ticker"], kind=c["kind"],
                             path=os.path.relpath(c["path"], args.data),
                             bars=r["bars"], bars_per_day=r["bars_per_day"],
                             first=str(c.get("first", "")), trade_start=str(r["start_ts"]),
                             last=str(r["end_ts"]),
                             median_close=c.get("median_close", "")))
    write_csv(os.path.join(args.out, "selection.csv"), sel_rows)
    write_csv(os.path.join(args.out, "per_ticker.csv"),
              [row for r in results for row in r["rows"]])
    write_csv(os.path.join(args.out, "calibration.csv"),
              [row for r in results for row in r["calib"]])
    write_csv(os.path.join(args.out, "portfolio.csv"), portfolio_rows(results, args))
    config = {k: v for k, v in vars(args).items()}
    config.update(capital=CAPITAL, probs=PROBS, rule_thresholds=RULES,
                  run_seconds=round(time.time() - t0, 1),
                  modefinder_version=modefinder.__version__)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str)

    import report
    path = report.write_report(args.out)
    print(f"report: {path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
