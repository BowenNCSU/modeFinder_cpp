"""Rolling quantile trading simulation with modefinder, on a large sample.

The three combinations that ranked first with 2 bps per trade in the
second round (``results/run_20261019_seed2027``) are tested on a random
sample of stocks and ETFs with 5-minute bars:

    q25/q90 · high  · 1560 bars · kernel
    q25/q90 · open  · 1560 bars · kernel
    q25/q90 · close · 1560 bars · kernel

At every bar, the last 1560 bars (about 20 trading days) of the price field
are fitted with ``modefinder.ModeFinder(..., smooth_option="kernel")``: the
oldest bar leaves the window and the newest enters.  A long-only strategy
starting with $10,000 per ticker

    buys  (all in)  when the current close is below the fitted 25% quantile,
    sells (all out) when the current close is above the fitted 90% quantile,

and holds in between.  Orders are decided at a bar's close and filled at
the next bar's open, paying the cost on each buy and sell.  Each ticker is
compared with buy and hold, and the tickers are ranked by profit.

    python simulations/stock_mode_quantile/simulate.py \\
        --data "/Users/bowenliu/Downloads/data/5 min" --count 2000

Requires numpy and modefinder (``pip install .`` at the repository root).
"""

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import modefinder

WINDOW = 1560                   # bars, about 20 trading days of 78 bars
METHOD = "kernel"
BUY_Q, SELL_Q = 0.25, 0.90      # buy below the 25% quantile, sell above 90%
FIELDS = ("high", "open", "close")
RULE = "q25/q90"
CAPITAL = 10_000.0
JUMP = 0.4                      # flag bar-to-bar moves above 40% (splits, bad ticks)


def combo_name(field):
    return f"{RULE} · {field} · {WINDOW} · {METHOD}"


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


def check_file(f, min_bars, min_price, min_dollar_volume):
    """Worker: f with a summary of its bars if it is eligible, else None."""
    try:
        bars = load_bars(f["path"])
    except (OSError, UnicodeDecodeError):
        return None
    if bars is None or len(bars["ts"]) < min_bars:
        return None
    close = bars["close"]
    if np.median(close) < min_price:
        return None
    vol = bars["volume"]
    if min_dollar_volume > 0 and np.isfinite(vol).any():
        if np.nanmedian(vol * close) < min_dollar_volume:
            return None
    return dict(f, bars=len(close), first=int(bars["ts"][0]),
                last=int(bars["ts"][-1]), median_close=float(np.median(close)))


def select(files, args, pool):
    """A seeded random sample of args.count eligible files.  Every file is
    checked (in parallel), so the sample is uniform over the eligible ones
    and the same seed always gives the same sample."""
    checked = pool.map(check_file, files, *([v] * len(files) for v in
                       (args.min_bars, args.min_price, args.min_dollar_volume)),
                       chunksize=64)
    ok = [f for f in checked if f is not None]
    rng = random.Random(args.seed)
    chosen = rng.sample(ok, min(args.count, len(ok)))
    chosen.sort(key=lambda f: f["ticker"])
    return chosen, len(ok)


# --------------------------------------------------------------------------
# Rolling fit and backtest

def ffill(a):
    """Forward-fill NaN along axis 0."""
    valid = ~np.isnan(a if a.ndim == 1 else a[:, 0])
    idx = np.where(valid, np.arange(len(valid)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def rolling_quantiles(values, step):
    """q[t] = the fitted (BUY_Q, SELL_Q) quantiles of values[t-WINDOW+1 .. t],
    refitted every step bars and carried forward in between (and over
    failed fits)."""
    n = len(values)
    q = np.full((n, 2), np.nan)
    for t in range(WINDOW - 1, n, step):
        w = values[t - WINDOW + 1:t + 1]
        if not w.max() > w.min():
            continue
        try:
            fit = modefinder.ModeFinder(w, smooth_option=METHOD)
        except (ValueError, ArithmeticError):
            continue
        q[t] = fit.quantile((BUY_Q, SELL_Q))
    return ffill(q)


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
    years = (len(eq) - 1) / bpd / 252
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


def simulate_ticker(info, step, cost):
    """Worker: the three combinations and buy and hold on one ticker."""
    t0 = time.time()
    bars = load_bars(info["path"])
    ts, close = bars["ts"], bars["close"]
    start = WINDOW - 1
    bpd = bars_per_day(ts)
    path = np.concatenate((bars["open"], close[-1:]))
    jump = float(np.abs(np.diff(np.log(path[start:]))).max())
    base = dict(ticker=info["ticker"], kind=info["kind"])
    rows, daily = [], {}

    eq, expo, trades, gross = buy_and_hold(bars, start, cost)
    rows.append(dict(base, field="-", combo="buy & hold",
                     **metrics(eq, expo, trades, gross, start, cost, bpd)))
    daily["buy & hold"] = daily_equity(ts, eq, start)
    for field in FIELDS:
        q = rolling_quantiles(bars[field], step)
        eq, expo, trades, gross = run_strategy(bars, q[:, 0], q[:, 1], start, cost)
        name = combo_name(field)
        rows.append(dict(base, field=field, combo=name,
                         **metrics(eq, expo, trades, gross, start, cost, bpd)))
        daily[name] = daily_equity(ts, eq, start)
    return {"ticker": info["ticker"], "rows": rows, "daily": daily,
            "bars": len(close), "bars_per_day": bpd,
            "start_ts": int(ts[start]), "end_ts": int(ts[-1]),
            "max_jump": jump, "seconds": time.time() - t0}


# --------------------------------------------------------------------------
# Portfolio, ranking and output

def portfolio_rows(results):
    """Equal split of $10,000 over the tickers, each part run on its own and
    marked at each day's close; a part is cash before its ticker's first
    trading day and keeps its last value after its ticker's data ends."""
    all_days = np.unique(np.concatenate(
        [r["daily"]["buy & hold"][0] for r in results]))
    bh = {r["ticker"]: r["rows"][0]["total_return"] for r in results}
    out = []
    for key in results[0]["daily"]:
        curves = []
        for r in results:
            days, eq = r["daily"][key]
            idx = np.searchsorted(days, all_days, side="right") - 1
            curves.append(np.where(idx >= 0, eq[np.maximum(idx, 0)], CAPITAL))
        port = np.concatenate(([CAPITAL], np.mean(curves, axis=0)))
        rets = port[1:] / port[:-1] - 1
        years = (len(port) - 1) / 252
        total = port[-1] / CAPITAL - 1
        sub = [row for r in results for row in r["rows"] if row["combo"] == key]
        ret = np.array([s["total_return"] for s in sub])
        out.append(dict(
            combo=key, final_value=float(port[-1]), total_return=float(total),
            annual_return=float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan,
            max_drawdown=float((port / np.maximum.accumulate(port) - 1).min()),
            sharpe=float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
            trades_per_ticker=float(np.mean([s["trades"] for s in sub])),
            time_in_market=float(np.mean([s["time_in_market"] for s in sub])),
            win_rate=float(np.nanmean([s["win_rate"] for s in sub])),
            profitable=float((ret > 0).mean()),
            beat_bh=float(np.mean([s["total_return"] > bh[s["ticker"]] for s in sub])),
            median_ticker_return=float(np.median(ret)),
        ))
    return out


def ranking_rows(results):
    """One row per ticker, ranked by the mean return of the three
    combinations."""
    out = []
    for r in results:
        by = {row["combo"]: row for row in r["rows"]}
        bh = by["buy & hold"]
        strat = [by[combo_name(f)] for f in FIELDS]
        mean_ret = float(np.mean([s["total_return"] for s in strat]))
        row = dict(ticker=r["ticker"], kind=r["rows"][0]["kind"],
                   mean_return=mean_ret,
                   final_value=float(np.mean([s["final_value"] for s in strat])))
        for f, s in zip(FIELDS, strat):
            row[f"return_{f}"] = s["total_return"]
        row.update(
            mean_max_drawdown=float(np.mean([s["max_drawdown"] for s in strat])),
            mean_sharpe=float(np.mean([s["sharpe"] for s in strat])),
            mean_trades=float(np.mean([s["trades"] for s in strat])),
            mean_time_in_market=float(np.mean([s["time_in_market"] for s in strat])),
            bh_return=bh["total_return"], bh_max_drawdown=bh["max_drawdown"],
            excess_vs_bh=mean_ret - bh["total_return"],
            max_jump=r["max_jump"], jump_flag=int(r["max_jump"] > np.log1p(JUMP)))
        out.append(row)
    out.sort(key=lambda x: -x["mean_return"])
    for i, row in enumerate(out, 1):
        row["rank"] = i
    return [dict(rank=row.pop("rank"), **row) for row in out]


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
    p.add_argument("--count", type=int, default=2000,
                   help="number of random stocks and ETFs")
    p.add_argument("--seed", type=int, default=2028)
    p.add_argument("--tickers", nargs="*", default=None,
                   help="use these tickers instead of a random selection")
    p.add_argument("--cost-bps", type=float, default=2.0,
                   help="cost per buy or sell, in basis points of equity")
    p.add_argument("--step", type=int, default=1,
                   help="refit every step bars (1: at every new bar)")
    p.add_argument("--min-bars", type=int, default=5000)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-volume", type=float, default=100_000.0,
                   help="minimum median dollar volume per bar")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    args.cost = args.cost_bps / 1e4
    if args.out is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(here, "results", f"run_{stamp}")
    args.min_bars = max(args.min_bars, WINDOW + 100)
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    files = find_files(args.data)
    if not files:
        sys.exit(f"no .txt/.csv files under {args.data}")
    print(f"{len(files)} files under {args.data}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        if args.tickers:
            want = {t.upper() for t in args.tickers}
            files = [f for f in files if f["ticker"] in want]
        chosen, n_eligible = select(files, args, pool)
        if not chosen:
            sys.exit("no eligible tickers")
        print(f"{n_eligible} eligible, {len(chosen)} selected "
              f"({sum(c['kind'] == 'stock' for c in chosen)} stocks, "
              f"{sum(c['kind'] == 'etf' for c in chosen)} ETFs) "
              f"in {time.time() - t0:.0f} s", flush=True)

        results = []
        futs = [pool.submit(simulate_ticker, c, args.step, args.cost) for c in chosen]
        for done, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if done % max(1, len(chosen) // 20) == 0 or done == len(chosen):
                print(f"  tickers {done}/{len(chosen)} ({time.time() - t0:.0f} s)",
                      flush=True)
    results.sort(key=lambda r: r["ticker"])
    by_ticker = {r["ticker"]: r for r in results}

    sel_rows = []
    for c in chosen:
        r = by_ticker[c["ticker"]]
        sel_rows.append(dict(ticker=c["ticker"], kind=c["kind"],
                             path=os.path.relpath(c["path"], args.data),
                             bars=r["bars"], bars_per_day=r["bars_per_day"],
                             first=str(c["first"]), trade_start=str(r["start_ts"]),
                             last=str(r["end_ts"]), median_close=c["median_close"]))
    write_csv(os.path.join(args.out, "selection.csv"), sel_rows)
    write_csv(os.path.join(args.out, "per_ticker.csv"),
              [row for r in results for row in r["rows"]])
    write_csv(os.path.join(args.out, "ranking.csv"), ranking_rows(results))
    write_csv(os.path.join(args.out, "portfolio.csv"), portfolio_rows(results))
    config = dict(vars(args), capital=CAPITAL, window=WINDOW, method=METHOD,
                  buy_quantile=BUY_Q, sell_quantile=SELL_Q, fields=FIELDS,
                  jump_flag=JUMP, eligible=n_eligible, files=len(files),
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
