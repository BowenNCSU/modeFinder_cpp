"""Which observed skew / time-in-market thresholds pick profitable tickers.

Reruns the three combinations of simulate.py (q25/q90 · high, open, close ·
1560 bars · kernel) on the tickers of a finished run and, at every bar,
computes features from the data observed up to that bar only:

    skew_d20, skew_dall   skewness of daily close-to-close log returns over
                          the last 20 days / all days so far
    skew_h20, skew_hall   skewness of hourly log returns within the session
                          over the last 140 hours (about 20 days) / all so far
    tim_all, tim_20d      share of bars the strategy has held the ticker since
                          trading started / over the last 1560 bars
    mom_5d, mom_20d       price change over the last 390 / 1560 bars
    vol_20d               annualized sd of the last 20 daily returns
    band_width            (fitted 90% quantile - fitted 25% quantile) / close
    strat_ret             the strategy's return on the ticker so far

It then tests thresholds on these features three ways:

1. Panel: one row per ticker and trading day, with the features at the
   day's close and the strategy's return over the next 20 trading days.
   Thresholds are searched on the first half of the period and checked on
   the second half.
2. Tickers at mid-period: which tickers pass at the middle day, and how the
   strategy does on them from there to the end.
3. Gated strategy: the strategy only buys while the thresholds hold
   ("entry"), or also sells as soon as they stop holding ("exit").

    python simulations/stock_mode_quantile/thresholds.py \\
        --run simulations/stock_mode_quantile/results/run_20261019_top3_2000

The rolling fits (the slow part, about 20 minutes for 2,000 tickers on 14
cores) are cached in <out>/cache, so a rerun only redoes the analysis.
"""

import argparse
import csv
import gzip
import itertools
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim  # noqa: E402

HORIZON = 20            # trading days ahead for the panel outcome
MIN_OBS = 10            # fewest returns for a skewness or sd
HOURS_20D = 140         # hourly returns in about 20 trading days
FEATURES = ("skew_d20", "skew_dall", "skew_h20", "skew_hall", "tim_all",
            "tim_20d", "mom_5d", "mom_20d", "vol_20d", "band_width", "strat_ret")
NATURAL = {"skew_d20": 0.0, "skew_dall": 0.0, "skew_h20": 0.0, "skew_hall": 0.0,
           "tim_all": 0.5, "tim_20d": 0.5, "mom_5d": 0.0, "mom_20d": 0.0,
           "strat_ret": 0.0}
MIN_COVER = 0.2         # a searched rule must pass at least 20% of train rows
VARIANTS = ("entry", "exit")


def rule(terms, join="and"):
    name = f" {join} ".join(f"{f} {op} {thr:.3g}" for f, op, thr in terms)
    return {"name": name or "no filter", "terms": list(terms), "join": join}


MUST = [rule([("skew_d20", ">", 0)]), rule([("skew_dall", ">", 0)]),
        rule([("skew_h20", ">", 0)]), rule([("skew_hall", ">", 0)]),
        rule([("tim_all", "<", 0.5)]), rule([("tim_20d", "<", 0.5)]),
        rule([("skew_d20", ">", 0), ("tim_all", "<", 0.5)]),
        rule([("skew_d20", ">", 0), ("tim_all", "<", 0.5)], "or"),
        rule([("skew_h20", ">", 0), ("tim_all", "<", 0.5)]),
        rule([("skew_h20", ">", 0), ("tim_all", "<", 0.5)], "or")]
BASE = rule([])
# benchmarks for the gated strategy: buy and hold, and gates that open on a
# random share of days, which cut exposure without using any information
BENCH = [dict(rule([]), name="buy & hold", bh=True),
         dict(rule([]), name="random 25% of days", random=0.25),
         dict(rule([]), name="random 50% of days", random=0.5)]


def variants_of(r):
    return VARIANTS if r["terms"] or "random" in r else ("-",)


# --------------------------------------------------------------------------
# Features, all causal: the value at bar t uses bars up to t only

def window_moments(r, n=None):
    """sd and skewness of r[i-n+1 .. i] (n None: r[0 .. i]) at every i."""
    k = len(r)
    i = np.arange(1, k + 1)
    lo = np.zeros(k, int) if n is None else np.maximum(i - n, 0)
    cnt = i - lo
    m1, m2, m3 = ((cs[i] - cs[lo]) / cnt for cs in
                  (np.concatenate(([0.0], np.cumsum(r ** p))) for p in (1, 2, 3)))
    var = m2 - m1 ** 2
    bad = (cnt < MIN_OBS) | ~(var > 1e-9 * m2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(var)
        sk = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / var ** 1.5
    sd[bad] = np.nan
    sk[bad] = np.nan
    return sd, sk


def at_bars(n, idx, vals):
    """vals placed at bars idx and carried forward to the following bars."""
    a = np.full(n, np.nan)
    a[idx] = vals
    return sim.ffill(a)


def day_ends(ts):
    day = ts // 1_000_000
    return np.flatnonzero(np.diff(np.concatenate((day, [-1]))) != 0)


def price_features(bars):
    ts, o, c = bars["ts"], bars["open"], bars["close"]
    n = len(c)
    dend = day_ends(ts)
    dret = np.diff(np.log(c[dend]))
    sd20, sk20 = window_moments(dret, 20)
    _, skall = window_moments(dret)
    # hourly returns within the session: the first hour of a day starts at
    # the day's first open, so overnight gaps are left out
    hour = ts // 10_000
    hend = np.flatnonzero(np.diff(np.concatenate((hour, [-1]))) != 0)
    dstart = np.concatenate(([0], dend[:-1] + 1))
    day_of = np.searchsorted(dend, hend)
    prev = np.empty(len(hend))
    prev[1:] = c[hend[:-1]]
    first = np.ones(len(hend), bool)
    first[1:] = day_of[1:] != day_of[:-1]
    prev[first] = o[dstart[day_of[first]]]
    hret = np.log(c[hend] / prev)
    _, skh20 = window_moments(hret, HOURS_20D)
    _, skhall = window_moments(hret)
    f = {"skew_d20": at_bars(n, dend[1:], sk20),
         "skew_dall": at_bars(n, dend[1:], skall),
         "vol_20d": at_bars(n, dend[1:], sd20 * np.sqrt(252)),
         "skew_h20": at_bars(n, hend, skh20),
         "skew_hall": at_bars(n, hend, skhall)}
    for name, lag in (("mom_5d", 390), ("mom_20d", sim.WINDOW - 1)):
        f[name] = np.full(n, np.nan)
        f[name][lag:] = c[lag:] / c[:-lag] - 1
    return f


def combo_features(bars, q, start, expo, equity):
    """Features that depend on the strategy, for one price field."""
    n = len(expo)
    t = np.arange(n)
    held = np.where(t >= start, expo, 0.0)     # expo[t] is known at bar t
    cs = np.cumsum(held)
    tim_all = np.full(n, np.nan)
    tim_all[start:] = cs[start:] / (t[start:] - start + 1)
    prev = np.where(t >= sim.WINDOW, cs[np.maximum(t - sim.WINDOW, 0)], 0.0)
    tim_20d = np.full(n, np.nan)
    tim_20d[start:] = (cs - prev)[start:] / np.minimum(sim.WINDOW, t[start:] - start + 1)
    strat = np.empty(n)                        # value at open[t], known at t
    strat[0] = 0.0
    strat[1:] = equity[:-1] / sim.CAPITAL - 1
    return {"tim_all": tim_all, "tim_20d": tim_20d, "strat_ret": strat,
            "band_width": (q[:, 1] - q[:, 0]) / bars["close"]}


def events(bars, q, start):
    """The orders of simulate.run_strategy: +1 buy, -1 sell, 0 hold."""
    c = bars["close"]
    ok = ~np.isnan(q[:, 0]) & ~np.isnan(q[:, 1])
    ev = np.zeros(len(c), np.int8)
    ev[ok & (c < q[:, 0])] = 1
    ev[ok & (c > q[:, 1])] = -1
    ev[:start] = 0
    ev[-1] = 0
    return ev


def gate_mask(r, f):
    """Bars (or panel rows) where rule r holds; NaN features fail."""
    if not r["terms"]:
        return np.ones(len(next(iter(f.values()))), bool)
    with np.errstate(invalid="ignore"):
        m = [f[name] > thr if op == ">" else f[name] < thr
             for name, op, thr in r["terms"]]
    return np.logical_and.reduce(m) if r["join"] == "and" else np.logical_or.reduce(m)


# --------------------------------------------------------------------------
# Workers

def fit_worker(info, step):
    """Rolling quantiles of the three price fields, saved to the cache."""
    if os.path.exists(info["cache"]):
        return info["ticker"], 0.0
    t0 = time.time()
    bars = sim.load_bars(info["path"])
    q = np.stack([sim.rolling_quantiles(bars[f], step) for f in sim.FIELDS])
    tmp = info["cache"][:-4] + ".tmp.npy"
    np.save(tmp, q)
    os.replace(tmp, info["cache"])
    return info["ticker"], time.time() - t0


def prepare(info, cost):
    bars = sim.load_bars(info["path"])
    q = np.load(info["cache"])
    start = sim.WINDOW - 1
    pf = price_features(bars)
    combos = []
    for k, field in enumerate(sim.FIELDS):
        ev = events(bars, q[k], start)
        equity, expo, _, _ = sim.equity_from_events(bars["open"], bars["close"], ev, cost)
        combos.append({"field": field, "ev": ev, "equity": equity,
                       "feats": dict(pf, **combo_features(bars, q[k], start, expo, equity))})
    return bars, start, combos


def panel_worker(info, cost):
    """One row per trading day: features at the day's close (strategy
    features averaged over the three fields) and forward returns."""
    bars, start, combos = prepare(info, cost)
    dend = day_ends(bars["ts"])
    dend = dend[dend >= start]
    X = np.column_stack([np.mean([c["feats"][f][dend] for c in combos], axis=0)
                         for f in FEATURES])
    eq = np.array([c["equity"] for c in combos])
    at = eq[:, dend]                           # value at the next open
    fwd = np.full(len(dend), np.nan)
    bh = np.full(len(dend), np.nan)
    if len(dend) > HORIZON:
        fwd[:-HORIZON] = (at[:, HORIZON:] / at[:, :-HORIZON] - 1).mean(axis=0)
        path = np.concatenate((bars["open"], bars["close"][-1:]))
        bh[:-HORIZON] = path[dend[HORIZON:] + 1] / path[dend[:-HORIZON] + 1] - 1
    fwd_end = (eq[:, -1:] / at - 1).mean(axis=0)
    return {"ticker": info["ticker"], "day": bars["ts"][dend] // 1_000_000,
            "X": X, "fwd": fwd, "fwd_end": fwd_end, "bh_fwd": bh,
            "base": {c["field"]: c["equity"][-1] / sim.CAPITAL - 1 for c in combos}}


def gate_worker(info, cost, rules, mid_day):
    """Each rule as an entry gate and as an entry + exit gate."""
    bars, start, combos = prepare(info, cost)
    ts, o, cl = bars["ts"], bars["open"], bars["close"]
    dend = day_ends(ts)
    dend = dend[dend >= start]
    k = np.searchsorted(ts[dend] // 1_000_000, mid_day)
    t_mid = dend[min(k, len(dend) - 1)]
    rows, daily = [], []
    _, day_id = np.unique(ts // 1_000_000, return_inverse=True)
    for r in rules:
        if "random" in r:
            rng = np.random.default_rng(zlib.crc32(info["ticker"].encode()))
            open_days = rng.random(day_id.max() + 1) < r["random"]
        for variant in variants_of(r):
            stats, curves = [], []
            for c in combos:
                if r.get("bh"):
                    eq, expo, trades, _ = sim.buy_and_hold(bars, start, cost)
                else:
                    g = open_days[day_id] if "random" in r else gate_mask(r, c["feats"])
                    ev = c["ev"].copy()
                    ev[(ev == 1) & ~g] = 0
                    if variant == "exit":
                        ev[~g] = -1
                        ev[:start] = 0
                        ev[-1] = 0
                    eq, expo, trades, _ = sim.equity_from_events(o, cl, ev, cost)
                stats.append((eq[-1] / sim.CAPITAL - 1, eq[-1] / eq[t_mid] - 1,
                              expo[start:].mean(), trades[start:].sum(),
                              trades[t_mid + 1:].sum()))
                curves.append(sim.daily_equity(ts, eq, start)[1])
            s = np.mean(stats, axis=0)
            rows.append({"ticker": info["ticker"], "rule": r["name"], "variant": variant,
                         "total_return": s[0], "second_half_return": s[1],
                         "time_in_market": s[2], "trades": s[3],
                         "second_half_trades": s[4]})
            daily.append(np.mean(curves, axis=0))
    days = sim.daily_equity(ts, combos[0]["equity"], start)[0]
    return {"rows": rows, "days": days, "daily": np.array(daily)}


# --------------------------------------------------------------------------
# Analysis

def avg_rank(x):
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    _, first, counts = np.unique(xs, return_index=True, return_counts=True)
    r = np.empty(len(x))
    r[order] = np.repeat(first + (counts - 1) / 2, counts)
    return r


def rank_ic(x, y, day):
    """Mean over days of the cross-sectional rank correlation of x and y."""
    out = []
    for d in np.unique(day):
        m = (day == d) & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 20 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
            out.append(np.corrcoef(avg_rank(x[m]), avg_rank(y[m]))[0, 1])
    return float(np.mean(out)) if out else np.nan


def summarize(mask, y):
    yp, yf = y[mask], y[~mask]
    f = lambda a, g: float(g(a)) if len(a) else np.nan   # noqa: E731
    return {"cover": float(mask.mean()), "n_pass": int(mask.sum()),
            "mean_pass": f(yp, np.mean), "mean_fail": f(yf, np.mean),
            "pos_pass": f(yp, lambda a: (a > 0).mean()),
            "pos_fail": f(yf, lambda a: (a > 0).mean()),
            "neg_pass": f(yp, lambda a: (a < 0).mean()),
            "neg_fail": f(yf, lambda a: (a < 0).mean())}


def search(sets):
    """Single thresholds (train deciles and natural values) and pairs (train
    30/50/70% points and natural values), scored on each row set."""
    train = sets["train"][0]
    grid1, grid2 = {}, {}
    for f in FEATURES:
        x = train[f][np.isfinite(train[f])]
        nat = [NATURAL[f]] if f in NATURAL else []
        grid1[f] = sorted(set(np.round(np.percentile(x, range(10, 100, 10)), 6)) | set(nat))
        grid2[f] = sorted(set(np.round(np.percentile(x, (30, 50, 70)), 6)) | set(nat))
    terms1 = [(f, op, float(t)) for f in FEATURES for t in grid1[f] for op in "<>"]
    terms2 = {f: [(f, op, float(t)) for t in grid2[f] for op in "<>"] for f in FEATURES}
    masks = {name: {} for name in sets}

    def term_mask(name, term):
        if term not in masks[name]:
            masks[name][term] = gate_mask(rule([term]), sets[name][0])
        return masks[name][term]

    rules = [(rule([t]), "single") for t in terms1]
    rules += [(rule([a, b]), "pair") for fa, fb in itertools.combinations(FEATURES, 2)
              for a in terms2[fa] for b in terms2[fb]]
    rules += [(r, "requested") for r in MUST]
    out = []
    for r, kind in rules:
        row = {"rule": r["name"], "kind": kind, "_rule": r}
        for name, (cols, y) in sets.items():
            ms = [term_mask(name, t) for t in r["terms"]]
            m = np.logical_and.reduce(ms) if r["join"] == "and" else np.logical_or.reduce(ms)
            row.update({f"{name}_{k}": v for k, v in summarize(m, y).items()})
        out.append(row)
    return out


def portfolio(days_list, daily_list, j, mid_day):
    all_days = np.unique(np.concatenate(days_list))
    curves = []
    for days, daily in zip(days_list, daily_list):
        idx = np.searchsorted(days, all_days, side="right") - 1
        curves.append(np.where(idx >= 0, daily[j][np.maximum(idx, 0)], sim.CAPITAL))
    port = np.concatenate(([sim.CAPITAL], np.mean(curves, axis=0)))
    k = 1 + np.searchsorted(all_days, mid_day)

    def stats(p):
        rets = p[1:] / p[:-1] - 1
        return (float(p[-1] / p[0] - 1),
                float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
                float((p / np.maximum.accumulate(p) - 1).min()))
    full, half = stats(port), stats(port[k:])
    return {"return": full[0], "sharpe": full[1], "max_drawdown": full[2],
            "second_half_return": half[0], "second_half_sharpe": half[1],
            "second_half_max_drawdown": half[2]}


def write_csv(path, rows):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if not k.startswith("_")],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_pool(pool, fn, items, label, t0, *args):
    futs = {pool.submit(fn, it, *args): it for it in items}
    out = []
    for done, fut in enumerate(as_completed(futs), 1):
        out.append(fut.result())
        if done % max(1, len(items) // 20) == 0 or done == len(items):
            print(f"  {label} {done}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="a finished simulate.py run directory")
    p.add_argument("--data", default=None, help="price files (default: the run's --data)")
    p.add_argument("--out", default=None, help="output directory (default <run>_thresholds)")
    p.add_argument("--cache", default=None, help="fitted quantiles (default <out>/cache)")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    args.run = args.run.rstrip("/")
    args.out = args.out or args.run + "_thresholds"
    args.cache = args.cache or os.path.join(args.out, "cache")
    return args


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    data = args.data or cfg["data"]
    cost = cfg["cost_bps"] / 1e4
    with open(os.path.join(args.run, "selection.csv"), newline="") as fh:
        sel = list(csv.DictReader(fh))
    with open(os.path.join(args.run, "per_ticker.csv"), newline="") as fh:
        ref = {(r["ticker"], r["field"]): float(r["total_return"])
               for r in csv.DictReader(fh)}
    os.makedirs(args.cache, exist_ok=True)
    infos = [{"ticker": s["ticker"], "path": os.path.join(data, s["path"]),
              "cache": os.path.join(args.cache, f"{i:04d}_{s['ticker']}.npy")}
             for i, s in enumerate(sel)]
    print(f"{len(infos)} tickers from {args.run}", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        run_pool(pool, fit_worker, infos, "fits", t0, cfg["step"])
        panels = run_pool(pool, panel_worker, infos, "panel", t0, cost)
        panels.sort(key=lambda p: p["ticker"])

        diff = max(abs(p["base"][f] - ref[(p["ticker"], f)])
                   for p in panels for f in sim.FIELDS)
        print(f"  largest gap to the run's per_ticker.csv returns: {diff:.2e}", flush=True)

        days = np.unique(np.concatenate([p["day"] for p in panels]))
        mid = len(days) // 2
        mid_day = int(days[mid])
        gidx = {int(d): i for i, d in enumerate(days)}
        P = {f: np.concatenate([p["X"][:, j] for p in panels]) for j, f in enumerate(FEATURES)}
        day = np.concatenate([p["day"] for p in panels])
        gi = np.array([gidx[int(d)] for d in day])
        fwd = np.concatenate([p["fwd"] for p in panels])
        fwd_end = np.concatenate([p["fwd_end"] for p in panels])
        bh_fwd = np.concatenate([p["bh_fwd"] for p in panels])
        ticker = np.concatenate([[p["ticker"]] * len(p["day"]) for p in panels])
        ok = np.isfinite(fwd)
        rows_of = {"train": ok & (gi + HORIZON <= mid), "test": ok & (gi >= mid),
                   "mid": gi == mid}
        sets = {name: ({f: P[f][m] for f in FEATURES}, (fwd_end if name == "mid" else fwd)[m])
                for name, m in rows_of.items()}
        print(f"  panel rows: train {rows_of['train'].sum()}, test {rows_of['test'].sum()}, "
              f"tickers at mid-period {rows_of['mid'].sum()} ({mid_day})", flush=True)

        results = search(sets)
        ok_cover = [r for r in results if r["train_cover"] >= MIN_COVER
                    and r["train_cover"] <= 1 - 1e-9]
        ok_cover.sort(key=lambda r: -r["train_mean_pass"])
        best_single, seen = [], set()
        for r in ok_cover:
            f = r["_rule"]["terms"][0][0]
            if r["kind"] == "single" and f not in seen:
                best_single.append(r)
                seen.add(f)
        best_pair, seen = [], set()
        for r in ok_cover:
            fs = tuple(t[0] for t in r["_rule"]["terms"])
            if r["kind"] == "pair" and fs not in seen:
                best_pair.append(r)
                seen.add(fs)
        ics = [{"feature": f,
                "train_ic": rank_ic(P[f][rows_of["train"]], fwd[rows_of["train"]],
                                    day[rows_of["train"]]),
                "test_ic": rank_ic(P[f][rows_of["test"]], fwd[rows_of["test"]],
                                   day[rows_of["test"]]),
                "mid_ic": rank_ic(P[f][rows_of["mid"]], fwd_end[rows_of["mid"]],
                                  day[rows_of["mid"]])} for f in FEATURES]

        gate_rules = ([BASE] + BENCH + MUST
                      + [r["_rule"] for r in best_single[:5] + best_pair[:5]])
        print(f"  gating {len(gate_rules)} rules", flush=True)
        gated = run_pool(pool, gate_worker, infos, "gated", t0, cost, gate_rules, mid_day)

    keys = [(r["name"], v) for r in gate_rules for v in variants_of(r)]
    port_rows = []
    for j, (name, variant) in enumerate(keys):
        sub = [g["rows"][j] for g in gated]
        ret = np.array([s["total_return"] for s in sub])
        half = np.array([s["second_half_return"] for s in sub])
        traded = np.array([s["trades"] > 0 for s in sub])
        row = {"rule": name, "variant": variant,
               **portfolio([g["days"] for g in gated], [g["daily"] for g in gated], j, mid_day),
               "mean_ticker_return": float(ret.mean()),
               "profitable": float((ret > 0).mean()), "losing": float((ret < 0).mean()),
               "traded": float(traded.mean()),
               "profitable_if_traded": float((ret[traded] > 0).mean()) if traded.any() else np.nan,
               "second_half_mean_ticker_return": float(half.mean()),
               "second_half_profitable": float((half > 0).mean()),
               "second_half_losing": float((half < 0).mean()),
               "time_in_market": float(np.mean([s["time_in_market"] for s in sub])),
               "trades_per_ticker": float(np.mean([s["trades"] for s in sub]))}
        port_rows.append(row)

    os.makedirs(args.out, exist_ok=True)
    panel_rows = [dict(ticker=t, day=int(d), **{f: P[f][i] for f in FEATURES},
                       fwd_return=fwd[i], fwd_return_to_end=fwd_end[i], bh_fwd_return=bh_fwd[i])
                  for i, (t, d) in enumerate(zip(ticker, day))]
    write_csv(os.path.join(args.out, "panel.csv.gz"), panel_rows)
    write_csv(os.path.join(args.out, "rules.csv"), results)
    write_csv(os.path.join(args.out, "rank_ic.csv"), ics)
    write_csv(os.path.join(args.out, "gated_ticker.csv"),
              [row for g in gated for row in g["rows"]])
    write_csv(os.path.join(args.out, "gated_portfolio.csv"), port_rows)
    config = {"run": args.run, "data": data, "cost_bps": cfg["cost_bps"],
              "step": cfg["step"], "tickers": len(infos), "horizon_days": HORIZON,
              "min_cover": MIN_COVER, "trading_days": len(days),
              "first_day": int(days[0]), "mid_day": mid_day, "last_day": int(days[-1]),
              "train_rows": int(rows_of["train"].sum()),
              "test_rows": int(rows_of["test"].sum()),
              "rules_searched": len(results), "check_max_gap": diff,
              "run_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    path = write_report(args.out, config, results, best_single, best_pair, ics, port_rows)
    print(f"report: {path}")


# --------------------------------------------------------------------------
# Report

def pct(x, d=1):
    return "–" if x is None or x != x else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def stamp(d):
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def rule_table(rows):
    return table(
        ["rule", "train: pass", "train: mean 20-day return (pass / fail)",
         "train: profitable (pass / fail)", "test: pass",
         "test: mean 20-day return (pass / fail)", "test: profitable (pass / fail)",
         "mid: tickers pass", "mid → end: mean return (pass / fail)",
         "mid → end: profitable (pass / fail)"],
        [[f"`{r['rule']}`", pct(r["train_cover"], 0),
          f"{pct(r['train_mean_pass'], 2)} / {pct(r['train_mean_fail'], 2)}",
          f"{pct(r['train_pos_pass'], 0)} / {pct(r['train_pos_fail'], 0)}",
          pct(r["test_cover"], 0),
          f"{pct(r['test_mean_pass'], 2)} / {pct(r['test_mean_fail'], 2)}",
          f"{pct(r['test_pos_pass'], 0)} / {pct(r['test_pos_fail'], 0)}",
          r["mid_n_pass"],
          f"{pct(r['mid_mean_pass'], 1)} / {pct(r['mid_mean_fail'], 1)}",
          f"{pct(r['mid_pos_pass'], 0)} / {pct(r['mid_pos_fail'], 0)}"] for r in rows])


def write_report(out, cfg, results, best_single, best_pair, ics, port):
    by_name = {r["rule"]: r for r in results}
    base = next(r for r in port if r["rule"] == "no filter")
    L = []
    w = L.append
    w("# Thresholds on observed skew and time in market\n")
    w("Generated by `thresholds.py`; the request is in `PROMPT.md`.\n")
    w("## Setup\n")
    w(f"- **Strategy:** the three combinations of `{cfg['run']}` "
      "(q25/q90 · high, open, close · 1560 bars · kernel) on the same "
      f"{cfg['tickers']} tickers, {cfg['cost_bps']:g} bps per trade. The rerun "
      f"reproduces that run's returns (largest gap {cfg['check_max_gap']:.1e}).")
    w("- **Features**, each computed at every bar from the bars up to it:")
    w("  - `skew_d20`, `skew_dall`: skewness of daily close-to-close log returns "
      "over the last 20 days, or over all days so far.")
    w("  - `skew_h20`, `skew_hall`: skewness of hourly log returns within the "
      "session (overnight gaps left out) over the last 140 hours, or all so far.")
    w("  - `tim_all`, `tim_20d`: the share of bars the strategy has held the "
      "ticker since trading started, or over the last 1560 bars.")
    w("  - Others: `mom_5d`, `mom_20d` (price change over 390 / 1560 bars), "
      "`vol_20d` (annualized sd of the last 20 daily returns), `band_width` "
      "((fitted q90 − fitted q25) / close) and `strat_ret` (the strategy's "
      "return on the ticker so far).")
    w("  - Strategy features (`tim_*`, `band_width`, `strat_ret`) are averaged "
      "over the three price fields in the panel; the gated strategy uses each "
      "field's own values. Values are fractions: `tim_all < 0.5` means held "
      "less than 50% of the time.")
    w(f"- **Period split:** {cfg['trading_days']} trading days from "
      f"{stamp(cfg['first_day'])} to {stamp(cfg['last_day'])}, split at "
      f"{stamp(cfg['mid_day'])}.")
    w(f"  - **Train:** checkpoints whose {cfg['horizon_days']}-day outcome ends "
      f"by the split ({cfg['train_rows']:,} ticker-days). Thresholds are chosen "
      "here.")
    w(f"  - **Test:** checkpoints from the split on ({cfg['test_rows']:,} "
      "ticker-days).")
    w("  - **Mid:** each ticker at the split day, with the outcome from there "
      "to the end of the data.")
    w(f"- **Outcome:** the strategy's return over the next {cfg['horizon_days']} "
      "trading days, starting at the next open (mean of the three fields). "
      "\"Profitable\" means above 0; a ticker the strategy stays out of "
      "returns exactly 0.")
    w(f"- **Search:** {cfg['rules_searched']:,} rules: single thresholds at the "
      "train deciles of each feature, pairs of thresholds at the train "
      "30/50/70% points, and the natural cut points (0 for skew, momentum and "
      "strategy return, 50% for time in market). Searched rules must pass "
      f"at least {pct(cfg['min_cover'], 0)} of train rows and are ranked by "
      "the train mean return of the rows they pass.")
    w("")

    w("## Requested thresholds\n")
    w(rule_table([by_name[r["name"]] for r in MUST]))
    w("")
    w("## Best threshold per feature (chosen on train)\n")
    w(rule_table(best_single))
    w("")
    w("## Best pairs of thresholds (chosen on train)\n")
    w(rule_table(best_pair[:10]))
    w("")
    w("## Rank correlation with the outcome\n")
    w("Mean over checkpoint days of the cross-sectional rank correlation "
      "between a feature and the next 20-day strategy return (train, test), "
      "and at mid-period with the return to the end.\n")
    w(table(["feature", "train", "test", "mid → end"],
            [[f"`{r['feature']}`", num(r["train_ic"], 3), num(r["test_ic"], 3),
              num(r["mid_ic"], 3)] for r in ics]))
    w("")

    w("## Gated strategy\n")
    w("*entry*: buy only while the rule holds. *exit*: also sell as soon as it "
      "stops holding. The portfolio splits $10,000 across all tickers; a "
      "ticker the gate keeps out stays in cash. The second half starts at "
      f"{stamp(cfg['mid_day'])}, after the thresholds were chosen. *buy & hold* "
      "and the *random* gates, which open on a random 25% or 50% of days, are "
      "benchmarks: a filter only adds information if it beats a random gate "
      "with similar time in market. For `tim_* > x` gates, entry and exit "
      "give the same result: while the gated strategy holds, the unfiltered "
      "strategy holds too, so its time in market can only rise and the gate "
      "never closes during a holding.\n")
    w(table(["rule", "gate", "return", "Sharpe", "max DD", "2nd half return",
             "2nd half Sharpe", "tickers profitable", "losing", "traded",
             "profitable if traded", "2nd half profitable / losing",
             "in market", "trades"],
            [[f"`{r['rule']}`", r["variant"], pct(r["return"], 2), num(r["sharpe"]),
              pct(r["max_drawdown"]), pct(r["second_half_return"], 2),
              num(r["second_half_sharpe"]), pct(r["profitable"], 0),
              pct(r["losing"], 0), pct(r["traded"], 0),
              pct(r["profitable_if_traded"], 0),
              f"{pct(r['second_half_profitable'], 0)} / {pct(r['second_half_losing'], 0)}",
              pct(r["time_in_market"], 0), num(r["trades_per_ticker"], 1)]
             for r in sorted(port, key=lambda r: -r["second_half_return"])]))
    w(f"\nWithout a filter the portfolio returns {pct(base['return'], 2)} "
      f"({pct(base['second_half_return'], 2)} in the second half).\n")

    w("## Caveats\n")
    w("- **Many rules, one period:** the best train rules are the best of "
      "thousands, so their train numbers are optimistic. Only the test, mid "
      "and second-half columns are out of sample, and only in time: the "
      "tickers and the three combinations are the same.")
    w("- **Overlapping outcomes:** 20-day outcomes on consecutive days overlap, "
      "so the panel holds far fewer independent observations than rows.")
    w("- **One market regime:** about five months of data. A threshold that "
      "works here may only reflect what this period rewarded.")
    w("")
    w(f"Run time {cfg['run_seconds']:.0f} s.")
    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    main()
