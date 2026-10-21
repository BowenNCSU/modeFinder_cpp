"""Improvements of the newest strategies, on round 3's tickers.

Reruns the tickers of a finished simulate.py run (round 3: q25/q90 · high,
open, close · 1560 bars · kernel) and compares the three strategies of
STRATEGIES.md with new variants:

  1. a combined entry gate: buy when the plain strategy's time in market
     over the last 20 days is above 85.3% OR since the start below 50%;
  2. the kernel fit weighted by each bar's volume (``weights=`` of
     ``modefinder.ModeFinder``), for every strategy;
  3. other sell thresholds: the fitted 95% and 99% quantiles and the
     highest value of the previous 1560 bars, for every strategy;
  4. Bernstein quantiles for every strategy, with the support fixed at
     [lowest - 1/1560, highest + 1/1560] of each window;
  5. filters on the trend of the Bernstein mode, from leave-one-day-out
     fits over the last 20 days and from the recent mode trajectory.

Every strategy is run on each price field (high, open, close) on its own,
deciding at a bar's close and filling at the next open, as in simulate.py.

    python simulations/stock_mode_quantile/improvements.py \\
        --run simulations/stock_mode_quantile/results/run_20261019_top3_2000

The rolling fits are cached in <out>/cache (kernel, volume-weighted kernel
and Bernstein, about 6 GB for 2,000 tickers and ignored by git).  The
Bernstein fits are the slow part: about 3 hours on 14 cores.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import modefinder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim  # noqa: E402
import thresholds as th  # noqa: E402

PROBS = (0.25, 0.90, 0.95, 0.99)    # fitted at every bar, for every source
DELTA = 1 / sim.WINDOW              # Bernstein support: [min - DELTA, max + DELTA]
SOURCES = {"kernel": "kernel", "kernel_vol": "kernel, volume weights",
           "bern": "Bernstein"}
SELLS = ("q90", "q95", "q99", "max")
TIM20_MIN, TIMALL_MAX = 0.853, 0.5  # entry gates of strategies 1 and 2
TIM_MAX, STOP = 0.5, 0.06           # cap and stop of strategy 3
JACK_Q = (0.10, 0.90)               # band of the leave-one-day-out modes
TRAJ_LAG = 78                       # trajectory: the mode now, 1 and 2 days ago


# --------------------------------------------------------------------------
# Rolling fits

def fit_window(w, source, vol=None):
    """The ModeFinder fit of one window, or None if it cannot be fitted."""
    if not w.max() > w.min():
        return None
    try:
        if source == "kernel":
            return modefinder.ModeFinder(w, smooth_option="kernel")
        if source == "kernel_vol":
            if not vol.sum() > 0:
                return None
            return modefinder.ModeFinder(w, smooth_option="kernel", weights=vol)
        return modefinder.ModeFinder(w, fix_lower=w.min() - DELTA,
                                     fix_upper=w.max() + DELTA)
    except (ValueError, ArithmeticError):
        return None


def rolling_fit(values, source, vol=None):
    """q[t] = the PROBS quantiles of values[t-WINDOW+1 .. t] (and, for
    Bernstein, the mode in the last column), carried forward over failed
    fits."""
    n, W = len(values), sim.WINDOW
    q = np.full((n, len(PROBS) + (source == "bern")), np.nan)
    for t in range(W - 1, n):
        fit = fit_window(values[t - W + 1:t + 1], source,
                         None if vol is None else vol[t - W + 1:t + 1])
        if fit is None:
            continue
        q[t, :len(PROBS)] = fit.quantile(PROBS)
        if source == "bern":
            q[t, -1] = fit.mode
    return sim.ffill(q)


def jackknife_modes(values, ts):
    """Leave-one-day-out Bernstein modes at the last bar of each day: for
    the window of the last WINDOW bars, the mode with each of its days left
    out in turn.  Returns the day-end bars and a (days, 21) array, NaN-padded
    when a window covers fewer days."""
    W = sim.WINDOW
    day = ts // 1_000_000
    ends = th.day_ends(ts)
    ends = ends[ends >= W - 1]
    out = np.full((len(ends), 21), np.nan)
    for i, t in enumerate(ends):
        w, d = values[t - W + 1:t + 1], day[t - W + 1:t + 1]
        for j, u in enumerate(np.unique(d)[:21]):
            fit = fit_window(w[d != u], "bern")
            if fit is not None:
                out[i, j] = fit.mode
    return ends, out


def cache_path(cache, source, info):
    return os.path.join(cache, source, f"{info['i']:04d}_{info['ticker']}.npy")


def fit_worker(job):
    """One ticker and source: rolling fits of the three price fields (and
    for Bernstein, the close's leave-one-day-out modes), saved to the cache."""
    info, source, cache = job
    path = cache_path(cache, source, info)
    if os.path.exists(path):
        return info["ticker"], 0.0
    t0 = time.time()
    bars = sim.load_bars(info["path"])
    vol = np.nan_to_num(bars["volume"], nan=0.0) if source == "kernel_vol" else None
    q = np.stack([rolling_fit(bars[f], source, vol) for f in sim.FIELDS])
    tmp = path[:-4] + ".tmp.npy"
    np.save(tmp, q)
    if source == "bern":
        ends, jack = jackknife_modes(bars["close"], bars["ts"])
        np.save(path[:-4] + "_jack.npy", np.column_stack((ends, jack)))
    os.replace(tmp, path)
    return info["ticker"], time.time() - t0


# --------------------------------------------------------------------------
# Strategies

STRATEGIES = {
    "plain": "q25 / sell, no rule",
    "tim20": "1. tim_20d > 85.3%, entry only",
    "timall": "2. tim_all < 50%, entry only",
    "capstop": "3. cap + stop",
    "either": "new: tim_20d > 85.3% or tim_all < 50%, entry only",
}
FILTERS = {
    "none": "no filter",
    "band_guard": "band: no buy if falling, no sell if rising",
    "band_literal": "band: no sell if falling, no buy if rising",
    "traj_guard": "trajectory: no buy if falling, no sell if rising",
    "traj_literal": "trajectory: no sell if falling, no buy if rising",
}
BASE = ("kernel", "q90", "plain", "none")      # round 3


def variants():
    return [(src, sell, st, f) for src in SOURCES for sell in SELLS
            for st in STRATEGIES for f in FILTERS]


def prev_max(values):
    """The highest of values[t-WINDOW .. t-1] (fewer bars at the start)."""
    W, n = sim.WINDOW, len(values)
    out = np.full(n, np.nan)
    out[W - 1] = values[:W - 1].max()
    out[W:] = np.lib.stride_tricks.sliding_window_view(values[:-1], W).max(axis=1)
    return out


def trend_masks(mode, jack):
    """(falling, rising) at each bar, from the close's Bernstein mode.

    band:       the mode is below (above) the 10% (90%) quantile of the
                leave-one-day-out modes of the previous day's close
    trajectory: the mode fell (rose) over each of the last two days,
                mode[t] < mode[t - 78] < mode[t - 156]"""
    n = len(mode)
    ends, loo = jack[:, 0].astype(int), jack[:, 1:]
    lo, hi = np.full(len(ends) + 1, np.nan), np.full(len(ends) + 1, np.nan)
    for i, row in enumerate(loo):
        row = row[np.isfinite(row)]
        if len(row):
            lo[i], hi[i] = np.quantile(row, JACK_Q)
    k = np.searchsorted(ends, np.arange(n), side="left") - 1   # last day end < t
    blo, bhi = lo[k], hi[k]                                     # k = -1: NaN
    m1 = np.concatenate((np.full(TRAJ_LAG, np.nan), mode[:-TRAJ_LAG]))
    m2 = np.concatenate((np.full(2 * TRAJ_LAG, np.nan), mode[:-2 * TRAJ_LAG]))
    with np.errstate(invalid="ignore"):
        return {"band": (mode < blo, mode > bhi),
                "traj": ((mode < m1) & (m1 < m2), (mode > m1) & (m1 > m2))}


def blocks(filt, masks, n):
    """(no buy, no sell) at each bar for filter filt."""
    if filt == "none":
        z = np.zeros(n, bool)
        return z, z
    falling, rising = masks[filt.split("_")[0]]
    return (falling, rising) if filt.endswith("guard") else (rising, falling)


def gate_events(c, buy, sell, start, gate, no_buy, no_sell):
    """Orders of a strategy whose rules do not depend on its own path:
    +1 buy, -1 sell, 0 keep."""
    ok = ~np.isnan(buy) & ~np.isnan(sell)
    ev = np.zeros(len(c), np.int8)
    ev[ok & (c < buy) & gate & ~no_buy] = 1
    ev[ok & (c > sell) & ~no_sell] = -1
    ev[:start] = 0
    ev[-1] = 0
    return ev


def capstop_events(o, c, buy, sell, start, no_buy, no_sell):
    """stop_rules.rule_events with the cap and the stop, plus the filter's
    blocks on buys and on signal sells (the cap and the stop always apply)."""
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
        over = t - start + 1 >= w and in_window > TIM_MAX * w
        x = cl[t]
        if pos:
            if x > sl[t] and not ns[t]:
                pos = False
            elif over:
                pos = False
            elif x < entry * (1 - STOP):
                pos = False
                lock = entry * (1 - STOP)
        else:
            if lock and x >= lock:
                lock = 0.0
            if x < bl[t] and not over and not lock and not nb[t]:
                pos = True
                entry = op[t + 1]
        out[t] = 1 if pos else -1
    return np.array(out, np.int8)


def load_fits(cache, info):
    q = {src: np.load(cache_path(cache, src, info)) for src in SOURCES}
    jack = np.load(cache_path(cache, "bern", info)[:-4] + "_jack.npy")
    return q, jack


def ticker_worker(info, cache, cost):
    """Every variant on one ticker.  daily[j] is the equity at each day's
    close of variants()[j], averaged over the three price fields; the last
    row is buy and hold."""
    bars = sim.load_bars(info["path"])
    q, jack = load_fits(cache, info)
    ts, o, c = bars["ts"], bars["open"], bars["close"]
    n, start = len(c), sim.WINDOW - 1
    masks = trend_masks(q["bern"][sim.FIELDS.index("close")][:, -1], jack)
    pmax = [prev_max(bars[f]) for f in sim.FIELDS]
    sell_col = {"q90": 1, "q95": 2, "q99": 3}
    rows, daily, check = [], [], {}
    for src, sell_name in ((s, x) for s in SOURCES for x in SELLS):
        per_field = []
        for k, field in enumerate(sim.FIELDS):
            buy = q[src][k][:, 0]
            sell = pmax[k] if sell_name == "max" else q[src][k][:, sell_col[sell_name]]
            z = np.zeros(n, bool)
            ev0 = gate_events(c, buy, sell, start, ~z, z, z)
            eq0, expo0, _, _ = sim.equity_from_events(o, c, ev0, cost)
            f = th.combo_features(bars, np.column_stack((buy, sell)), start, expo0, eq0)
            with np.errstate(invalid="ignore"):
                gates = {"plain": ~z, "tim20": f["tim_20d"] > TIM20_MIN,
                         "timall": f["tim_all"] < TIMALL_MAX}
            gates["either"] = gates["tim20"] | gates["timall"]
            per_field.append((buy, sell, gates))
        for st in STRATEGIES:
            for filt in FILTERS:
                nb, ns = blocks(filt, masks, n)
                stats, curves = [], []
                for k, (buy, sell, gates) in enumerate(per_field):
                    if st == "capstop":
                        ev = capstop_events(o, c, buy, sell, start, nb, ns)
                    else:
                        ev = gate_events(c, buy, sell, start, gates[st], nb, ns)
                    eq, expo, trades, _ = sim.equity_from_events(o, c, ev, cost)
                    stats.append((eq[-1] / sim.CAPITAL - 1, expo[start:].mean(),
                                  trades[start:].sum()))
                    curves.append(sim.daily_equity(ts, eq, start)[1])
                    if src == "kernel" and sell_name == "q90" and filt == "none":
                        check[(st, sim.FIELDS[k])] = stats[-1][0]
                s = np.mean(stats, axis=0)
                rows.append({"ticker": info["ticker"], "kind": info["kind"], "source": src,
                             "sell": sell_name, "strategy": st, "filter": filt,
                             "total_return": s[0], "time_in_market": s[1], "trades": s[2]})
                daily.append(np.mean(curves, axis=0))
    eq, expo, trades, _ = sim.buy_and_hold(bars, start, cost)
    days, curve = sim.daily_equity(ts, eq, start)
    daily.append(curve)
    active = {f"{name}_{d}": float(m[start:].mean()) for name, pair in masks.items()
              for d, m in zip(("falling", "rising"), pair)}
    return {"ticker": info["ticker"], "rows": rows, "days": days,
            "daily": np.array(daily, np.float64), "check": check, "active": active,
            "bh_return": float(eq[-1] / sim.CAPITAL - 1)}


# --------------------------------------------------------------------------
# Summaries

def summary_rows(results, mid_day):
    keys = variants()
    days_list = [r["days"] for r in results]
    daily_list = [r["daily"] for r in results]
    rets = np.array([[x["total_return"] for x in r["rows"]] for r in results])
    tim = np.array([[x["time_in_market"] for x in r["rows"]] for r in results])
    trades = np.array([[x["trades"] for x in r["rows"]] for r in results])
    base = rets[:, keys.index(BASE)]
    bh = np.array([r["bh_return"] for r in results])
    out = []
    for j, (src, sell, st, filt) in enumerate(keys):
        half = np.array([d[j][-1] / d[j][max(np.searchsorted(r["days"], mid_day, "right") - 1, 0)] - 1
                         for r, d in zip(results, daily_list)])
        out.append({"source": src, "sell": sell, "strategy": st, "filter": filt,
                    **th.portfolio(days_list, daily_list, j, mid_day),
                    "mean_ticker_return": float(rets[:, j].mean()),
                    "median_ticker_return": float(np.median(rets[:, j])),
                    "profitable": float((rets[:, j] > 0).mean()),
                    "beat_round3": float((rets[:, j] > base + 1e-12).mean()),
                    "beat_bh": float((rets[:, j] > bh).mean()),
                    "second_half_profitable": float((half > 0).mean()),
                    "time_in_market": float(tim[:, j].mean()),
                    "trades_per_ticker": float(trades[:, j].mean())})
    out.append({"source": "-", "sell": "-", "strategy": "buy & hold", "filter": "-",
                **th.portfolio(days_list, daily_list, len(keys), mid_day),
                "mean_ticker_return": float(bh.mean()),
                "median_ticker_return": float(np.median(bh)),
                "profitable": float((bh > 0).mean()), "time_in_market": 1.0,
                "trades_per_ticker": 1.0})
    return out


def check_gaps(args, results):
    """Largest per-ticker gaps to round 3 (plain) and to the stop-rules run
    (cap + stop, which that run calls q25/q90 + tim + stop)."""
    with open(os.path.join(args.run, "per_ticker.csv"), newline="") as fh:
        r3 = {(r["ticker"], r["field"]): float(r["total_return"])
              for r in csv.DictReader(fh) if r["combo"] != "buy & hold"}
    gaps = {"round3": max(abs(r["check"][("plain", f)] - r3[(r["ticker"], f)])
                          for r in results for f in sim.FIELDS)}
    path = os.path.join(args.run + "_stop_rules", "per_ticker.csv")
    if os.path.exists(path):
        with open(path, newline="") as fh:
            sr = {(r["ticker"], r["field"]): float(r["total_return"])
                  for r in csv.DictReader(fh) if r["strategy"] == "q25/q90 + tim + stop"}
        gaps["stop_rules"] = max(abs(r["check"][("capstop", f)] - sr[(r["ticker"], f)])
                                 for r in results for f in sim.FIELDS)
    return gaps


# --------------------------------------------------------------------------
# Report

SELL_NAMES = {"q90": "q90", "q95": "q95", "q99": "q99", "max": "max of last 1560"}


def write_report(out):
    pct, num, table, stamp = th.pct, th.num, th.table, th.stamp
    cfg = json.load(open(os.path.join(out, "config.json")))
    with open(os.path.join(out, "portfolio.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if k not in ("source", "sell", "strategy", "filter"):
                r[k] = float(v) if v not in ("", None) else np.nan
    by = {(r["source"], r["sell"], r["strategy"], r["filter"]): r for r in rows}
    bh = by[("-", "-", "buy & hold", "-")]
    L = []
    w = L.append

    def ret2(r):
        return f"{pct(r['return'], 2)} / {pct(r['second_half_return'], 2)}"

    def name(key):
        src, sell, st, filt = key
        f = "" if filt == "none" else f" · {FILTERS[filt]}"
        return f"{STRATEGIES[st]} · {SOURCES[src]} · sell {SELL_NAMES[sell]}{f}"

    def full_rows(keys):
        return [[k if isinstance(k, str) else name(k), pct(by[k]["return"], 2),
                 num(by[k]["sharpe"]), pct(by[k]["max_drawdown"]),
                 pct(by[k]["second_half_return"], 2), pct(by[k]["profitable"], 0),
                 pct(by[k]["beat_round3"], 0), pct(by[k]["time_in_market"], 0),
                 num(by[k]["trades_per_ticker"], 1)] for k in keys]
    full_head = ["strategy", "return", "Sharpe", "max DD", "2nd half return",
                 "tickers profitable", "beat round 3", "in market", "trades"]

    w("# Improvements of the newest strategies\n")
    w("Generated by `improvements.py`; the request is in `PROMPT.md`.\n")
    w("## Setup\n")
    w(f"- **Tickers:** the {cfg['tickers']} stocks and ETFs of round 3 "
      f"(`{cfg['run']}`), {cfg['cost_bps']:g} bps per buy or sell, $10,000 per "
      f"ticker. {cfg['trading_days']} trading days from {stamp(cfg['first_day'])} "
      f"to {stamp(cfg['last_day'])}; returns are totals, not annualized. The "
      f"second half starts at the close of {stamp(cfg['mid_day'])}.")
    w("- **Fits:** at every bar, the last 1560 bars of each price field (high, "
      "open, close), run on its own; tables use the mean of the three fields.")
    w("  - **kernel:** as in round 3.")
    w("  - **kernel, volume weights:** each bar's volume is its weight, divided by "
      "the window's total volume; each bar's gaussian kernel enters the density "
      "and the CDF with that weight. The bandwidth is unchanged.")
    w(f"  - **Bernstein:** the support is fixed at [lowest − 1/1560, highest + "
      f"1/1560] of the window ({cfg['bernstein_delta']:.6f} on each side), with "
      "the degree chosen automatically.")
    w("- **Strategies:** all buy below the fitted 25% quantile.")
    w("  - **q25 / sell, no rule:** round 3 when the fit is the kernel and the "
      "sell is q90.")
    w(f"  - **1. tim_20d > {pct(cfg['tim20_min'])}, entry only** and "
      f"**2. tim_all < {pct(cfg['timall_max'], 0)}, entry only:** buy only while "
      "the plain strategy (same fit and sell threshold) has held the ticker more "
      "than 85.3% of the last 1560 bars, or less than 50% of all bars since "
      "trading started.")
    w("  - **new: either gate:** buy while gate 1 OR gate 2 holds.")
    w(f"  - **3. cap + stop:** sell, and do not buy, while the strategy has held "
      f"the ticker more than {pct(cfg['tim_max'], 0)} of the last 1560 bars (after "
      f"1560 bars of trading); sell {pct(cfg['stop'], 0)} below the buy price and "
      "stay out until the close is back above that level.")
    w("- **Sell thresholds:** the fitted 90%, 95% and 99% quantiles, and *max*: "
      "the highest value of the price field over the previous 1560 bars, not "
      "counting the current bar (with it, the close could never be above it).")
    w("- **Bernstein mode filters:** from the close's Bernstein fits.")
    w(f"  - **band:** at each day's close, the mode is refitted 20 times, leaving "
      f"out one of the window's days each time. The mode is *falling* when it is "
      f"below the {pct(cfg['jack_q'][0], 0)} quantile of the previous day's 20 "
      f"leave-one-day-out modes, and *rising* when it is above their "
      f"{pct(cfg['jack_q'][1], 0)} quantile.")
    w("  - **trajectory:** the mode is *falling* when it fell over each of the "
      "last two days (now < 1 day ago < 2 days ago, 78 bars a day), and "
      "*rising* when it rose over both.")
    w("  - **Two readings.** *no buy if falling, no sell if rising* is the usual "
      "guard for this strategy: its buys fire during drops and its sells "
      "during rallies. *no sell if falling, no buy if rising* is the request "
      "as written. Both block only buys and signal sells; the cap and the stop "
      "always apply.")
    act = cfg["filter_active"]
    w(f"  - **How often they are on** (share of bars, mean over tickers): band "
      f"falling {pct(act['band_falling'], 0)}, rising {pct(act['band_rising'], 0)}; "
      f"trajectory falling {pct(act['traj_falling'], 0)}, rising "
      f"{pct(act['traj_rising'], 0)}.")
    g = cfg["check_gaps"]
    w("- **Checks:** the plain kernel q90 strategy reproduces round 3 (largest "
      f"per-ticker gap {g['round3']:.1e}), and kernel q90 cap + stop reproduces "
      f"the stop-rules run ({g.get('stop_rules', float('nan')):.1e}).")
    w("")

    w("## 1. The three strategies and the combined gate\n")
    w("Kernel fits, sell above q90, as in `STRATEGIES.md`.\n")
    keys = [("kernel", "q90", st, "none") for st in ("either", "tim20", "timall",
                                                       "capstop", "plain")]
    w(table(full_head, full_rows(keys) + [["buy & hold", pct(bh["return"], 2),
                                           num(bh["sharpe"]), pct(bh["max_drawdown"]),
                                           pct(bh["second_half_return"], 2),
                                           pct(bh["profitable"], 0), "–", "100%", "1.0"]]))
    w("")

    w("## 2. Volume-weighted kernel\n")
    w("Return / second-half return, sell above q90, no filter.\n")
    w(table(["strategy"] + [SOURCES[s] for s in SOURCES],
            [[STRATEGIES[st]] + [ret2(by[(s, "q90", st, "none")]) for s in SOURCES]
             for st in STRATEGIES]))
    w("")

    w("## 3. Sell thresholds, for each fit\n")
    w("Return / second-half return, no filter.\n")
    for src in SOURCES:
        w(f"### {SOURCES[src]}\n")
        w(table(["strategy"] + [f"sell {SELL_NAMES[x]}" for x in SELLS],
                [[STRATEGIES[st]] + [ret2(by[(src, x, st, "none")]) for x in SELLS]
                 for st in STRATEGIES]))
        w("")
        w(table(["strategy"] + [f"in market, sell {x}" for x in SELLS]
                + [f"trades, sell {x}" for x in SELLS],
                [[STRATEGIES[st]] + [pct(by[(src, x, st, "none")]["time_in_market"], 0)
                                     for x in SELLS]
                 + [num(by[(src, x, st, "none")]["trades_per_ticker"], 1) for x in SELLS]
                 for st in STRATEGIES]))
        w("")

    w("## 4. Bernstein mode filters\n")
    for src in ("bern", "kernel"):
        w(f"### Quantiles from {SOURCES[src]}, sell above q90\n")
        w("Return / second-half return.\n")
        w(table(["strategy"] + [FILTERS[f] for f in FILTERS],
                [[STRATEGIES[st]] + [ret2(by[(src, "q90", st, f)]) for f in FILTERS]
                 for st in STRATEGIES]))
        w("")
    w("Mean over every fit, sell threshold and strategy (60 combinations each):\n")
    w(table(["filter", "mean return", "mean 2nd half return", "better than no filter",
             "mean in market"],
            [[FILTERS[f],
              pct(np.mean([r["return"] for k, r in by.items() if k[3] == f]), 2),
              pct(np.mean([r["second_half_return"] for k, r in by.items() if k[3] == f]), 2),
              pct(np.mean([by[k]["return"] > by[k[:3] + ("none",)]["return"]
                           for k in by if k[3] == f]), 0) if f != "none" else "–",
              pct(np.mean([r["time_in_market"] for k, r in by.items() if k[3] == f]), 0)]
             for f in FILTERS]))
    w("")

    combos = [k for k in by if k[0] != "-"]
    w("## 5. Best combinations\n")
    w(f"All {len(combos)} combinations of fit × sell threshold × strategy × filter.\n")
    w("### By return\n")
    w(table(full_head, full_rows(sorted(combos, key=lambda k: -by[k]["return"])[:20])))
    w("\n### By second-half return\n")
    w(table(full_head, full_rows(sorted(combos,
                                        key=lambda k: -by[k]["second_half_return"])[:15])))
    w("\n### By Sharpe\n")
    w(table(full_head, full_rows(sorted(combos, key=lambda k: -by[k]["sharpe"])[:15])))
    w("\nEvery combination is in `portfolio.csv`, and each ticker's results "
      "(mean of the three fields) in `per_ticker.csv.gz`.\n")

    w("## Caveats\n")
    w("- **One period, one market:** about five months, with a falling second "
      "half that favours any rule holding more cash.")
    w("- **Many combinations:** with this many tried on the same tickers and "
      "period, the best ones are partly luck. The 85.3% gate was itself chosen "
      "on the first half.")
    w("- **Max sell:** selling only at a new 20-day high often holds for weeks; "
      "it behaves more like buy and hold with a dip entry.")
    w("")
    w(f"Run time {cfg['run_seconds']:.0f} s.")
    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------
# Main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="a finished simulate.py run directory")
    p.add_argument("--data", default=None, help="price files (default: the run's --data)")
    p.add_argument("--out", default=None, help="output directory (default <run>_improvements)")
    p.add_argument("--cache", default=None, help="rolling fits (default <out>/cache)")
    p.add_argument("--fits-only", action="store_true", help="only fill the cache")
    p.add_argument("--report-only", action="store_true",
                   help="rebuild report.md from portfolio.csv and config.json")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    args.run = args.run.rstrip("/")
    args.out = args.out or args.run + "_improvements"
    args.cache = args.cache or os.path.join(args.out, "cache")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.report_only:
        print(f"report: {write_report(args.out)}")
        return
    t0 = time.time()
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    data = args.data or cfg["data"]
    cost = cfg["cost_bps"] / 1e4
    with open(os.path.join(args.run, "selection.csv"), newline="") as fh:
        sel = list(csv.DictReader(fh))
    infos = [{"i": i, "ticker": s["ticker"], "kind": s["kind"],
              "path": os.path.join(data, s["path"])} for i, s in enumerate(sel)]
    for source in SOURCES:
        os.makedirs(os.path.join(args.cache, source), exist_ok=True)
    print(f"{len(infos)} tickers from {args.run}", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for source in SOURCES:
            th.run_pool(pool, fit_worker, [(f, source, args.cache) for f in infos],
                        f"fits {source}", t0)
        if args.fits_only:
            return
        results = th.run_pool(pool, ticker_worker, infos, "backtests", t0,
                              args.cache, cost)
    results.sort(key=lambda r: r["ticker"])
    gaps = check_gaps(args, results)
    print(f"  largest gaps to earlier runs: {gaps}", flush=True)
    days = np.unique(np.concatenate([r["days"] for r in results]))
    mid_day = int(days[len(days) // 2])

    os.makedirs(args.out, exist_ok=True)
    summary = summary_rows(results, mid_day)
    th.write_csv(os.path.join(args.out, "portfolio.csv"), summary)
    th.write_csv(os.path.join(args.out, "per_ticker.csv.gz"),
                 [row for r in results for row in r["rows"]])
    active = {k: float(np.mean([r["active"][k] for r in results]))
              for k in results[0]["active"]}
    config = {"run": args.run, "data": data, "cost_bps": cfg["cost_bps"],
              "tickers": len(infos), "window": sim.WINDOW, "fields": sim.FIELDS,
              "probs": PROBS, "bernstein_delta": DELTA, "sources": SOURCES,
              "sells": SELLS, "strategies": STRATEGIES, "filters": FILTERS,
              "tim20_min": TIM20_MIN, "timall_max": TIMALL_MAX, "tim_max": TIM_MAX,
              "stop": STOP, "jack_q": JACK_Q, "traj_lag": TRAJ_LAG,
              "filter_active": active, "check_gaps": gaps,
              "trading_days": len(days), "first_day": int(days[0]),
              "mid_day": mid_day, "last_day": int(days[-1]),
              "modefinder_version": modefinder.__version__,
              "run_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    path = write_report(args.out)
    print(f"report: {path}")


if __name__ == "__main__":
    main()
