"""q05/q95 strategy with a time-in-market cap and a 6% stop loss.

Reruns the tickers of a finished simulate.py run (round 3: q25/q90 · high,
open, close · 1560 bars · kernel) with a new strategy.  At every bar the
last 1560 bars (about 20 trading days) of the price field are fitted with
``modefinder.ModeFinder(..., smooth_option="kernel")`` and the strategy

    buys  (all in)  when the close is below the fitted 5% quantile,
    sells (all out) when the close is above the fitted 95% quantile,

with two extra rules:

    time in market  once the strategy has traded for 1560 bars (20 days),
                    it sells, and does not buy, as long as it has held the
                    ticker more than 50% of the last 1560 bars
    stop loss       it sells when the close is more than 6% below the
                    price it bought at, and does not buy again as long as
                    the close stays below that stop level

Orders are decided at a bar's close and filled at the next open, as in
simulate.py.  To show where any difference comes from, the same run also
backtests q05/q95 with each rule on its own, q25/q90 with both rules, a
variant whose stop allows buying again at once, and round 3's q25/q90
strategy, which must reproduce the earlier run.

    python simulations/stock_mode_quantile/stop_rules.py \\
        --run simulations/stock_mode_quantile/results/run_20261019_top3_2000

The rolling fits (the slow part, about 20 minutes for 2,000 tickers on 14
cores) are cached in <out>/cache, so a rerun only redoes the backtests.
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate as sim  # noqa: E402
import thresholds as th  # noqa: E402

PROBS = (0.05, 0.25, 0.90, 0.95)   # fitted once per bar, for every strategy
TIM_MAX = 0.5                       # largest share of the last WINDOW bars held
STOP = 0.06                         # stop loss below the buy price
NEW = "q05/q95 + tim + stop"
OLD = "q25/q90"


def strategy(name, buy, sell, tim=False, stop=False, rebuy=False):
    return dict(name=name, buy=PROBS.index(buy), sell=PROBS.index(sell),
                tim=TIM_MAX if tim else None, stop=STOP if stop else None,
                rebuy=rebuy)


STRATEGIES = [
    strategy(NEW, 0.05, 0.95, tim=True, stop=True),
    strategy("q05/q95", 0.05, 0.95),
    strategy("q05/q95 + tim", 0.05, 0.95, tim=True),
    strategy("q05/q95 + stop", 0.05, 0.95, stop=True),
    strategy("q05/q95 + tim + stop, rebuy at once", 0.05, 0.95, tim=True, stop=True,
             rebuy=True),
    strategy("q25/q90 + tim + stop", 0.25, 0.90, tim=True, stop=True),
    strategy(OLD, 0.25, 0.90),
]


# --------------------------------------------------------------------------
# Backtest

def rule_events(o, c, buy, sell, start, tim=None, stop=None, rebuy=False):
    """Events for sim.equity_from_events (1: hold after bar t, -1: flat) and
    the number of exits for each reason.  The time-in-market and stop rules
    depend on the strategy's own path, so this walks the bars one by one."""
    n, w = len(c), sim.WINDOW
    cl, op, bl, sl = c.tolist(), o.tolist(), buy.tolist(), sell.tolist()
    held = [False] * n          # held over interval t, decided at t - 1
    out = [0] * n
    exits = {"signal": 0, "tim": 0, "stop": 0}
    pos, in_window, entry, lock = False, 0, 0.0, 0.0
    for t in range(start, n - 1):
        held[t] = pos
        in_window += pos
        if t >= w:
            in_window -= held[t - w]
        over = tim is not None and t - start + 1 >= w and in_window > tim * w
        x = cl[t]
        if pos:
            if x > sl[t]:
                pos = False
                exits["signal"] += 1
            elif over:
                pos = False
                exits["tim"] += 1
            elif stop is not None and x < entry * (1 - stop):
                pos = False
                exits["stop"] += 1
                if not rebuy:
                    lock = entry * (1 - stop)
        else:
            if lock and x >= lock:
                lock = 0.0
            if x < bl[t] and not over and not lock:
                pos = True
                entry = op[t + 1]
        out[t] = 1 if pos else -1
    return np.array(out, np.int8), exits


def fit_worker(info, step):
    """Rolling PROBS quantiles of the three price fields, saved to the cache."""
    if os.path.exists(info["cache"]):
        return info["ticker"], 0.0
    t0 = time.time()
    bars = sim.load_bars(info["path"])
    q = np.stack([sim.rolling_quantiles(bars[f], step, PROBS) for f in sim.FIELDS])
    tmp = info["cache"][:-4] + ".tmp.npy"
    np.save(tmp, q)
    os.replace(tmp, info["cache"])
    return info["ticker"], time.time() - t0


def ticker_worker(info, cost):
    """Every strategy on the three fields, and buy and hold, on one ticker.
    daily[j] is the equity at each day's close for keys()[j]."""
    bars = sim.load_bars(info["path"])
    q = np.load(info["cache"])
    ts, o, c = bars["ts"], bars["open"], bars["close"]
    start = sim.WINDOW - 1
    bpd = sim.bars_per_day(ts)
    base = dict(ticker=info["ticker"], kind=info["kind"])
    rows, daily = [], []
    eq, expo, trades, gross = sim.buy_and_hold(bars, start, cost)
    rows.append(dict(base, strategy="buy & hold", field="-",
                     **sim.metrics(eq, expo, trades, gross, start, cost, bpd),
                     exits_signal=0, exits_tim=0, exits_stop=0))
    days, curve = sim.daily_equity(ts, eq, start)
    daily.append(curve)
    for s in STRATEGIES:
        curves = []
        for k, field in enumerate(sim.FIELDS):
            ev, exits = rule_events(o, c, q[k][:, s["buy"]], q[k][:, s["sell"]], start,
                                    s["tim"], s["stop"], s["rebuy"])
            eq, expo, trades, gross = sim.equity_from_events(o, c, ev, cost)
            rows.append(dict(base, strategy=s["name"], field=field,
                             **sim.metrics(eq, expo, trades, gross, start, cost, bpd),
                             **{f"exits_{why}": v for why, v in exits.items()}))
            curves.append(sim.daily_equity(ts, eq, start)[1])
        daily.extend(curves)
        daily.append(np.mean(curves, axis=0))
    return {"ticker": info["ticker"], "rows": rows, "days": days, "daily": np.array(daily)}


def keys():
    """(strategy, field) of each daily curve; field "mean" is the mean of
    the three fields' curves."""
    out = [("buy & hold", "-")]
    for s in STRATEGIES:
        out += [(s["name"], f) for f in sim.FIELDS + ("mean",)]
    return out


# --------------------------------------------------------------------------
# Summaries

def ticker_returns(results, j, mid_day):
    """Each ticker's total and second-half return for curve j."""
    total, half = [], []
    for r in results:
        d = r["daily"][j]
        k = np.searchsorted(r["days"], mid_day, side="right") - 1
        total.append(d[-1] / sim.CAPITAL - 1)
        half.append(d[-1] / (d[k] if k >= 0 else sim.CAPITAL) - 1)
    return np.array(total), np.array(half)


def summary_rows(results, mid_day):
    rows_by = {}
    for r in results:
        for row in r["rows"]:
            rows_by.setdefault((row["strategy"], row["field"]), []).append(row)
    bh_j = keys().index(("buy & hold", "-"))
    old_j = keys().index((OLD, "mean"))
    bh_ret, _ = ticker_returns(results, bh_j, mid_day)
    old_ret, _ = ticker_returns(results, old_j, mid_day)
    out = []
    for j, (name, field) in enumerate(keys()):
        sub = [x for f in (sim.FIELDS if field == "mean" else (field,))
               for x in rows_by[(name, f)]]
        ret, half = ticker_returns(results, j, mid_day)
        row = {"strategy": name, "field": field,
               **th.portfolio([r["days"] for r in results],
                              [r["daily"] for r in results], j, mid_day),
               "mean_ticker_return": float(ret.mean()),
               "median_ticker_return": float(np.median(ret)),
               "profitable": float((ret > 0).mean()), "losing": float((ret < 0).mean()),
               "beat_bh": float((ret > bh_ret).mean()),
               "beat_round3": float((ret > old_ret + 1e-12).mean()),
               "second_half_profitable": float((half > 0).mean()),
               "time_in_market": float(np.mean([x["time_in_market"] for x in sub])),
               "trades_per_ticker": float(np.mean([x["trades"] for x in sub])),
               "win_rate": float(np.nanmean([x["win_rate"] for x in sub]))
               if any(x["win_rate"] == x["win_rate"] for x in sub) else np.nan}
        for reason in ("signal", "tim", "stop"):
            row[f"exits_{reason}"] = float(np.mean([x[f"exits_{reason}"] for x in sub]))
        out.append(row)
    return out


def ranking_rows(results, mid_day):
    """One row per ticker, ranked by the new strategy's return (mean of the
    three fields), next to round 3 and buy and hold."""
    j_new, j_old = keys().index((NEW, "mean")), keys().index((OLD, "mean"))
    j_bh = keys().index(("buy & hold", "-"))
    new, new_half = ticker_returns(results, j_new, mid_day)
    old, _ = ticker_returns(results, j_old, mid_day)
    bh, _ = ticker_returns(results, j_bh, mid_day)
    out = []
    for i, r in enumerate(results):
        by = {(x["strategy"], x["field"]): x for x in r["rows"]}
        mine = [by[(NEW, f)] for f in sim.FIELDS]
        out.append(dict(
            ticker=r["ticker"], kind=r["rows"][0]["kind"], return_new=new[i],
            **{f"return_{f}": by[(NEW, f)]["total_return"] for f in sim.FIELDS},
            second_half_new=new_half[i], return_round3=old[i], bh_return=bh[i],
            new_minus_round3=new[i] - old[i], new_minus_bh=new[i] - bh[i],
            time_in_market=float(np.mean([x["time_in_market"] for x in mine])),
            trades=float(np.mean([x["trades"] for x in mine])),
            stop_exits=float(np.mean([x["exits_stop"] for x in mine])),
            tim_exits=float(np.mean([x["exits_tim"] for x in mine])),
            max_drawdown=float(np.mean([x["max_drawdown"] for x in mine]))))
    out.sort(key=lambda x: -x["return_new"])
    return [dict(rank=i, **row) for i, row in enumerate(out, 1)]


# --------------------------------------------------------------------------
# Main

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="a finished simulate.py run directory")
    p.add_argument("--data", default=None, help="price files (default: the run's --data)")
    p.add_argument("--out", default=None, help="output directory (default <run>_stop_rules)")
    p.add_argument("--cache", default=None, help="fitted quantiles (default <out>/cache)")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    args.run = args.run.rstrip("/")
    args.out = args.out or args.run + "_stop_rules"
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
    infos = [{"ticker": s["ticker"], "kind": s["kind"],
              "path": os.path.join(data, s["path"]),
              "cache": os.path.join(args.cache, f"{i:04d}_{s['ticker']}.npy")}
             for i, s in enumerate(sel)]
    print(f"{len(infos)} tickers from {args.run}", flush=True)

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        th.run_pool(pool, fit_worker, infos, "fits", t0, cfg["step"])
        results = th.run_pool(pool, ticker_worker, infos, "backtests", t0, cost)
    results.sort(key=lambda r: r["ticker"])

    diff = max(abs(x["total_return"] - ref[(x["ticker"], x["field"])])
               for r in results for x in r["rows"] if x["strategy"] in (OLD, "buy & hold"))
    print(f"  largest gap to the run's per_ticker.csv returns: {diff:.2e}", flush=True)
    days = np.unique(np.concatenate([r["days"] for r in results]))
    mid_day = int(days[len(days) // 2])

    os.makedirs(args.out, exist_ok=True)
    summary = summary_rows(results, mid_day)
    ranking = ranking_rows(results, mid_day)
    th.write_csv(os.path.join(args.out, "per_ticker.csv"),
                 [row for r in results for row in r["rows"]])
    th.write_csv(os.path.join(args.out, "ranking.csv"), ranking)
    th.write_csv(os.path.join(args.out, "portfolio.csv"), summary)
    config = {"run": args.run, "data": data, "cost_bps": cfg["cost_bps"],
              "step": cfg["step"], "tickers": len(infos), "window": sim.WINDOW,
              "method": sim.METHOD, "fields": sim.FIELDS, "probs": PROBS,
              "tim_max": TIM_MAX, "stop": STOP,
              "strategies": [{k: v for k, v in s.items()} for s in STRATEGIES],
              "trading_days": len(days), "first_day": int(days[0]),
              "mid_day": mid_day, "last_day": int(days[-1]), "check_max_gap": diff,
              "run_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    path = write_report(args.out, config, summary, ranking)
    print(f"report: {path}")


# --------------------------------------------------------------------------
# Report

def write_report(out, cfg, summary, ranking):
    pct, num, table, stamp = th.pct, th.num, th.table, th.stamp
    mean = [r for r in summary if r["field"] in ("mean", "-")]
    by = {(r["strategy"], r["field"]): r for r in summary}
    new, old = by[(NEW, "mean")], by[(OLD, "mean")]
    L = []
    w = L.append
    w("# q05/q95 with a time-in-market cap and a stop loss\n")
    w("Generated by `stop_rules.py`; the request is in `PROMPT.md`.\n")
    w("## Setup\n")
    w(f"- **Tickers:** the {cfg['tickers']} stocks and ETFs of round 3 "
      f"(`{cfg['run']}`), {cfg['cost_bps']:g} bps per buy or sell, $10,000 per "
      f"ticker. {cfg['trading_days']} trading days from {stamp(cfg['first_day'])} "
      f"to {stamp(cfg['last_day'])}; returns are totals over this period, not "
      "annualized.")
    w(f"- **New strategy** (`{NEW}`): at every bar the last {cfg['window']} bars "
      "(20 trading days) of the price field are fitted with the kernel method.")
    w("  - **Buy:** all in when the close is below the fitted 5% quantile.")
    w("  - **Sell:** all out when the close is above the fitted 95% quantile.")
    w(f"  - **Time in market:** once the strategy has traded for {cfg['window']} "
      "bars (20 days), it sells, and does not buy, as long as it held the ticker "
      f"more than {pct(cfg['tim_max'], 0)} of the last {cfg['window']} bars.")
    w(f"  - **Stop loss:** it sells when the close is more than {pct(cfg['stop'], 0)} "
      "below its buy price, and does not buy again as long as the close stays "
      "below that stop level.")
    w("  - Orders are decided at a bar's close and filled at the next open. "
      "Each price field (high, open, close) is fitted and run on its own; the "
      "tables use the mean of the three.")
    w("- **Comparisons:** each rule on its own, a stop that allows buying again "
      "at once, q25/q90 with both rules, round 3's q25/q90, and buy and hold. "
      "The rerun of round 3 reproduces its returns (largest gap "
      f"{cfg['check_max_gap']:.1e}).")
    w(f"- **Second half:** from the close of {stamp(cfg['mid_day'])}, the split "
      "used in the threshold study.")
    w("")

    w("## Portfolio\n")
    w("$10,000 split equally across the tickers and the three price fields.\n")
    w(table(["strategy", "return", "Sharpe", "max DD", "2nd half return",
             "tickers profitable / losing", "beat round 3", "beat B&H",
             "median ticker", "in market", "trades", "exits: signal / tim / stop"],
            [[f"**{r['strategy']}**" if r["strategy"] == NEW else r["strategy"],
              pct(r["return"], 2), num(r["sharpe"]), pct(r["max_drawdown"]),
              pct(r["second_half_return"], 2),
              f"{pct(r['profitable'], 0)} / {pct(r['losing'], 0)}",
              "–" if r["strategy"] == OLD else pct(r["beat_round3"], 0),
              "–" if r["strategy"] == "buy & hold" else pct(r["beat_bh"], 0),
              pct(r["median_ticker_return"], 2), pct(r["time_in_market"], 0),
              num(r["trades_per_ticker"], 1),
              f"{num(r['exits_signal'], 1)} / {num(r['exits_tim'], 1)} / "
              f"{num(r['exits_stop'], 1)}"]
             for r in sorted(mean, key=lambda r: -r["return"])]))
    w("\n*beat round 3*: share of tickers where the strategy returns more than "
      "round 3's q25/q90. *exits*: mean sells per ticker and field, by reason.\n")

    w("## By price field\n")
    w(table(["strategy"] + [f"{f}: return / 2nd half" for f in sim.FIELDS],
            [[s["name"]] + [f"{pct(by[(s['name'], f)]['return'], 2)} / "
                            f"{pct(by[(s['name'], f)]['second_half_return'], 2)}"
                            for f in sim.FIELDS] for s in STRATEGIES]))
    w("")

    gated = os.path.join(cfg["run"] + "_thresholds", "gated_portfolio.csv")
    if os.path.exists(gated):
        with open(gated, newline="") as fh:
            g = {(r["rule"], r["variant"]): r for r in csv.DictReader(fh)}
        pick = [("no filter", "-"), ("tim_all < 0.5", "entry"), ("tim_all < 0.5", "exit"),
                ("skew_d20 > 0 and tim_all < 0.5", "entry"), ("tim_20d > 0.853", "entry"),
                ("random 50% of days", "exit")]
        w("## Earlier gated strategies\n")
        w("From the threshold study (`" + os.path.relpath(gated, out) + "`): round 3's "
          "q25/q90 with a gate on observed features, same tickers and portfolio. "
          "*entry*: buy only while the rule holds; *exit*: also sell when it stops "
          "holding. There, `tim` is the unfiltered strategy's time in market, not "
          "the gated one's.\n")
        w(table(["gate", "variant", "return", "Sharpe", "max DD", "2nd half return",
                 "in market"],
                [[f"`{k[0]}`", k[1], pct(float(g[k]["return"]), 2),
                  num(float(g[k]["sharpe"])), pct(float(g[k]["max_drawdown"])),
                  pct(float(g[k]["second_half_return"]), 2),
                  pct(float(g[k]["time_in_market"]), 0)] for k in pick if k in g]))
        w("")

    d = np.array([r["new_minus_round3"] for r in ranking])
    w("## New strategy against round 3, per ticker\n")
    w(f"- The new strategy returns more than round 3 on {pct((d > 1e-12).mean(), 0)} "
      f"of tickers, less on {pct((d < -1e-12).mean(), 0)}; median difference "
      f"{pct(np.median(d), 2)}, mean {pct(d.mean(), 2)}.")
    w(f"- Portfolio: {pct(new['return'], 2)} against {pct(old['return'], 2)} "
      f"(second half {pct(new['second_half_return'], 2)} against "
      f"{pct(old['second_half_return'], 2)}); in market {pct(new['time_in_market'], 0)} "
      f"against {pct(old['time_in_market'], 0)}.")
    w("")
    cols = ["rank", "ticker", "kind", "new", "2nd half", "round 3", "B&H",
            "in market", "trades", "stop exits"]

    def rows(rs):
        return [[r["rank"], r["ticker"], r["kind"], pct(r["return_new"]),
                 pct(r["second_half_new"]), pct(r["return_round3"]), pct(r["bh_return"]),
                 pct(r["time_in_market"], 0), num(r["trades"], 1), num(r["stop_exits"], 1)]
                for r in rs]
    w("## Most profitable tickers, new strategy\n")
    w(table(cols, rows(ranking[:30])))
    w("\n## Least profitable tickers, new strategy\n")
    w(table(cols, rows(ranking[-15:])))
    w("\nThe full ranking is in `ranking.csv`.\n")

    w("## Caveats\n")
    w("- **One period:** about five months, and the market fell in the second "
      "half, which favours any rule that holds more cash.")
    w("- **Rules chosen after the fact:** the 6% stop and the 50% cap were set "
      "after seeing round 3 and the threshold study on the same tickers and "
      "period, so this is not an out-of-sample test.")
    w("- **Fills:** a stop sells at the next open, which can be well below the "
      "stop level after a gap.")
    w("")
    w(f"Run time {cfg['run_seconds']:.0f} s.")
    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


if __name__ == "__main__":
    main()
