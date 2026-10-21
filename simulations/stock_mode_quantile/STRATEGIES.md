# Newest strategies as CASE WHEN rules

This file covers the three best strategies of the improvements run
(`improvements.py`, `results/run_20261019_top3_2000_improvements`), out of
its 300 combinations ranked by full-period return. Each one is written as
the decision taken at every bar. Earlier strategies are in the reports of
their runs.

## Common setup

- **Fit:** at each 5-minute bar *t*, the last 1,560 bars (about 20 trading
  days) of the price field are fitted with
  `modefinder.ModeFinder(..., smooth_option="kernel", weights=volume)`. Each
  bar's volume, divided by the window's total volume, weights its gaussian
  kernel. The bandwidth is the unweighted `bw.nrd0`.
- **Price fields:** high, open and close are each run on their own, and the
  results are the mean of the three.
- **Bernstein mode filter:** it blocks some buys and signal sells. The mode
  is fitted on the last 1,560 closes with the support fixed at
  [lowest − 1/1560, highest + 1/1560]. The same filter is used for every
  price field.
- **Decision:** taken at the close of bar *t* and filled at the open of bar
  *t* + 1, all in or all out. Every buy or sell costs 2 bps. Each ticker
  starts with $10,000.
- **Reading the rules:** the first `WHEN` that matches wins. `KEEP` keeps the
  current position.

| name | meaning |
|---|---|
| `close_t` | close of bar *t* |
| `holding` | the strategy holds the ticker going into bar *t* |
| `wq25_t`, `wq99_t` | 25% and 99% quantiles of the volume-weighted kernel fit |
| `max_prev_t` | highest value of the price field in the 1,560 bars before *t* (not counting *t*) |
| `mode_t` | Bernstein mode of the last 1,560 closes |
| `base_tim_all` | share of all bars since trading started (2026-05-13) in which the plain strategy held the ticker. The plain strategy buys below `wq25_t` and sells above the same threshold as the strategy (`max_prev_t` for 1, `wq99_t` for 2 and 3), with no other rule |

`base_tim_all` starts at 0, so the first buy signal is always taken.

## 1. Volume weights · sell above the 20-day max · trajectory guard

```sql
CASE
  WHEN holding     AND close_t > max_prev_t
                   AND NOT mode_rising                     THEN 'SELL'
  WHEN NOT holding AND close_t < wq25_t
                   AND base_tim_all < 0.5
                   AND NOT mode_falling                    THEN 'BUY'
  ELSE 'KEEP'
END
-- mode_falling = mode_t < mode_{t-78} AND mode_{t-78} < mode_{t-156}
--                (fell over each of the last two days, 78 bars a day)
-- mode_rising  = mode_t > mode_{t-78} AND mode_{t-78} > mode_{t-156}
```

## 2. Volume weights · sell above q99 · trajectory guard

```sql
CASE
  WHEN holding     AND close_t > wq99_t
                   AND NOT mode_rising                     THEN 'SELL'
  WHEN NOT holding AND close_t < wq25_t
                   AND base_tim_all < 0.5
                   AND NOT mode_falling                    THEN 'BUY'
  ELSE 'KEEP'
END
-- mode_falling, mode_rising as in 1
```

## 3. Volume weights · sell above q99 · band, as written in the request

```sql
CASE
  WHEN holding     AND close_t > wq99_t
                   AND NOT band_falling                    THEN 'SELL'
  WHEN NOT holding AND close_t < wq25_t
                   AND base_tim_all < 0.5
                   AND NOT band_rising                     THEN 'BUY'
  ELSE 'KEEP'
END
-- loo_modes    = at the close of the last full day before t, the Bernstein
--                modes of the 1,560-close window refitted with each of its
--                days (20 or 21) left out in turn
-- band_falling = mode_t < QUANTILE(loo_modes, 0.10)
-- band_rising  = mode_t > QUANTILE(loo_modes, 0.90)
```

## How they differ

| | 1 | 2 | 3 |
|---|---|---|---|
| Buy | below `wq25_t` | below `wq25_t` | below `wq25_t` |
| Sell | above the 20-day max | above `wq99_t` | above `wq99_t` |
| Gate | `base_tim_all < 0.5` | `base_tim_all < 0.5` | `base_tim_all < 0.5` |
| Mode filter | two-day trajectory | two-day trajectory | leave-one-day-out band |
| Blocks while the mode falls | buys | buys | sells |
| Blocks while the mode rises | sells | sells | buys |

## Results

These are for the 2,000 tickers of round 3. The period is 109 trading days,
2026-05-13 to 2026-10-16. Returns are totals over that period, not
annualized. The portfolio splits $10,000 equally across the tickers and the
three price fields. The second half starts at the close of 2026-07-31.
Round 3 and buy and hold are shown for reference.

| strategy | return | Sharpe | max drawdown | 2nd half return | tickers profitable | in market | trades per ticker |
|---|---|---|---|---|---|---|---|
| 1. volume weights · sell above 20-day max · trajectory guard | **2.55%** | **0.92** | −2.4% | +0.60% | 61% | 45% | 3.5 |
| 2. volume weights · sell above q99 · trajectory guard | 2.51% | 0.91 | −2.4% | **+0.66%** | 61% | 44% | 3.6 |
| 3. volume weights · sell above q99 · band, as written | 2.48% | **0.92** | −2.3% | +0.57% | 60% | 43% | 3.7 |
| Round 3: q25/q90, kernel, no rule | 0.34% | 0.14 | −4.7% | −1.85% | 53% | 58% | 7.1 |
| Buy and hold | −0.64% | −0.06 | −6.6% | −2.14% | 46% | 100% | 1.0 |

- **The higher sell threshold accounts for most of the gain.** With the
  unweighted kernel and no filter, the same gate selling above q90 returns
  1.26%, and selling above q99 returns 2.13%.
- **Volume weights add about 0.3 points:** 2.44% at q99 with no filter.
- **The mode filter adds about 0.1 points or less.** Strategies 1–2 and
  strategy 3 apply it in opposite directions and score about the same.
- **Not an out-of-sample test:** these are the best of 300 combinations on
  one period and the same tickers. Part of the gain also comes from holding
  more cash while the market fell in the second half.

The full tables are in the run's `report.md`.
