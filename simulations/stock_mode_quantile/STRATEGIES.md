# Newest strategies as CASE WHEN rules

This file covers the strategies from the threshold study
(`thresholds.py`, `results/run_20261019_top3_2000_thresholds`) and the
stop-rules run (`stop_rules.py`, `results/run_20261019_top3_2000_stop_rules`).
Each one is written as the decision taken at every bar.

## Common setup

- **Fit:** at each 5-minute bar *t*, the last 1,560 bars (about 20 trading
  days) of the price field are fitted with
  `modefinder.ModeFinder(..., smooth_option="kernel")`. Each price field
  (high, open, close) is run on its own, and the results are the mean of the
  three.
- **Decision:** taken at the close of bar *t* and filled at the open of bar
  *t* + 1, all in or all out. Every buy or sell costs 2 bps. Each ticker
  starts with $10,000.
- **Reading the rules:** the first `WHEN` that matches wins. `KEEP` keeps the
  current position.

| name | meaning |
|---|---|
| `close_t` | close of bar *t* |
| `q05_t`, `q25_t`, `q90_t`, `q95_t` | fitted quantiles of the last 1,560 bars |
| `holding` | the strategy holds the ticker going into bar *t* |
| `traded_bars` | bars since trading started (2026-05-13): *t* − start + 1 |

## 1. Time in market over the last 20 days > 85%, entry only

From the threshold study. The 85.3% threshold was chosen on the first half
of the period (2026-05-13 to 2026-07-31). It is the 70th percentile of that
half's values.

```sql
CASE
  WHEN holding     AND close_t > q90_t                     THEN 'SELL'
  WHEN NOT holding AND close_t < q25_t
                   AND base_tim_20d > 0.853                THEN 'BUY'
  ELSE 'KEEP'
END
-- base_tim_20d = share of the last LEAST(1560, traded_bars) bars in which the
--                plain q25/q90 strategy (no rule) held the ticker
```

- **What it picks:** tickers that the plain strategy has held for a long
  time without reaching the 90% quantile, so the price has stayed low. The
  strategy enters these only on a fresh dip below the 25% quantile.
- **Entry vs exit:** selling when the rule stops holding gives the same
  result. While this strategy holds, the plain one holds too, so
  `base_tim_20d` only rises and the gate never closes during a holding.

## 2. Time in market since start < 50%, entry only

One of the requested thresholds in the threshold study.

```sql
CASE
  WHEN holding     AND close_t > q90_t                     THEN 'SELL'
  WHEN NOT holding AND close_t < q25_t
                   AND base_tim_all < 0.5                  THEN 'BUY'
  ELSE 'KEEP'
END
-- base_tim_all = share of all bars since trading started in which the
--                plain q25/q90 strategy (no rule) held the ticker
```

- **First buy:** `base_tim_all` starts at 0, so the first buy signal is
  always taken.
- **Exit version:** also selling once `base_tim_all` reaches 50% did much
  worse: +0.04% for the full period, in the market 19% of the time.

## 3. q25/q90 + cap + stop

From the stop-rules run. These are the rules of the new strategy below,
applied to round 3's quantiles.

```sql
CASE
  WHEN holding AND close_t > q90_t                         THEN 'SELL'   -- signal
  WHEN holding AND traded_bars >= 1560
               AND own_tim_20d > 0.5                       THEN 'SELL'   -- cap
  WHEN holding AND close_t < entry_price * 0.94            THEN 'SELL'   -- stop:
                                                         -- stop_level = entry_price * 0.94
  WHEN NOT holding AND close_t < q25_t
               AND NOT (traded_bars >= 1560 AND own_tim_20d > 0.5)
               AND (stop_level IS NULL OR close_t >= stop_level)
                                                           THEN 'BUY'    -- entry_price = next open
  ELSE 'KEEP'
END
-- own_tim_20d = share of the last 1560 bars in which THIS strategy held the
--               ticker, counting the current bar
-- stop_level  = set by a stop sale; while flat, cleared once close_t >= stop_level
```

## 4. New strategy: q05/q95 + cap + stop

The strategy requested on 2026-10-20 (see `PROMPT.md`). It is case 3 with
`q25_t` replaced by `q05_t` and `q90_t` by `q95_t`.

```sql
CASE
  WHEN holding AND close_t > q95_t                         THEN 'SELL'   -- signal
  WHEN holding AND traded_bars >= 1560
               AND own_tim_20d > 0.5                       THEN 'SELL'   -- cap
  WHEN holding AND close_t < entry_price * 0.94            THEN 'SELL'   -- stop
  WHEN NOT holding AND close_t < q05_t
               AND NOT (traded_bars >= 1560 AND own_tim_20d > 0.5)
               AND (stop_level IS NULL OR close_t >= stop_level)
                                                           THEN 'BUY'
  ELSE 'KEEP'
END
```

## How they differ

| | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Buy / sell quantiles | q25 / q90 | q25 / q90 | q25 / q90 | q05 / q95 |
| Whose time in market | plain round-3 strategy | plain round-3 strategy | its own | its own |
| Window | last 20 days | since start | last 20 days | last 20 days |
| Condition | > 85.3% | < 50% | must stay ≤ 50% | must stay ≤ 50% |
| Effect | blocks buys | blocks buys | forces sales and blocks buys | forces sales and blocks buys |
| Stop loss | no | no | 6% below buy price | 6% below buy price |
| Threshold | chosen on the first half | requested | requested | requested |

## Results

These are for the 2,000 tickers of round 3. The period is 109 trading days,
2026-05-13 to 2026-10-16. Returns are totals over that period, not
annualized. The portfolio splits $10,000 equally across the tickers and the
three price fields. The second half starts at the close of 2026-07-31.

| strategy | return | Sharpe | max drawdown | 2nd half return | in market | trades per ticker |
|---|---|---|---|---|---|---|
| 1. Time in market over the last 20 days > 85%, entry only | 0.92% | 0.49 | −2.0% | **+0.62%** | 26% | 3.1 |
| 2. Time in market since start < 50%, entry only | **1.26%** | **0.50** | −2.5% | −0.30% | 40% | 4.9 |
| 3. q25/q90 + cap + stop | 0.78% | 0.40 | −3.0% | −1.62% | 35% | 14.9 |
| 4. q05/q95 + cap + stop (new) | 0.35% | 0.22 | −2.5% | −1.47% | 29% | 10.0 |
| Round 3: q25/q90, no rule | 0.34% | 0.14 | −4.7% | −1.85% | 58% | 7.1 |
| Buy and hold | −0.64% | −0.06 | −6.6% | −2.14% | 100% | 1.0 |

- **All four beat round 3** on return, Sharpe and drawdown. The new
  strategy's lead on return is only 0.01 points. Part of the gain comes from
  holding more cash while the market fell in the second half.
- **Strategies 1 and 2 point in opposite directions:** one needs high time
  in market and the other low.
- **Strategy 1 is weak evidence:** its threshold was fitted on the first
  half. At the ticker level, the tickers passing it did worse over the next
  20 days in the second half (−2.10% vs −1.10%).
- **Only 3 and 4 were set before the run.** Even so, the 6% stop and the
  50% cap were chosen after seeing round 3 on the same tickers and period.
  None of the four is an out-of-sample test.

The full tables are in each run's `report.md`.
