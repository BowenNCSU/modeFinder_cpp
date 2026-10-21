# Rolling quantile trading simulation

This folder backtests a simple trading rule. It refits `modefinder` on a
rolling window of 5-minute prices, then buys below a lower fitted quantile
and sells above an upper one. `PROMPT.md` holds the requests.

The script now runs only the three combinations that ranked first, with
2 bps per trade, in the second round (`results/run_20261019_seed2027`):

| rule | field | window | method |
|---|---|---|---|
| q25/q90 | high | 1560 bars | kernel |
| q25/q90 | open | 1560 bars | kernel |
| q25/q90 | close | 1560 bars | kernel |

It tests them on a large random sample of stocks and ETFs and ranks the
tickers by profit.

## Files

- `simulate.py` picks the tickers, runs the rolling fits in parallel,
  backtests the three combinations and buy and hold, and writes the CSV
  files and the report.
- `report.py` rebuilds `report.md` from a run's CSV files.
- `thresholds.py` reruns a finished run's tickers and tests thresholds on
  features observed at each step (daily and hourly skew, time in market,
  momentum, volatility and others). It looks for thresholds that pick out
  the tickers the strategy goes on to profit from. The rolling fits are
  cached in `<out>/cache`, about 1 GB for 2,000 tickers and ignored by git,
  so a rerun only redoes the analysis.
- `stop_rules.py` reruns a finished run's tickers with a new strategy:
  q05/q95 on the same 1560-bar kernel fits, with two extra rules. After 20
  days of trading, it sells and stays out while it has held the ticker more
  than 50% of the last 1560 bars. It also sells when the close is more than
  6% below the buy price. It also backtests each rule on its own and
  round 3's q25/q90 for comparison. Fits are cached in `<out>/cache`, about
  2 GB for 2,000 tickers and ignored by git.
- `STRATEGIES.md` writes the newest strategies as CASE WHEN rules and
  compares their results.
- `improvements.py` reruns a finished run's tickers over 300 variants:
  three fits (kernel, kernel weighted by bar volume, and Bernstein with
  bounds at the window's min and max ± 1/1560), four sell thresholds (q90,
  q95, q99 and the previous 1560-bar max), five strategies (plain, the two
  time-in-market entry gates, their OR, and cap + stop) and five filters
  from the close's Bernstein mode (none, plus a leave-one-day-out band and
  a two-day trajectory, each read two ways). Fits are cached in
  `<out>/cache`, ignored by git.
- `results/run_<timestamp>/` holds one run's output:
  - `config.json`: the arguments.
  - `selection.csv`: the tickers.
  - `per_ticker.csv`: each ticker × combination, plus buy and hold.
  - `ranking.csv`: one row per ticker, ranked by the mean return of the
    three combinations, with buy and hold and a flag for suspect price
    jumps.
  - `portfolio.csv`: an equal-weight portfolio per combination.
  - `report.md`

## Running

From the repository root, with modefinder installed (`pip install .`):

``` sh
python simulations/stock_mode_quantile/simulate.py \
    --data "/Users/bowenliu/Downloads/data/5 min" --count 2000
```

- **Defaults:** 2,000 tickers picked at random (seed 2028) from the eligible
  files, 2 bps per buy or sell, and a refit at every bar. Eligible files
  have at least 5,000 bars, a median close of at least $5 and a median
  dollar volume of at least $100,000 per bar.
- **Options:** `--help` lists them all. The main ones:
  - `--tickers SPY AAPL ...`: use these tickers instead of a random pick.
  - `--step N`: refit every N bars, which runs about N times faster.
  - `--cost-bps`, `--seed` and `--jobs`.
- **Rebuilding the report:**
  `python simulations/stock_mode_quantile/report.py <run dir>`.

## Runs

- **`results/run_20261019_seed2026`:** the original request. It used 10
  stocks and 5 ETFs, with the five rules mode, median, q25/q75, q10/q90 and
  q05/q95.
- **`results/run_20261019_seed2027`:** the revised request. It used 20
  stocks and 10 ETFs, with nine rules, three windows, four price fields,
  both methods and costs of 0 and 2 bps (216 combinations per cost).
- **`results/run_20261019_top3_2000`:** the three winning combinations of
  the run above on 2,000 tickers:

  ``` sh
  python simulations/stock_mode_quantile/simulate.py \
      --data "/Users/bowenliu/Downloads/data/5 min" --count 2000 --seed 2028 \
      --out simulations/stock_mode_quantile/results/run_20261019_top3_2000
  ```

- **`results/run_20261019_top3_2000_thresholds`:** thresholds on observed
  features for the run above:

  ``` sh
  python simulations/stock_mode_quantile/thresholds.py \
      --run simulations/stock_mode_quantile/results/run_20261019_top3_2000
  ```

  It writes `panel.csv.gz` (features and forward returns per ticker and
  day), `rules.csv` (every rule tried), `rank_ic.csv`, `gated_ticker.csv`,
  `gated_portfolio.csv` and `report.md`.

- **`results/run_20261019_top3_2000_stop_rules`:** the q05/q95 strategy
  with the time-in-market cap and the stop loss, on the same tickers:

  ``` sh
  python simulations/stock_mode_quantile/stop_rules.py \
      --run simulations/stock_mode_quantile/results/run_20261019_top3_2000
  ```

  It writes `per_ticker.csv` (each ticker × strategy × field),
  `ranking.csv` (tickers ranked by the new strategy, next to round 3 and
  buy and hold), `portfolio.csv` and `report.md`.

- **`results/run_20261019_top3_2000_improvements`:** the 300 variants of
  `improvements.py` on the same tickers:

  ``` sh
  python simulations/stock_mode_quantile/improvements.py \
      --run simulations/stock_mode_quantile/results/run_20261019_top3_2000
  ```

  It writes `portfolio.csv` (one row per variant), `per_ticker.csv.gz`,
  `config.json`, `fits.log` and `report.md`.

The first two runs used the grid version of the scripts, which is kept in
`results/run_20261019_seed2027/scripts_used/`.

## Data

The loader expects Stooq's bulk text files, such as
`5 min/us/nasdaq stocks/1/aapl.us.txt` with the header
`<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`.

- **Other CSVs:** files with `Date,Time,Open,High,Low,Close,Volume` or
  `Datetime,Open,...` headers also work.
- **Asset kind:** a path containing "etf" marks an ETF, and one containing
  "stock" marks a stock.
- **Local copies:** put a copy of the data in `data/` here. Git ignores it.

## Method

- **Rolling fit:** at each bar t, the last 1560 values of the price field
  (about 20 trading days) are fitted with
  `modefinder.ModeFinder(..., smooth_option="kernel")`. The window drops its
  oldest bar and adds the newest at every step.
- **Trading rule:** long only, all in or all out. Each ticker starts with
  $10,000.
  - **Buy:** when close[t] is below the fitted 25% quantile.
  - **Sell:** when close[t] is above the fitted 90% quantile.
- **Fills:** orders fill at open[t + 1], so nothing uses future data.
- **Comparison:** trading starts once the window has filled, so the three
  combinations and buy and hold cover the same period.
- **Ranking:** tickers are ranked by the mean total return of the three
  combinations. A ticker is flagged when a bar-to-bar price move exceeds
  40%, which often means an unadjusted split or a bad tick.
