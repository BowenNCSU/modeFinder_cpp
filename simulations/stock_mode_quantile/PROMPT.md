# Prompt

The requests this simulation was built for, kept verbatim for future runs.

## Improvements (2026-10-20)

Run with `improvements.py` as `results/run_20261019_top3_2000_improvements`.

> try some improvements of strategies on the same 3rd round 2k tickers and compared with the last three. 1. try a combination of Time in market over the last 20 days > 85%, entry only + Time in market since start < 50%, entry only which is BUY when either base_tim_20d> 0.853 or base_tim_all < 0.5. 2. enable weight option for kernel estimation, allow to input an array of double with same length of data input, then first normalization: calculate the summation of weight and use it to divide all weights as normalized weight, then apply normalized weight on the pdf and cdf of each unit distribution generated centered at each data point which is the easiest way to apply weights. no need to enable weight for Bernstein. Then read the volume column from each row of tickers as weight to rerun kernel for all strategies. 3. try different SELL threshold, e.g., P95, P99, MAX on all strategies 4. try Bernstein quantile for all strategies with Lower bound = lowest price in the window + 1/1560. 5. use Bernstein mode with Lower bound = lowest price in the window + 1/1560, and use leave-1-out strategy among past 20 days to calculate the quantile of Bernstein mode. Then use either upper/lower quantile of Bernstein mode or the time-series ordered trajectory of recent Bernstein mode of either going up or down to help avoiding SELL when constantly dropping or BUY when constantly rising. compare all of them.

How the script reads it:

- **"The last three"** are the threshold study's two entry gates and the
  stop-rules strategy (cap + stop), all on q25 buys. Every strategy is rerun
  on each fit and sell threshold, with the gates computed from that same
  fit's base strategy.
- **Weights** are the bar volumes of the window, so the kernel option of
  `emp_mode`/`ModeFinder` gained `weights=`. The bandwidth stays unweighted.
- **MAX** is the highest value of the previous 1560 bars, so it sells on a
  new 20-day high.
- **The Bernstein lower bound** is min − 1/1560, not +: a bound above the
  lowest price would leave that price outside the support. The upper bound
  is max + 1/1560 by the same rule.
- **Leave one out** refits the Bernstein mode on the close at each day's end,
  leaving out each of the window's days in turn. The 10% and 90% quantiles
  of those modes form a band, used from the next bar.
- **"Constantly dropping"** is the mode below the band's low (band filter),
  or falling over each of the last two days, mode[t] < mode[t − 78] <
  mode[t − 156] (trajectory filter). Rising is the
  reverse. The request can be read two ways, so both are run: the guard
  blocks buys while dropping and sells while rising, and the literal reading
  blocks sells while dropping and buys while rising.

## Stop rules (2026-10-20)

Run with `stop_rules.py` as `results/run_20261019_top3_2000_stop_rules`.

> use new strategy of q5/q95 on a 20-day window with the kernel method, if historical bars has over 20 days, sell all stock/etf as long as time in market ratio > 50% in last 20 days, also sell all stock as long as price has droped over 6%. try on 3rd round's 2k stocks/etfs and compared with previous strategies' results.

The script reads "over 20 days" as: the time-in-market rule starts once the
strategy has traded for 20 days, so its 20-day ratio covers a full window.
It measures the 6% drop from the buy price. After a stop, it does not buy
again until the close is back above the stop level.

## Threshold study (2026-10-19)

Run with `thresholds.py` as `results/run_20261019_top3_2000_thresholds`.

> for 3rd round simulation on 2k tickers, rerun simulation and calculate daily/hourly skew and time in market ratio based on observed data in each step of simulation. Check what threshold or combination of thresholds can distinguish profitable tickers after using our strategy. Must try threshold skewness > 0 or time in market ratio < 50%. Also try other possible thresholds.

## Top-3 follow-up (2026-10-19)

Run as `results/run_20261019_top3_2000`. The grid scripts used for the two
requests below are in `results/run_20261019_seed2027/scripts_used/`.

> use top 3 combinations by considering 2 bps per trade in the 2nd round, test it among 2k stocks etfs. rank the most profitable targets. remove all other combinations from simulation scripts.

## Revised request (2026-10-19)

Run as `results/run_20261019_seed2027`.

> using script in dir  simulations. historial us stock market 5-min interval data are in /Users/bowenliu/Downloads/data/5 min. randomly select some
>   stocks/etfs, try either the mode or different quantiles of modefinder, try different range/window of stock open/close/max/min
>   prices. keep removing oldest data and adding newest data and running mode or quantile of modefinder based on new
>   data set and use mode or quantile to predict the distribution of future stock price. assume you have 10k usd, Use
>   the strategy that buy stocks/etf when current price is below predicted mode or lower quantile and sell stocks/etf when
>   current price is over predicted mode or upper quantile. use the history data to do this simulation: use the strategy
>   from the start date of data till the end date. compare the performance of different combination of parameters
>   above over the history data and give me a simulation report. First save all script of running this simulation, as well as this prompt into simulation dir for future run, then run this simulation.

## Original request (2026-10-19)

Run as `results/run_20261019_seed2026`, with the rules mode, median,
q25/q75, q10/q90 and q05/q95.

> historial us stock market 5-min interval data are in /Users/bowenliu/Downloads/data/5 min. randomly select some
> stocks/etfs, try either the mode or quantile of modefinder, try different range/window of stock open/close/max/min
> prices. keep removing oldest data and adding newest data and running mode or quantile of modefinder based on new
> data set and use mode or quantile to predict the distribution of future stock price. assum you have 10k usd, Use
> the strategy that buy stocks/ets when current price is below predicted mode or quantile and sell stocks/ets when
> current price is over predicted mode or quantile. use the history data to do this simulation: use the strategy
> from the start date of data till the end date. compare the performance of different combination of parameters
> above over the history data and give me a simulation report. First save all script of running this simulation, as well as this prompt into simulation dir for future run, then run this simulation.
