# Prompt

The requests this simulation was built for, kept verbatim for future runs.

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
