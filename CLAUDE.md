# Thesis project

Master's thesis in Finance on machine-learning return models for US stocks:
how training choices and retraining affect forecasts and portfolio performance.

## How the models work
- Inputs: each stock's characteristics in month t (JKP's 153 characteristics).
- Target: the stock's excess return in month t+1 (`ret_exc_lead1m`).
- Use: each month, sort stocks by predicted return; long the top 10%, short the bottom 10%.

## Current plan (agreed with supervisor; details may change)
- Models: elastic net, boosted trees (LightGBM), small neural network.
- Part 1: for each model, compare training+validation window lengths
  (e.g. 5 vs 10 years) and training once vs retraining every year.
- Part 2: value of retraining: compare A (last year's model, unchanged)
  with C (retrained with the new year), over the same months.
- Measures: out-of-sample R² and, mainly, the Sharpe ratio of the long–short portfolio.
- Benchmark: Fama–French 5 factors (and possibly momentum): alphas.

## Still open (do not assume)
- Rolling vs expanding windows; training/validation split inside a window.
- Hyperparameters: tuned once or at every retrain.
- Neural network size.
- Trading costs and turnover.

## Data
- `data/` holds the JKP Global Factor Data from WRDS (licensed):
  never commit it, never copy it outside `data/`, never print large extracts.
- Work with parquet; read big files in chunks or by columns.
- `data/test_sample.parquet` (once created): small version for quick tests.

## Data decisions
- Source: JKP Global Factor Data (WRDS), US only, common stocks, main exchange,
  one security per firm, CRSP stocks only (`source_crsp = 1`), so `id` = PERMNO.
- Drop rows with missing `me` (rare). Keep rows with a missing `ret_exc_lead1m`
  (dropped only at training) to avoid look-ahead.
- Characteristics: JKP's 153, ranked each month across the universe to [−1, 1];
  missing values = 0 (the median), following Gu, Kelly & Xiu (2020). No winsorizing.
- Universe: the 1,000 largest stocks by `me` each month.
- A held stock with a missing return counts as 0 and is logged; missing spread or
  volume is filled with that month's size-decile median.

## Working rules
- Go step by step: explain what you will do, wait for my OK, then do it.
- One script per step in `src/`, numbered (01_, 02_, ...).
- No look-ahead: a model may only use data available before the month it predicts.
- Test every script on the test sample first, then run on the full data.
- Fixed random seeds; save every set of predictions so nothing is refit.
- Commit after each working step with a clear message.
