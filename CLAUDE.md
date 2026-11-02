# Thesis project

Master's thesis in Finance on machine-learning return models for US stocks:
how training choices and retraining affect forecasts and portfolio performance.

## How the models work
- Inputs: each stock's characteristics in month t (JKP's 153 characteristics).
- Target: the stock's excess return in month t+1 (`ret_exc_lead1m`).
- Use: each month, sort stocks by predicted return; long the top 10%, short the bottom 10%.

## Current plan (agreed with supervisor; details may change)
- Models: elastic net, boosted trees (LightGBM), small neural network (1–2 seeds).
- Part 1: for each model, a grid of window × training frequency.
  - Windows: rolling 5, 10 and 15 years of training + validation, plus expanding
    (from January 1975, the start of the data).
  - Frequency: trained once (fit on data up to the year before the test period,
    then kept unchanged) or retrained every year.
  - The data start in January 1975. The longest window (15 years) sets the first
    test year: 1990. Test period: 1990–2025.
  - All setups predict the same test period, so R² and Sharpe ratios are comparable.
  - Report the full grid; never pick a best window using test data.
- Training/validation split inside each window: a baseline split, plus a few
  variations as a sensitivity check.
- Hyperparameters: Part 1, tuned separately for each window setup (at its first fit).
  Part 2, tuned once and then fixed.
- Part 2 (A vs C, value of retraining): for all three models, in the expanding,
  yearly-retrained setup, over the same months. C = this December's refit on all
  data so far; A = last December's model, unchanged. No extra fits needed.
- Measures: out-of-sample R² (Gu, Kelly & Xiu, no demeaning) and, mainly, the
  Sharpe ratio of the decile long–short portfolio; alphas against
  Fama–French 5 (+ momentum).
- Trading costs: likely in the thesis; keep the code ready for them.

## Still open (do not assume)
- Neural network size.

## Data
- `data/` holds the JKP Global Factor Data from WRDS (licensed):
  never commit it, never copy it outside `data/`, never print large extracts.
- Work with parquet; read big files in chunks or by columns.
- `data/raw_parquet/`: the raw WRDS download (global, 1960–2025) as verified parquet
  (`00_csv_to_parquet.py`).
- `data/test_sample.parquet`: the 200 largest stocks each month, for quick tests.

## Data decisions
- Source: JKP Global Factor Data (WRDS), US only (`01_clean.py` keeps `excntry = 'USA'`),
  common stocks, main exchange, one security per firm, CRSP stocks only
  (`source_crsp = 1`), so `id` = PERMNO.
- Sample: January 1975 to December 2025; earlier months are dropped
  (`03_panel.py --start-year`, default 1975). Chosen from characteristic coverage:
  in the universe, the mean missing share is 47% in 1960, 15% in 1975, about 11% from 1977.
- Drop rows with missing `me` (rare). Keep rows with a missing `ret_exc_lead1m`
  (dropped only at training) to avoid look-ahead.
- Characteristics: JKP's 153, ranked each month across the universe to [−1, 1];
  missing values = 0 (the median), following Gu, Kelly & Xiu (2020). No winsorizing.
- Universe: the 1,000 largest stocks by `me` each month.
- Suspect returns: `ret_exc_lead1m` ≤ −0.90 while the stock's `me` in t+1 is at least
  50% of its `me` in t. The original is kept in `ret_exc_lead1m_raw`, the row is
  flagged in `ret_exc_lead1m_suspect`, and `ret_exc_lead1m` is set to missing: dropped
  from training; in portfolios it counts as 0 and is logged (4 rows in 1975–2025).
- A held stock with a missing return counts as 0 and is logged; missing spread or
  volume is filled with that month's size-decile median.

## Working rules
- Go step by step: explain what you will do, wait for my OK, then do it.
- One script per step in `src/`, numbered (01_, 02_, ...).
- No look-ahead: a model may only use data available before the month it predicts.
- Test every script on the test sample first, then run on the full data.
- Fixed random seeds; save every set of predictions so nothing is refit.
- Commit after each working step with a clear message.
