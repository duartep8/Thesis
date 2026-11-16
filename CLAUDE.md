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
    (from January 1975, the start of the data). Windows are defined by the month of
    the return predicted; a model built in December Y−1 uses returns up to December Y−1.
  - Frequency: trained once (fit on data up to the year before the test period,
    then kept unchanged) or retrained every year.
  - Returns start in January 1975 (the panel starts with the December 1974
    cross-section). The longest window (15 years) sets the first test year: 1990.
    Test period: return months 1990–2025.
  - Trained once, the 15-year and expanding windows are the same December 1989 fit;
    report them side by side, marked as identical.
  - All setups predict the same test period, so R² and Sharpe ratios are comparable.
  - Report the full grid; never pick a best window using test data.
- Training/validation split inside each window: validation = the last 30% of the
  window's months (baseline), 25% and 35% as variations, rounded to whole months.
  The model is fit on the training part only.
- Hyperparameters: Part 1, tuned separately for each window setup (at its first fit).
  Part 2, tuned once and then fixed.
- Part 2 (A vs C, value of retraining): for all three models, in the expanding,
  yearly-retrained setup, over return months 1991–2025. C = this December's refit on
  all data so far; A = last December's model, unchanged. No extra fits needed: each
  yearly model also predicts the year after its test year (saved with age = 1).
- Measures: out-of-sample R² (Gu, Kelly & Xiu, no demeaning) and, mainly, the
  Sharpe ratio of the decile long–short portfolio; alphas against
  Fama–French 5 (+ momentum).
- Evaluation (`06_evaluate.py`, any model; reads saved outputs, never refits):
  - Window, frequency and validation share are all part of the comparison: report
    every combination equally; never pick a best setup on test years; don't change
    model settings based on these results.
  - Sharpe ratio tests: Ledoit & Wolf (2008), studentized circular block bootstrap
    (block 5, 5,000 draws, seed 42) as the main p-value, HAC (prewhitened QS) alongside,
    Holm within each group. Groups: yearly vs once (each window × share); the three
    shares against each other (each window × frequency); each setup against the
    reference expanding, yearly, 30% (fixed in advance as the main design).
  - Alphas: FF5 + momentum, Newey–West 6 lags; gross and net.
  - Trading costs: cost = Σ |trade| × bidaskhl_21d / 2, the first month pays for the
    initial purchase. No intraday spreads, so bidaskhl_21d stays; it overstates large
    stocks' spreads (Chen & Velikov: 25–50 bp too high after 2003), so every net result
    is labelled an upper bound on costs. Break-evens: the cost multiple at which the net
    Sharpe ratio is zero, and at which yearly and once have equal net Sharpe ratios.
    Price impact comes later (add it to `COST_COMPONENTS`); no short-selling fees yet.
  - Validation-share diagnostic: `05_train.py --cross-fit` refits each share's
    training part with each other share's settings (diagnostic only, no reported setup
    changes), to separate the training-data effect from the tuning effect.

## Status of results
- All performance results are labelled **pending the MthDelFlg check**: the models are run
  on the delisting-corrected panel (see "Delisting returns" under Data decisions), but the
  pattern rule has not yet been checked against CRSP's `MthDelFlg`. Every 05/06 table carries
  this label (`04_portfolios.RESULT_STATUS`); remove it once the check is done.
- `06_evaluate.py` reports every setup under three treatments of the remaining held missing returns:
  0 (baseline), −30% for all, and −30% for distressed stocks only (Sharpe ratios, alphas,
  yearly-vs-once tests; `06_<model>_missing_sensitivity.csv`).

## Still open (do not assume)
- Neural network size.
- CRSP `MthDelFlg` (monthly delisting flag): extract requested from the supervisor. It says
  exactly which monthly returns already include the delisting return; use it to check the
  pattern rule in `01_clean.py` (a check, not a blocker).

## Data
- `data/` holds the JKP Global Factor Data from WRDS (licensed):
  never commit it, never copy it outside `data/`, never print large extracts.
  Model outputs that carry PERMNOs go only to the ignored `results/` folders below.
- Work with parquet; read big files in chunks or by columns.
- `data/raw_parquet/`: the raw WRDS download (global, 1960–2025) as verified parquet
  (`00_csv_to_parquet.py`).
- `data/test_sample.parquet`: the 200 largest stocks each month, for quick tests.
- `data/ff/`: Fama–French 5 factors (2x3) and momentum, monthly, from Kenneth French's
  data library; versions (CRSP vintage, SHA-256) in `data/ff/README.txt`.

## Folder layout
- `data/`: the licensed data and everything derived from it row by row (clean,
  panel, test sample, raw parquet, logs in `data/reports/`). Never committed.
- `results/predictions/`, `results/models/`, `results/portfolios/`: model outputs.
  They contain PERMNOs, so they are never committed (in `.gitignore`).
- `results/tables/`: summary tables such as `05_enet_summary.csv`. Aggregates only;
  committed. Figures in `results/tables/figures/`; notes such as
  `results/tables/notes_comparison.md` (our results against the literature).

## Data decisions
- Source: JKP Global Factor Data (WRDS), US only (`01_clean.py` keeps `excntry = 'USA'`),
  common stocks, main exchange, one security per firm, CRSP stocks only
  (`source_crsp = 1`), so `id` = PERMNO.
- Sample: return months January 1975 to December 2025; the panel starts with the
  December 1974 cross-section (`03_panel.py --start`, default 1974-12). Chosen from
  characteristic coverage: in the universe, the mean missing share is 47% in 1960,
  15% in 1975, about 11% from 1977.
- Drop rows with missing `me` (rare). Keep rows with a missing `ret_exc_lead1m`
  (dropped only at training) to avoid look-ahead. In 1975–2025 almost all of them are the
  stock's delisting month: it stopped trading during month t, so there is no month t+1.
  93% are mergers.
- Delisting returns (`01_clean.py`, step 6; CRSP CIZ file `data/crsp_delist/Data_Delisting.csv`):
  - JKP already compounds CRSP's delisting return into the return of the delisting month
    (the target of row t−1), with −30% for performance-related delistings without one.
    Our JKP vintage counts it twice when CRSP's `MthRet` already includes it (JKP fixed
    this in jkp-data on 02-09-2026; our download predates the fix).
  - Pattern rule, per delisting month: compare (1 + ret) / price relative with
    (1 + delret)^k. k = 2 → corrected to (1 + ret)/(1 + delret) − 1 − rf (rf from the
    delisting-month row): `fixed`. k = 1, or delret = 0 or −1: `unchanged`. Missing delret
    with JKP's ×0.70 on Nasdaq → ×0.45 (Shumway & Warther −55%): `nasdaq55`. Exact match
    first (tolerance 1e-4), else allowing a dividend yield of 0–5% in that month; if
    neither or both readings fit: `ambiguous`, left unchanged.
  - Columns: `ret_exc_lead1m_jkp` (JKP's original), `delist_fix` / `delist_fix_how` on
    row t−1, `delisted` on row t. `ret_exc_lead1m_raw` is the target before the suspect
    rule (after the delisting fix).
  - Stocks that delist during month t stay in the universe and the ranks at t but are left
    out of the month-t portfolios (known at t). A position held into its delisting month
    is closed at t: counted in turnover, no spread charged (the holder is paid out).
  - Not corrected: characteristics of the delisting-month row (e.g. `ret_1_0`) still hold
    the double count; that row is never traded or trained on.
- Characteristics: JKP's 153, ranked each month across the universe to [−1, 1];
  missing values = 0 (the median), following Gu, Kelly & Xiu (2020). No winsorizing.
- Universe: the 1,000 largest stocks by `me` each month.
- Suspect returns: `ret_exc_lead1m` ≤ −0.90 while the stock's `me` in t+1 is at least
  50% of its `me` in t. The original is kept in `ret_exc_lead1m_raw`, the row is
  flagged in `ret_exc_lead1m_suspect`, and `ret_exc_lead1m` is set to missing: dropped
  from training; in portfolios it counts as 0 and is logged (4 rows in 1975–2025).
- A held stock with a still-missing return (suspect returns, trading halts without a CRSP
  delisting record, a few gaps) counts as 0 and is logged (baseline). Sensitivity
  (`04_portfolios.py --missing-return -0.3 --missing-scope all|distressed`): it counts as
  −30%, for all missing returns or only for distressed stocks (unranked `ret_12_1` below
  −50% or last-month `ret_exc` below −30%, both known at t).
- Missing spread or volume is filled with that month's size-decile median.

## Working rules
- Go step by step: explain what you will do, wait for my OK, then do it.
- One script per step in `src/`, numbered (01_, 02_, ...).
- No look-ahead: a model may only use data available before the month it predicts.
- Test every script on the test sample first, then run on the full data.
- Fixed random seeds; save every set of predictions so nothing is refit.
- Each predictions file has a manifest (panel hash, test years, characteristics, model and
  tuning grid, seed, package versions, training-code hash; git commit recorded). 05 reuses
  a file only if the manifest matches and every test month is covered; otherwise it stops,
  and `--refit` refits and overwrites.
- Commit after each working step with a clear message.
