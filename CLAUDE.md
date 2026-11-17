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
- Target and predictions, all models (`05_train.py`; changed 2026-11-16, see below):
  - Every model is fit, tuned and early-stopped on `ret_exc_lead1m` minus that month's
    cross-sectional mean over universe rows with a target.
  - Predictions: the model's output minus its mean within each month (over all universe
    rows, known at t), plus c = the average raw return of the build's training rows (known at
    the December build date). The cross-sectional part is pure ranking; portfolios are
    unaffected by the monthly shift; R² keeps its definition (1 − Σ(y − f)²/Σy², raw y).
  - Validation R² is computed the same way (predictions demeaned within each validation
    month, plus c), and it ranks grid points and drives LightGBM's early stopping.
  - Models cannot time the market by construction (unlike Gu, Kelly & Xiu's).
- Near-constant rule, all models, pre-committed 2026-11-16 before any run under it: a grid
  point qualifies only if its validation predictions take at least 50 distinct values in
  every validation month. The best qualifying point by validation R² is chosen. If none
  qualifies (in the final pass), the point with the most distinct predictions is taken
  (then by validation R²) and the setup is flagged `near_constant_rule_failed`. For LightGBM
  this comes before the tree-count edge rule, and again after its rerun. Not a rule but
  reported: distinct predictions per test month for every fit and setup.
- Protocol change, 2026-11-16, after the first LightGBM run (archived with a README in
  `results/tables/archive/lgbm_v1_raw_target/`): trained on the raw target, LightGBM used
  tied ranks of discrete characteristics (`ni_inc8q`, `f_score`), whose tied value differs by
  month, as a month label to fit monthly market returns (`ni_inc8q` 65–80% of gain;
  several 5y/10y fits with 1–13 trees and 3–5 distinct predictions per month). The change
  (demeaned target and predictions, near-constant rule, random tie-break) answers this
  mechanism in the training data, not test performance; it applies to all models, and the
  elastic net is refit under it.
- LightGBM (`05_train.py --model lgbm`): squared loss, learning rate 0.05, bagging 0.8
  every iteration, seed 42, deterministic, 8 threads. Grid num_leaves {7, 15, 31} ×
  min_child_samples {200, 1000}; at the first fit (December 1989) each point is fit on the
  training part with early stopping on the validation part (at most 1,000 trees, 50 rounds
  without improvement); the point with the highest validation R² wins, and its best
  iteration is n_trees (validation R² as defined above; near-constant rule first). Settings,
  n_trees included, are then fixed; refits use the training part only, with exactly n_trees trees.
  - Edge rule, pre-committed before any LightGBM result was seen (2026-11-16), applied
    once per setup at the December 1989 tuning: if the chosen point's n_trees is below 20,
    rerun the whole tuning grid at learning rate 0.01; if it reaches the 1,000 cap, rerun
    it at 0.1. The rerun's best point by validation R² is chosen. The trigger and the first
    pass are logged in `05_lgbm_tuning.csv` (`rule`, `first_pass`).
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
- Results include CRSP delisting returns, with the double count in our JKP vintage removed
  by CRSP's `MthDelFlg` (see "Delisting returns" under Data decisions). No provisional label
  is set; `04_portfolios.RESULT_STATUS` adds one to every 05/06 table (a `status` column)
  if an input becomes provisional again.
- `06_evaluate.py` reports every setup under three treatments of the remaining held missing returns:
  0 (baseline), −30% for all, and −30% for distressed stocks only (Sharpe ratios, alphas,
  yearly-vs-once tests; `06_<model>_missing_sensitivity.csv`).

## Still open (do not assume)
- Neural network size.
- Optional: CRSP `MthRet` for the delisting months, to check the corrected returns value by
  value (the flag already decides which months were double counted).

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
- Delisting returns (`01_clean.py`, step 6). Two CRSP CIZ files from the supervisor:
  `data/crsp_delist/Data_Delisting.csv` (delisting events: date, delret, reasons) and
  `data/crsp_delist/16_11_data.csv` (the CIZ monthly file's `MthDelFlg` by PERMNO and month;
  duplicate rows dropped):
  - JKP already compounds CRSP's delisting return into the return of the delisting month
    (the target of row t−1), with −30% for performance-related delistings without one.
    Our JKP vintage counts it twice when CRSP's `MthRet` already includes it (JKP fixed
    this in jkp-data on 02-09-2026; our download predates the fix).
  - Primary rule, CRSP's `MthDelFlg` of the delisting month (definitions from CRSP's US
    Stock guide): A or P (MthRet includes the delisting return, so JKP counted it twice) →
    corrected to (1 + ret)/(1 + delret) − 1 − rf (rf from the delisting-month row): `fixed`
    (`unchanged` if delret = 0 or −1). M or G (MthRet excludes it, so JKP's single
    compounding, or its −30% for a missing delret, stands) → `unchanged`; JKP's ×0.70 on
    Nasdaq → ×0.45 (Shumway & Warther −55%): `nasdaq55`.
  - Pattern rule, for V and for months without a flag, and as a logged check on every row
    (each disagreement with the flag is listed in `data/reports/01_clean.log`): compare
    (1 + ret) / price relative with (1 + delret)^k; k = 2 → `fixed`, k = 1 → `unchanged`;
    ×0.70 on Nasdaq → `nasdaq55`. Exact match first (tolerance 1e-4), else allowing a
    dividend yield of 0–5%; if neither or both readings fit: `ambiguous`, left unchanged.
  - Why V uses the pattern: V is not defined in CRSP's guide. JKP treats it as "included",
    and most V months are indeed counted twice, but 11 (mostly 1975–1983 liquidations)
    contain (1 + delret) exactly once in JKP's return; there, dividing by (1 + delret) would
    remove a real delisting return. The pattern reads JKP's actual number, so it decides.
  - Columns: `ret_exc_lead1m_jkp` (JKP's original), `delist_flag` (MthDelFlg),
    `delist_fix` / `delist_fix_how` (`flag` or the pattern's match) on row t−1, `delisted`
    on row t. `ret_exc_lead1m_raw` is the target before the suspect
    rule (after the delisting fix).
  - Stocks that delist during month t stay in the universe and the ranks at t but are left
    out of the month-t portfolios (known at t). A position held into its delisting month
    is closed at t: counted in turnover, no spread charged (the holder is paid out).
  - Not corrected: characteristics of the delisting-month row (e.g. `ret_1_0`) still hold
    the double count; that row is never traded or trained on.
- Characteristics: JKP's 153, ranked each month across the universe to [−1, 1];
  missing values = 0 (the median), following Gu, Kelly & Xiu (2020). No winsorizing.
- Universe: the 1,000 largest stocks by `me` each month.
- Portfolio sort (`04_portfolios.py`): ties in the signal are broken by a seeded hash of
  (PERMNO, month), not by PERMNO order (related to firm age). Ties are rare for the elastic
  net (0.05% of predictions; a decile boundary falls in a tie in at most 0.23% of months).
- Suspect returns: `ret_exc_lead1m` ≤ −0.90 while the stock's `me` in t+1 is at least
  50% of its `me` in t. The original is kept in `ret_exc_lead1m_raw`, the row is
  flagged in `ret_exc_lead1m_suspect`, and `ret_exc_lead1m` is set to missing: dropped
  from training; in portfolios it counts as 0 and is logged (3 rows in 1975–2025).
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
