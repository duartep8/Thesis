# LightGBM, first run (raw target): archived, not a reported result

Run on 2026-11-16 with the protocol then in CLAUDE.md: LightGBM trained, early-stopped and
tuned on the raw target `ret_exc_lead1m` (not demeaned), on the delisting-corrected panel
(commit 2fbff8a plus the uncommitted LightGBM code). The tables and tuning figures here are
that run's 05/06/07 output; its predictions and model files were overwritten by the rerun.

## What went wrong

**Tied ranks act as a month label.** Characteristics are ranked each month to [-1, 1] with
average ranks for ties. Some characteristics take few distinct values (`f_score`, 0-9;
`ni_inc8q`, the number of consecutive quarters of earnings increases, 0-8), so large groups
of stocks share one rank, and that shared value differs from month to month. In the
1975-1985 training data, 66% of `f_score` rows and 62% of `ni_inc8q` rows sit in a tie
group whose value occurs in at most two months. With a non-demeaned target, a tree can use
these values to isolate single months and fit their market return. Example: one pair of
splits in the expanding 30% model of December 1989, `ni_inc8q` in (0.4177, 0.4202],
isolates exactly the months 1979-09, 1979-11 and 1980-06 (universe mean returns -9.4%,
+2.9%, +7.7%). A linear model in ranks (the elastic net) cannot do this.

**Symptoms.**
- `ni_inc8q` ranks first in all 9 unique December-1989 fits, with 65% of total gain in the
  expanding 30% setup and 80% on average (`05_lgbm_importance_1989.csv`). Short-term
  reversal, momentum, size and volatility are present but small.
- Month-fitting does not generalize, so early stopping chose very few trees. The
  tree-count edge rule fired in 6 of 9 tunings. The 5y and 10y setups ended at 1-13 trees
  (learning rate 0.01) and predicted a median of 3-5 distinct values per month (against 999
  for the expanding setup), so their decile sides were decided mostly by the tie-break
  (then PERMNO order). Their Sharpe ratios were near zero.
- Average over the 24 setups, LightGBM vs elastic net: R² 0.10% vs 0.29%, gross Sharpe 0.33
  vs 0.59, rank IC 0.013 vs 0.028.

## What changed (CLAUDE.md, 2026-11-16)

All models are now trained and tuned on the target minus its monthly cross-sectional mean.
Predictions are demeaned within each month, then the training window's average return is
added back as a constant, so validation and test R² keep their definition. Ties in the
portfolio sort are broken by a seeded random key instead of PERMNO. A pre-committed rule
excludes grid points with fewer than 50 distinct validation predictions in any month. The
change was made because of the mechanism in the training data above, not because of test
performance.
