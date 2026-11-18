# Comparison with the literature

Our results include CRSP delisting returns; the double count in our JKP vintage is removed in
`01_clean.py` using CRSP's `MthDelFlg`.

## Elastic net vs Gu, Kelly & Xiu (2020), ENet+H

Gross results of the decile long–short (H–L) portfolio. GKX: Tables 7–8, value-weighted.
Ours: `05_enet_summary.csv` and `06_enet_summary.csv`, return months 1990–2025. The range
covers all 24 setups (window × frequency × validation share), all trained on the month-demeaned
target (protocol of 2026-11-16); the reference setup
(expanding, yearly, 30%) is shown because it is the one closest to GKX's design (expanding
training, refit every year). It is not a selected best setup.

| | H–L mean / month | Sharpe (ann.) | FF5+Mom alpha / month | Turnover / month | OOS R² |
|---|---|---|---|---|---|
| GKX ENet+H (value-weighted) | 0.60% | 0.39 | −0.23% (n.s.) | 152% | 0.11% (all stocks) |
| Ours, reference setup | 0.93% | 0.66 | 0.37% (t 1.93) | 232% | 0.51% |
| Ours, range over 24 setups | 0.52–1.05% | 0.34–0.99 | 0.18–1.09% (t 1.0–4.5) | 195–282% | −0.61% to 0.51% |

Our elastic net earns a higher gross mean, Sharpe ratio and R² than GKX's ENet+H, has a
positive rather than a negative alpha, and trades about 1.5 times as much. Six design
differences make the two not directly comparable:
- **Universe:** we use the 1,000 largest stocks each month; GKX use all CRSP stocks (several
  thousand a month). Our R² is measured on large stocks only; GKX's 0.11% is for all
  stocks, and they report top-1,000 and bottom-1,000 figures separately in Table 1, which
  is the closer comparison.
- **Predictors:** we use JKP's 153 characteristics, ranked to [−1, 1], with no macro
  interactions or industry dummies. GKX use 94 characteristics interacted with 8 macro
  variables, plus 74 industry dummies (920 predictors). They also fit with a Huber loss
  and tune on a rolling 12-year validation window; we use squared loss and the last 30%
  (25%, 35%) of each window.
- **Market timing:** our models cannot time the market, by construction. They are fit on
  each month's returns minus that month's cross-sectional mean, and their predictions are
  demeaned within each month before the training window's average return is added back as a
  constant (CLAUDE.md, 2026-11-16). So every month gets the same predicted average return, and
  the models only rank stocks within the month. GKX's models predict the raw return, and
  their macro interactions let the predicted level move with the state of the economy, so
  part of their R² can come from timing the market's average return. Our R² measures only
  the cross-sectional part plus a constant. The long–short portfolio is not affected, since
  a common monthly shift cancels between the two sides.
- **Test period:** ours is 1990–2025, GKX's is 1987–2016. They overlap in 1990–2016; our
  test includes 2017–2025 and leaves out 1987–1989.
- **Portfolio construction:** we hold 100 stocks per side (10% of 1,000), with value weights
  capped at the NYSE 80th percentile. GKX's value-weighted deciles over all stocks hold far
  more names and put more weight on the largest firms. Concentrated sides tilted away from
  mega caps can raise both the mean and the turnover. Turnover definitions are close but
  not identical: ours counts buys and sells on both sides, with drift renormalized within
  each side, so a full replacement of both sides is 400%. Check GKX's exact normalization
  before putting the two figures in one table.
- **Factor exposure:** our setups trained once load heavily on profitability (RMW 0.6–1.3)
  and, except for the 5-year window, on value (HML 0.45–0.7 for 10y, 15y and expanding;
  0.0–0.1 for 5y). Yearly setups have smaller and mixed loadings (RMW −0.3 to 0.7, HML
  −0.13 to 0.5). So our alphas differ from GKX's for reasons of composition as well as skill.

Net of spread costs (an upper bound, `06_enet_summary.csv`), our Sharpe ratios fall to
between −0.18 and 0.25; GKX report gross results only.
