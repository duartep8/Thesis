"""Step 7b: independent re-tune of two elastic-net setups (December 1989 fit).

Usage:
    python src/07b_retune_check.py [--panel data/panel.parquet] [--tuning results/tables/05_enet_tuning.csv]

Written separately from 05_train.py and shares no code with it: a plain loop over the same
grid with scikit-learn's ElasticNet, one fresh fit per grid point. Grid: l1_ratio 0.1, 0.5,
0.9; for each, 30 penalties log-spaced from the smallest penalty that sets every
coefficient to zero down to 1e-4 of it. Protocol (CLAUDE.md, 2026-11-16): the model is fit on
the target minus its monthly cross-sectional mean; validation predictions are demeaned within
each month and the training part's average raw return is added; validation R^2 =
1 - sum (y - f)^2 / sum y^2 against the raw target. Near-constant rule: only grid points with at
least 50 distinct validation predictions in every month qualify.
Windows by return month (= eom + 1 month); validation = the last 30% of the months.
Convergence settings (max_iter, tol) must equal 05's; the script checks 05's source text for
them without importing it. One line per fit (iterations, time); BLAS limited to THREADS threads.
Compares the choice and every grid point with 05's tuning table.
"""

import argparse
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent.parent
SETUPS = {"10y": "1980-01", "expanding": "1975-01"}  # first return month of the December 1989 window
LAST, VAL = "1989-12", 0.30
MAX_ITER, TOL = 100_000, 1e-6  # must equal 05_train.ElasticNet (checked below)
THREADS = 4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--tuning", default=str(ROOT / "results" / "tables" / "05_enet_tuning.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "tables" / "07b_enet_retune_check.csv"))
    args = parser.parse_args()
    src = (ROOT / "src" / "05_train.py").read_text()
    m05 = re.search(r"MAX_ITER, TOL = ([\d_]+), ([\de.-]+)", src)
    assert m05 and (int(m05[1]), float(m05[2])) == (MAX_ITER, TOL), "convergence settings differ from 05"
    with threadpool_limits(THREADS):
        run(args)


def run(args):

    chars = [l.strip() for l in (ROOT / "src" / "characteristics_153.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    df = pd.read_parquet(args.panel, columns=["eom", "ret_exc_lead1m"] + chars).dropna(subset=["ret_exc_lead1m"])
    df["ret_month"] = (pd.to_datetime(df["eom"]) + pd.offsets.MonthEnd(1)).dt.to_period("M")
    df["y_dm"] = df["ret_exc_lead1m"] - df.groupby("ret_month")["ret_exc_lead1m"].transform("mean")
    ours = pd.read_csv(args.tuning)

    rows = []
    for window, first in SETUPS.items():
        months = pd.period_range(first, LAST, freq="M")
        n_val = int(np.floor(VAL * len(months) + 0.5))
        tr, va = df[df["ret_month"].isin(months[:-n_val])], df[df["ret_month"].isin(months[-n_val:])]
        X_tr, y_tr = tr[chars].to_numpy(float), tr["y_dm"].to_numpy(float)
        X_va, y_va = va[chars].to_numpy(float), va["ret_exc_lead1m"].to_numpy(float)
        c = tr["ret_exc_lead1m"].mean()  # training part's average raw return
        va_month = va["ret_month"].to_numpy()
        print(f"{window} 30%: train {months[0]} to {months[-n_val - 1]} ({len(y_tr):,} rows), "
              f"validation {months[-n_val]} to {months[-1]} ({len(y_va):,} rows)", flush=True)
        for l1 in (0.1, 0.5, 0.9):
            alpha_max = np.abs((X_tr - X_tr.mean(0)).T @ (y_tr - y_tr.mean())).max() / (len(y_tr) * l1)
            for k, alpha in enumerate(alpha_max * np.logspace(0, -4, 30)):
                t0 = time.time()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always", ConvergenceWarning)
                    m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=MAX_ITER, tol=TOL).fit(X_tr, y_tr)
                warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
                p = pd.DataFrame({"f": m.predict(X_va), "month": va_month})
                f = (p["f"] - p.groupby("month")["f"].transform("mean")).to_numpy() + c
                r2 = 1 - np.sum((y_va - f) ** 2) / np.sum(y_va ** 2)
                rows.append({"window": window, "val_share": VAL, "l1_ratio": l1, "alpha": alpha, "val_r2": r2,
                             "min_distinct": int(p.groupby("month")["f"].nunique().min()),
                             "n_nonzero": int((m.coef_ != 0).sum()), "n_iter": int(m.n_iter_),
                             "not_converged": warned})
                print(f"  l1 {l1:g} point {k + 1:2d}/30  penalty {alpha:.3e}  val R2 {r2:.4%}  "
                      f"nonzero {rows[-1]['n_nonzero']:3d}  iterations {m.n_iter_:6d}"
                      f"{'  NOT CONVERGED' if warned else ''}  {time.time() - t0:5.1f}s", flush=True)
    check = pd.DataFrame(rows)
    ok = check["min_distinct"] >= 50
    best_ok = check[ok].groupby("window")["val_r2"].transform("max")
    check["chosen"] = ok & (check["val_r2"] == best_ok.reindex(check.index))

    out = []
    print()
    for window, c in check.groupby("window", sort=False):
        o = ours[(ours["window"] == window) & np.isclose(ours["val_share"], VAL)].reset_index(drop=True)
        c = c.reset_index(drop=True)
        # 05's table stores 6 significant digits, so its penalties match ours to about 5e-6 (relative)
        assert len(o) == len(c) == 90 and np.allclose(o["l1_ratio"], c["l1_ratio"])
        assert np.allclose(c["alpha"], o["alpha"], rtol=1e-5, atol=0), "the two grids differ"
        a, b = c[c["chosen"]].iloc[0], o[o["chosen"]].iloc[0]
        same = bool(c["chosen"].idxmax() == o["chosen"].idxmax())  # same grid point (l1_ratio and penalty)
        rank = (c["val_r2"].rank(ascending=False) == o["val_r2"].rank(ascending=False)).mean()
        top = c["val_r2"].sort_values(ascending=False).iloc[:2].to_numpy()
        r = {"window": window, "val_share": VAL, "chosen_alpha": a["alpha"], "chosen_l1_ratio": a["l1_ratio"],
             "chosen_val_r2": a["val_r2"], "chosen_n_nonzero": int(a["n_nonzero"]), "ours_alpha": b["alpha"],
             "ours_l1_ratio": b["l1_ratio"], "ours_val_r2": b["val_r2"], "same_choice": same,
             "max_grid_alpha_rel_diff": float(np.max(np.abs(c["alpha"] / o["alpha"] - 1))),
             "max_grid_val_r2_abs_diff": float(np.max(np.abs(c["val_r2"] - o["val_r2"]))),
             "same_nonzero_share": float((c["n_nonzero"] == o["n_nonzero"]).mean()),
             "same_rank_share": float(rank), "gap_best_to_second": float(top[0] - top[1]),
             "max_n_iter": int(c["n_iter"].max()), "n_not_converged": int(c["not_converged"].sum())}
        out.append(r)
        print(f"{window} 30%: re-tune picks penalty {a['alpha']:.6e}, l1_ratio {a['l1_ratio']:g} "
              f"(val R2 {a['val_r2']:.6%}); 05 picked {b['alpha']:.6e}, l1_ratio {b['l1_ratio']:g} "
              f"(val R2 {b['val_r2']:.6%}) -> {'SAME' if same else 'DIFFERENT'}")
        print(f"  over the 90 grid points: penalties differ by at most {r['max_grid_alpha_rel_diff']:.1e} (relative; "
              f"05's table keeps 6 significant digits), "
              f"val R2 by at most {r['max_grid_val_r2_abs_diff']:.1e}; same number of nonzero coefficients at "
              f"{r['same_nonzero_share']:.0%}, same rank at {r['same_rank_share']:.0%}; best minus second-best val R2 "
              f"{r['gap_best_to_second']:.1e}; most iterations {r['max_n_iter']:,} of {MAX_ITER:,}, "
              f"not converged {r['n_not_converged']}")
    pd.DataFrame(out).to_csv(args.out, index=False, float_format="%.8g")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
