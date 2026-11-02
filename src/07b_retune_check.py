"""Step 7b: independent re-tune of two elastic-net setups (December 1989 fit).

Usage:
    python src/07b_retune_check.py [--panel data/panel.parquet] [--tuning results/tables/05_enet_tuning.csv]

Written separately from 05_train.py and shares no code with it: a plain loop over the same
grid with scikit-learn's ElasticNet, one fresh fit per grid point. Grid: l1_ratio 0.1, 0.5,
0.9; for each, 30 penalties log-spaced from the smallest penalty that sets every
coefficient to zero down to 1e-4 of it. Validation R^2 = 1 - sum (y - f)^2 / sum y^2.
Windows by return month (= eom + 1 month); validation = the last 30% of the months.
Compares the choice and every grid point with 05's tuning table.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

ROOT = Path(__file__).resolve().parent.parent
SETUPS = {"10y": "1980-01", "expanding": "1975-01"}  # first return month of the December 1989 window
LAST, VAL = "1989-12", 0.30


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--tuning", default=str(ROOT / "results" / "tables" / "05_enet_tuning.csv"))
    parser.add_argument("--out", default=str(ROOT / "results" / "tables" / "07b_enet_retune_check.csv"))
    args = parser.parse_args()

    chars = [l.strip() for l in (ROOT / "src" / "characteristics_153.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    df = pd.read_parquet(args.panel, columns=["eom", "ret_exc_lead1m"] + chars).dropna(subset=["ret_exc_lead1m"])
    df["ret_month"] = (pd.to_datetime(df["eom"]) + pd.offsets.MonthEnd(1)).dt.to_period("M")
    ours = pd.read_csv(args.tuning)

    rows = []
    for window, first in SETUPS.items():
        months = pd.period_range(first, LAST, freq="M")
        n_val = int(np.floor(VAL * len(months) + 0.5))
        tr, va = df[df["ret_month"].isin(months[:-n_val])], df[df["ret_month"].isin(months[-n_val:])]
        X_tr, y_tr = tr[chars].to_numpy(float), tr["ret_exc_lead1m"].to_numpy(float)
        X_va, y_va = va[chars].to_numpy(float), va["ret_exc_lead1m"].to_numpy(float)
        print(f"{window} 30%: train {months[0]} to {months[-n_val - 1]} ({len(y_tr):,} rows), "
              f"validation {months[-n_val]} to {months[-1]} ({len(y_va):,} rows)", flush=True)
        for l1 in (0.1, 0.5, 0.9):
            alpha_max = np.abs((X_tr - X_tr.mean(0)).T @ (y_tr - y_tr.mean())).max() / (len(y_tr) * l1)
            for alpha in alpha_max * np.logspace(0, -4, 30):
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=100_000, tol=1e-6).fit(X_tr, y_tr)
                r2 = 1 - np.sum((y_va - m.predict(X_va)) ** 2) / np.sum(y_va ** 2)
                rows.append({"window": window, "val_share": VAL, "l1_ratio": l1, "alpha": alpha, "val_r2": r2,
                             "n_nonzero": int((m.coef_ != 0).sum())})
    check = pd.DataFrame(rows)
    check["chosen"] = check.groupby("window")["val_r2"].transform("max") == check["val_r2"]

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
             "same_rank_share": float(rank), "gap_best_to_second": float(top[0] - top[1])}
        out.append(r)
        print(f"{window} 30%: re-tune picks penalty {a['alpha']:.6e}, l1_ratio {a['l1_ratio']:g} "
              f"(val R2 {a['val_r2']:.6%}); 05 picked {b['alpha']:.6e}, l1_ratio {b['l1_ratio']:g} "
              f"(val R2 {b['val_r2']:.6%}) -> {'SAME' if same else 'DIFFERENT'}")
        print(f"  over the 90 grid points: penalties differ by at most {r['max_grid_alpha_rel_diff']:.1e} (relative; "
              f"05's table keeps 6 significant digits), "
              f"val R2 by at most {r['max_grid_val_r2_abs_diff']:.1e}; same number of nonzero coefficients at "
              f"{r['same_nonzero_share']:.0%}, same rank at {r['same_rank_share']:.0%}; best minus second-best val R2 "
              f"{r['gap_best_to_second']:.1e}")
    pd.DataFrame(out).to_csv(args.out, index=False, float_format="%.8g")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
