"""Step 7b: independent re-tune of the December 1989 tunings (elastic net: all 9; LightGBM: expanding 30%).

Usage:
    python src/07b_retune_check.py [--panel data/panel.parquet] [--results results] [--skip-lgbm]

Written separately from 05_train.py and shares no code with it: the target, the windows, the
validation split, the validation R^2, the near-constant rule and (for LightGBM) early stopping and
the tree-count rule are all rebuilt here from the protocol in CLAUDE.md (2026-11-16):
- target: ret_exc_lead1m minus its mean over the month's rows with a target; rows sorted by
  (eom, id), as the protocol's data order (LightGBM's bagging samples rows by position);
- windows by return month (= eom + 1 month), ending December 1989: 5y from 1985-01, 10y from
  1980-01, 15y and expanding from 1975-01 (the same tuning); validation = the last share of the
  months, rounded half up;
- validation R^2: predictions demeaned within each validation month (over its validation rows),
  plus the training part's average raw return, against the raw target: 1 - sum (y - f)^2 / sum y^2;
- near-constant rule: a grid point qualifies only with at least 50 distinct validation predictions
  in every month; best qualifying point by validation R^2, else the most distinct (then R^2), flagged.
Elastic net: scikit-learn's ElasticNet, one fresh fit per grid point (05 uses a warm-started path
with a Gram matrix); l1_ratio 0.1, 0.5, 0.9, 30 penalties from the all-zero penalty down to 1e-4
of it; max_iter and tol must equal 05's (checked in 05's source text). BLAS limited to THREADS.
LightGBM (expanding 30%): plain lightgbm Booster grown one tree at a time; validation R^2 tracked
tree by tree; stop after 50 rounds without a strict improvement (at most 1,000 trees); grid
num_leaves {7, 15, 31} x min_child_samples {200, 1000} at learning rate 0.05; if the chosen
point has fewer than 20 trees, the grid is rerun at 0.01, at the 1,000 cap at 0.1.
Compares every grid point and the choice with 05's tuning tables and its fit log (row counts).
A mismatch is reported as a finding (columns *_matches / same_*), not raised.
Writes <results>/tables/07b_enet_retune_check.csv and 07b_lgbm_retune_check.csv.
"""

import argparse
import json
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
LAST = "1989-12"
FIRSTS = {"5y": "1985-01", "10y": "1980-01", "15y/expanding": "1975-01"}  # first return month, December 1989 window
SHARES = (0.25, 0.30, 0.35)
MIN_DISTINCT = 50
MAX_ITER, TOL = 100_000, 1e-6  # must equal 05_train.ElasticNet (checked below)
THREADS = 4
LGBM = {"setup": ("15y/expanding", 0.30), "leaves": (7, 15, 31), "min_child": (200, 1000), "lr": 0.05,
        "max_trees": 1000, "patience": 50, "min_trees": 20, "lr_few": 0.01, "lr_cap": 0.1}


def load(panel, chars):
    df = pd.read_parquet(panel, columns=["id", "eom", "ret_exc_lead1m"] + chars)
    df = df.dropna(subset=["ret_exc_lead1m"]).sort_values(["eom", "id"], ignore_index=True)
    df["ret_month"] = (pd.to_datetime(df["eom"]) + pd.offsets.MonthEnd(1)).dt.to_period("M")
    df["y_dm"] = df["ret_exc_lead1m"] - df.groupby("ret_month")["ret_exc_lead1m"].transform("mean")
    return df


def split(df, chars, first, share):
    months = pd.period_range(first, LAST, freq="M")
    n_val = int(np.floor(share * len(months) + 0.5))
    tr, va = df[df["ret_month"].isin(months[:-n_val])], df[df["ret_month"].isin(months[-n_val:])]
    return {"X_tr": tr[chars].to_numpy(float), "y_tr": tr["y_dm"].to_numpy(float),
            "X_va": va[chars].to_numpy(float), "y_va": va["ret_exc_lead1m"].to_numpy(float),
            "c": tr["ret_exc_lead1m"].mean(), "month_va": va["ret_month"].to_numpy(),
            "label": f"train {months[0]} to {months[-n_val - 1]} ({len(tr):,} rows), "
                     f"validation {months[-n_val]} to {months[-1]} ({len(va):,} rows)",
            "n_train": len(tr), "n_val": len(va)}


def scorer(s):
    """Validation R^2 and the fewest distinct predictions in a validation month, for raw model output f."""
    month = pd.Series(s["month_va"])

    def score(f):
        f = pd.Series(f)
        g = f.groupby(month)
        r2 = 1 - np.sum((s["y_va"] - (f - g.transform("mean")).to_numpy() - s["c"]) ** 2) / np.sum(s["y_va"] ** 2)
        return float(r2), int(g.nunique().min())
    return score


def pick(rows):
    """Near-constant rule, then the best validation R^2; marks rows in place."""
    ok = [r for r in rows if r["min_distinct"] >= MIN_DISTINCT]
    best = max(ok, key=lambda r: r["val_r2"]) if ok else max(rows, key=lambda r: (r["min_distinct"], r["val_r2"]))
    for r in rows:
        r["chosen"], r["fallback"] = r is best, not ok
    return best


def ours_rows(table, window, share):
    """05's rows for this tuning (15y and expanding share theirs; both are returned for comparison)."""
    names = ["15y", "expanding"] if window == "15y/expanding" else [window]
    return [table[(table["window"] == w) & np.isclose(table["val_share"], share)].reset_index(drop=True) for w in names]


def fit_log_counts(fits, window, share):
    names = ["15y", "expanding"] if window == "15y/expanding" else [window]
    r = fits[fits["window"].isin(names) & np.isclose(fits["val_share"], share) & (fits["build_year"] == 1989)]
    return set(zip(r["n_train"], r["n_val"]))


def enet_check(df, chars, tables, out):
    ours, fits = pd.read_csv(tables / "05_enet_tuning.csv"), pd.read_csv(tables / "05_enet_fits.csv")
    result = []
    for window, first in FIRSTS.items():
        for share in SHARES:
            s = split(df, chars, first, share)
            score = scorer(s)
            print(f"{window} {share:.0%}: {s['label']}", flush=True)
            rows = []
            for l1 in (0.1, 0.5, 0.9):
                Xc = s["X_tr"] - s["X_tr"].mean(0)
                alpha_max = np.abs(Xc.T @ (s["y_tr"] - s["y_tr"].mean())).max() / (len(s["y_tr"]) * l1)
                for k, alpha in enumerate(alpha_max * np.logspace(0, -4, 30)):
                    t0 = time.time()
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always", ConvergenceWarning)
                        m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=MAX_ITER, tol=TOL).fit(s["X_tr"], s["y_tr"])
                    warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
                    r2, md = score(m.predict(s["X_va"]))
                    rows.append({"l1_ratio": l1, "alpha": alpha, "val_r2": r2, "min_distinct": md,
                                 "n_nonzero": int((m.coef_ != 0).sum()), "n_iter": int(m.n_iter_), "not_converged": warned})
                    print(f"  l1 {l1:g} point {k + 1:2d}/30  penalty {alpha:.3e}  val R2 {r2:.4%}  min distinct {md:4d}  "
                          f"nonzero {rows[-1]['n_nonzero']:3d}  iterations {m.n_iter_:6d}"
                          f"{'  NOT CONVERGED' if warned else ''}  {time.time() - t0:5.1f}s", flush=True)
            best = pick(rows)
            c = pd.DataFrame(rows)
            o_all = ours_rows(ours, window, share)
            o = o_all[0]
            grid_ok = (len(o) == len(c) and np.allclose(o["l1_ratio"], c["l1_ratio"])
                       and np.allclose(o["alpha"], c["alpha"], rtol=1e-5, atol=0))  # 05's table keeps 6 digits
            ob = o[o["chosen"].astype(bool)].iloc[0] if len(o) and o["chosen"].astype(bool).any() else None
            r = {"window": window, "val_share": share, "n_train": s["n_train"], "n_val": s["n_val"],
                 "rows_match_05_fit_log": fit_log_counts(fits, window, share) == {(s["n_train"], s["n_val"])},
                 "same_tuning_15y_expanding": (len(o_all) == 1 or o_all[0].drop(columns="window").equals(o_all[1].drop(columns="window"))),
                 "grid_matches": bool(grid_ok),
                 "chosen_alpha": best["alpha"], "chosen_l1_ratio": best["l1_ratio"], "chosen_val_r2": best["val_r2"],
                 "chosen_n_nonzero": best["n_nonzero"], "fallback": best["fallback"],
                 "ours_alpha": ob["alpha"] if ob is not None else np.nan,
                 "ours_l1_ratio": ob["l1_ratio"] if ob is not None else np.nan,
                 "ours_val_r2": ob["val_r2"] if ob is not None else np.nan,
                 "same_choice": bool(ob is not None and np.isclose(best["l1_ratio"], ob["l1_ratio"])
                                     and np.isclose(best["alpha"], ob["alpha"], rtol=1e-5, atol=0)),
                 "max_grid_val_r2_abs_diff": float(np.max(np.abs(c["val_r2"] - o["val_r2"]))) if grid_ok else np.nan,
                 "same_nonzero_share": float((c["n_nonzero"] == o["n_nonzero"]).mean()) if grid_ok else np.nan,
                 "same_disqualified": bool(((c["min_distinct"] >= MIN_DISTINCT) == o["passes_near_constant"].astype(bool)).all())
                 if grid_ok else False,
                 "same_rank_share": float((c["val_r2"].rank(ascending=False) == o["val_r2"].rank(ascending=False)).mean())
                 if grid_ok else np.nan,
                 "gap_best_to_second_qualifying": float(np.diff(np.sort(c.loc[c["min_distinct"] >= MIN_DISTINCT, "val_r2"])[-2:])[0]),
                 "max_n_iter": int(c["n_iter"].max()), "n_not_converged": int(c["not_converged"].sum())}
            result.append(r)
            print(f"  -> re-tune: penalty {r['chosen_alpha']:.6e}, l1 {r['chosen_l1_ratio']:g}, val R2 {r['chosen_val_r2']:.6%}; "
                  f"05: {r['ours_alpha']:.6e}, l1 {r['ours_l1_ratio']:g}, val R2 {r['ours_val_r2']:.6%} -> "
                  f"{'SAME' if r['same_choice'] else 'DIFFERENT'}; grid {'matches' if grid_ok else 'DOES NOT MATCH'}; "
                  f"rows match 05's fit log: {r['rows_match_05_fit_log']}; max val R2 diff {r['max_grid_val_r2_abs_diff']:.1e}; "
                  f"same rank {r['same_rank_share']:.0%}; same disqualified points: {r['same_disqualified']}; "
                  f"most iterations {r['max_n_iter']:,}, not converged {r['n_not_converged']}", flush=True)
    pd.DataFrame(result).to_csv(out, index=False, float_format="%.8g")
    print(f"Saved {out}\n", flush=True)


def lgbm_grid(s, score, lr):
    import lightgbm as lgb

    rows = []
    for leaves in LGBM["leaves"]:
        for min_child in LGBM["min_child"]:
            t0 = time.time()
            p = {"objective": "regression", "learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child,
                 "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42, "deterministic": True,
                 "force_col_wise": True, "num_threads": 8, "verbose": -1}
            booster = lgb.Booster(params=p, train_set=lgb.Dataset(s["X_tr"], s["y_tr"], params=p))
            f = np.zeros(len(s["y_va"]))
            best_r2, best_n = -np.inf, 0
            for n in range(1, LGBM["max_trees"] + 1):
                booster.update()
                f += booster.predict(s["X_va"], start_iteration=n - 1, num_iteration=1)  # the new tree's contribution
                r2, _ = score(f)
                if r2 > best_r2:
                    best_r2, best_n = r2, n
                if n - best_n >= LGBM["patience"]:
                    break
            f_best = booster.predict(s["X_va"], num_iteration=best_n)
            r2, md = score(f_best)
            rows.append({"learning_rate": lr, "num_leaves": leaves, "min_child_samples": min_child, "n_trees": best_n,
                         "val_r2": r2, "min_distinct": md, "trees_grown": n,
                         "tree_sum_vs_predict": float(np.max(np.abs(booster.predict(s["X_va"], num_iteration=n) - f)))})
            print(f"  lr {lr:g} leaves {leaves:2d} min_child {min_child:4d}: {best_n:4d} trees, val R2 {r2:.4%}, "
                  f"min distinct {md}, grew {n}, {time.time() - t0:5.1f}s", flush=True)
    return rows


def lgbm_check(df, chars, tables, out):
    window, share = LGBM["setup"]
    s = split(df, chars, FIRSTS[window], share)
    score = scorer(s)
    print(f"LightGBM {window} {share:.0%}: {s['label']}", flush=True)
    rows = lgbm_grid(s, score, LGBM["lr"])
    best = pick(rows)
    rule = ""
    if best["n_trees"] < LGBM["min_trees"] or best["n_trees"] >= LGBM["max_trees"]:
        lr = LGBM["lr_few"] if best["n_trees"] < LGBM["min_trees"] else LGBM["lr_cap"]
        rule = f"fired: {best['n_trees']} trees at {LGBM['lr']:g}, rerun at {lr:g}"
        print(f"  tree-count rule {rule}", flush=True)
        rows = lgbm_grid(s, score, lr)
        best = pick(rows)
    c = pd.DataFrame(rows)
    ours = pd.read_csv(tables / "05_lgbm_tuning.csv")
    o_all = ours_rows(ours, window, share)
    o = o_all[0]
    key = ["learning_rate", "num_leaves", "min_child_samples"]
    m = c.merge(o[key + ["n_trees", "val_r2", "min_distinct", "chosen", "rule"]], on=key, how="outer",
                suffixes=("", "_05"), indicator=True)
    m["grid_point_in_both"] = m["_merge"] == "both"
    m["same_n_trees"] = m["n_trees"] == m["n_trees_05"]
    m["val_r2_abs_diff"] = (m["val_r2"] - m["val_r2_05"]).abs()
    m["same_min_distinct"] = m["min_distinct"] == m["min_distinct_05"]
    m = m.drop(columns="_merge")
    m["rule_07b"] = rule
    m["rows_match_05_fit_log"] = fit_log_counts(pd.read_csv(tables / "05_lgbm_fits.csv"), window, share) == {
        (s["n_train"], s["n_val"])}
    m.to_csv(out, index=False, float_format="%.8g")
    ob = o[o["chosen"].astype(bool)].iloc[0] if o["chosen"].astype(bool).any() else None
    same = ob is not None and all(best[k] == ob[k] for k in key + ["n_trees"])
    print(f"  -> re-tune: leaves {best['num_leaves']}, min_child {best['min_child_samples']}, lr {best['learning_rate']:g}, "
          f"{best['n_trees']} trees, val R2 {best['val_r2']:.6%} (rule: {rule or 'not fired'}); 05: leaves {ob['num_leaves']}, "
          f"min_child {ob['min_child_samples']}, lr {ob['learning_rate']:g}, {ob['n_trees']} trees, val R2 {ob['val_r2']:.6%} "
          f"(rule: {ob['rule'] if isinstance(ob['rule'], str) and ob['rule'] else 'not fired'}) -> {'SAME' if same else 'DIFFERENT'}")
    print(f"  grid points in both: {int(m['grid_point_in_both'].sum())}/{len(m)}; same trees {int(m['same_n_trees'].sum())}/{len(m)}; "
          f"same min distinct {int(m['same_min_distinct'].sum())}/{len(m)}; max val R2 diff {m['val_r2_abs_diff'].max():.1e}; "
          f"rows match 05's fit log: {bool(m['rows_match_05_fit_log'].iloc[0])}; "
          f"max |tree sum - predict| {m['tree_sum_vs_predict'].max():.1e}")
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--skip-lgbm", action="store_true", help="elastic-net check only")
    args = parser.parse_args()
    src = (ROOT / "src" / "05_train.py").read_text()
    m05 = re.search(r"MAX_ITER, TOL = ([\d_]+), ([\de.-]+)", src)
    assert m05 and (int(m05[1]), float(m05[2])) == (MAX_ITER, TOL), "convergence settings differ from 05"
    chars = [l.strip() for l in (ROOT / "src" / "characteristics_153.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    tables = Path(args.results) / "tables"
    t0 = time.time()
    df = load(args.panel, chars)
    with threadpool_limits(THREADS):
        enet_check(df, chars, tables, tables / "07b_enet_retune_check.csv")
    print(f"elastic net done in {time.time() - t0:,.0f}s", flush=True)
    if not args.skip_lgbm:
        t1 = time.time()
        lgbm_check(df, chars, tables, tables / "07b_lgbm_retune_check.csv")
        print(f"LightGBM done in {time.time() - t1:,.0f}s")


if __name__ == "__main__":
    main()
