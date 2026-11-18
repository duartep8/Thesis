"""Step 7: audit of the elastic-net tuning (reads saved outputs; changes no model).

Usage:
    python src/07_tuning_audit.py [--model enet] [--panel data/panel.parquet] [--results results]
                                  [--test-end 2025]

Audits the windows and validation shares present in 05_<model>_summary.csv; --test-end
must be the last test year 05 was run with (once-trained fits are checked to predict up to it).

1. Sample-split table. For every setup, the training, validation and test months of
   its fits, rederived from the design (windows by return month; validation = the last
   share of the window's months, rounded half up), with the rows in each (training and
   validation: rows with a target, as in 05; test: rows predicted, and with a target).
   Yearly fits are audited twice: age 0 (the year after the build, used in Part 1) and
   age 1 (the year after that, used in Part 2), each with its own test months.
   Checks, for every fit: the rederived months and row counts equal 05's fit log; the
   training, validation and test months do not overlap; validation ends before the
   first test month, taken from the saved predictions; the settings stay those tuned
   at the first fit, and equal the chosen row of 05_<model>_tuning.csv. results/tables/07_<model>_audit.csv has one row per setup (its
   first fit, December 1989, plus the result of the checks over all its fits);
   07_<model>_audit_builds.csv has every fit.
2. Tuning figures. Validation R^2 against the penalty (log scale), one line per
   l1_ratio (LightGBM: one per min_child_samples), chosen point marked, grid points
   disqualified by the near-constant rule greyed out, the fallback stated if used, per
   window and validation share (from
   05_<model>_tuning.csv): results/tables/figures/07_<model>_tuning_<window>_val<share>.png.
   The 15-year and expanding windows share their December 1989 tuning, so their figures
   are the same.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIRST_RETURN = pd.Period("1975-01", "M")
WINDOW_ORDER = ["5y", "10y", "15y", "expanding"]
# Reference categorical palette (validated: scripts/validate_palette.js, light surface); each line also
# has its own marker shape, so identity never rests on colour alone
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DISQ = "#9b9a96"  # disqualified grid points (neutral grey, with its own X marker)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def split(window, val, build):
    """Training and validation return months of the fit built in December `build`."""
    first = FIRST_RETURN if window == "expanding" else pd.Period(f"{build + 1 - int(window[:-1])}-01", "M")
    months = pd.period_range(first, pd.Period(f"{build}-12", "M"), freq="M")
    n_val = int(np.floor(val * len(months) + 0.5))
    return months[:-n_val], months[-n_val:]


def fmt(months):
    return f"{months[0]:%Y-%m} to {months[-1]:%Y-%m}"


def audit(model, res, panel, test_end):
    rows_per_month = panel[panel["ret_exc_lead1m"].notna()].groupby("ret_month").size()
    fits = pd.read_csv(res / "tables" / f"05_{model}_fits.csv")
    fits["val_share"] = fits["val_share"].round(2)
    summary = pd.read_csv(res / "tables" / f"05_{model}_summary.csv")
    summary["val_share"] = summary["val_share"].round(2)
    target = panel.set_index(["id", "eom"])["ret_exc_lead1m"]
    tuning = pd.read_csv(res / "tables" / f"05_{model}_tuning.csv")
    tuning["val_share"] = tuning["val_share"].round(2)
    builds_out, setups_out = [], []
    for w in [w for w in WINDOW_ORDER if w in set(summary["window"])]:
        for v in sorted(summary["val_share"].unique()):
            preds = pd.read_parquet(res / "predictions" / model / f"{w}_val{round(v * 100)}.parquet",
                                    columns=["id", "eom", "ret_month", "freq", "build_year", "age", "settings"])
            preds["has_target"] = target.reindex(pd.MultiIndex.from_frame(preds[["id", "eom"]])).notna().to_numpy()
            preds["rm"] = pd.to_datetime(preds["ret_month"]).dt.to_period("M")
            tuned = json.loads(summary[(summary["window"] == w) & (summary["val_share"] == v)]["settings"].iloc[0])
            chosen = tuning[(tuning["window"] == w) & (tuning["val_share"] == v) & tuning["chosen"].astype(bool)]
            # summary settings against 05's tuning table (it stores 6 significant digits)
            match_tuning = bool(len(chosen) == 1 and all(
                np.isclose(float(chosen[k].iloc[0]), float(x), rtol=1e-5, atol=0) for k, x in tuned.items()))
            for freq, age in (("once", None), ("yearly", 0), ("yearly", 1)):
                p = preds[preds["freq"] == freq]
                if age is not None:
                    p = p[p["age"] == age]
                b_rows = []
                for b, pb in p.groupby("build_year"):
                    tr, va = split(w, v, b)
                    test = pd.PeriodIndex(sorted(pb["rm"].unique()))
                    lo_year = b + 1 + (age or 0)
                    expected = pd.period_range(f"{lo_year}-01", f"{lo_year if freq == 'yearly' else test_end}-12", freq="M")
                    log = fits[(fits["window"] == w) & (fits["val_share"] == v) & (fits["build_year"] == b)].iloc[0]
                    n_tr, n_va = int(rows_per_month.reindex(tr).sum()), int(rows_per_month.reindex(va).sum())
                    r = {"window": w, "freq": freq, "age": age if age is not None else "once", "val_share": v, "build_year": b,
                         "train": fmt(tr), "val": fmt(va), "test": fmt(test),
                         "train_months": len(tr), "val_months": len(va), "test_months": len(test),
                         "n_train": n_tr, "n_val": n_va, "n_test": len(pb), "n_test_with_target": int(pb["has_target"].sum()),
                         "matches_05_log": bool(log["train_months"] == len(tr) and log["val_months"] == len(va)
                                                and log["n_train"] == n_tr and log["n_val"] == n_va),
                         "no_overlap": bool(len(tr.intersection(va)) == 0 and len(tr.union(va).intersection(test)) == 0),
                         "val_before_test": bool(tr.max() < va.min() and va.max() < test.min()),
                         "test_as_designed": bool(test.equals(expected)),
                         "settings_fixed": bool((pb["settings"].map(json.loads) == tuned).all()),
                         "settings_match_tuning": match_tuning}
                    b_rows.append(r)
                    check(r["matches_05_log"] and r["no_overlap"] and r["val_before_test"] and r["test_as_designed"]
                          and r["settings_fixed"] and r["settings_match_tuning"], f"audit failed: {r}")
                if not b_rows:  # e.g. no age-1 fits when the test period is a single year
                    continue
                builds_out += b_rows
                first = dict(b_rows[0])
                first.update({"fits": len(b_rows), "all_fits_ok": True,
                              "last_fit_val_end": b_rows[-1]["val"][-7:], "last_fit_test": b_rows[-1]["test"]})
                setups_out.append(first)
    return pd.DataFrame(setups_out), pd.DataFrame(builds_out)


def figures(model, res):
    tuning = pd.read_csv(res / "tables" / f"05_{model}_tuning.csv")
    tuning["val_share"] = tuning["val_share"].round(2)
    out = res / "tables" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False})
    paths = []
    for (w, v), t in tuning.groupby(["window", "val_share"]):
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200, facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        c = t[t["chosen"]].iloc[0]
        rule = ""
        if model == "lgbm":
            # x: num_leaves at even spacing; one line per min_child_samples; labels give each point's tree count
            leaves = sorted(t["num_leaves"].unique())
            pos = {n: i for i, n in enumerate(leaves)}
            # at each num_leaves, the higher point's label goes above it and the lower one's below
            higher = t.groupby("num_leaves")["val_r2"].transform("max")
            for i, (mc, g) in enumerate(t.groupby("min_child_samples")):
                g = g.sort_values("num_leaves")
                x = g["num_leaves"].map(pos)
                ax.plot(x, g["val_r2"] * 100, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=0.6, label=f"min_child_samples = {mc}", zorder=3)
                for xi, r, top in zip(x, g.itertuples(), g["val_r2"] == higher[g.index]):
                    # coloured by line; on a tie the first line's label goes above, the second's below
                    above = top and not (i > 0 and (t.loc[t["num_leaves"] == r.num_leaves, "val_r2"] == r.val_r2).sum() > 1)
                    ax.annotate(f"{r.n_trees}", (xi, r.val_r2 * 100), textcoords="offset points",
                                xytext=(6, 4) if above else (6, -11), fontsize=7, color=SERIES[i], fontweight="bold")
            dq = t[~t["passes_near_constant"].astype(bool)]
            if len(dq):
                ax.plot(dq["num_leaves"].map(pos), dq["val_r2"] * 100, marker="X", markersize=8, color=DISQ,
                        markeredgecolor=SURFACE, linestyle="none", zorder=5, label="disqualified (near-constant rule)")
            ax.plot(pos[c["num_leaves"]], c["val_r2"] * 100, marker="o", markersize=12, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=1.6, zorder=4, linestyle="none", label="chosen")
            ax.set_xticks(range(len(leaves)), [str(n) for n in leaves])
            ax.set_xlim(-0.4, len(leaves) - 0.6)
            ax.text(0.98, 0.04, f"chosen (ring): num_leaves {int(c['num_leaves'])}, min_child_samples "
                    f"{int(c['min_child_samples'])}\n{int(c['n_trees'])} trees at learning rate {c['learning_rate']:g}, "
                    f"validation R² {c['val_r2']:.2%}, {int(c['n_nonzero'])} characteristics used",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
            ax.set_xlabel("num_leaves (labels: trees chosen by early stopping)")
            title = "LightGBM tuning"
            if isinstance(c.get("rule"), str) and c["rule"]:
                rule = f" Edge rule {c['rule']}."
        else:
            for i, (l1, g) in enumerate(t.groupby("l1_ratio")):
                g = g.sort_values("alpha")
                ax.plot(g["alpha"], g["val_r2"] * 100, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=4,
                        markeredgecolor=SURFACE, markeredgewidth=0.6, label=f"l1_ratio = {l1:g}", zorder=3)
            dq = t[~t["passes_near_constant"].astype(bool)]
            if len(dq):
                ax.plot(dq["alpha"], dq["val_r2"] * 100, marker="X", markersize=7, color=DISQ,
                        markeredgecolor=SURFACE, linestyle="none", zorder=5, label="disqualified (near-constant rule)")
            ax.plot(c["alpha"], c["val_r2"] * 100, marker="o", markersize=11, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=1.6, zorder=4, linestyle="none", label="chosen")
            ax.text(0.98, 0.04, f"chosen (ring): penalty {c['alpha']:.2e}, l1_ratio {c['l1_ratio']:g}\n"
                    f"validation R² {c['val_r2']:.2%}, {int(c['n_nonzero'])} nonzero coefficients",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
            ax.set_xscale("log")
            ax.set_xlabel("penalty (alpha), log scale")
            title = "Elastic net tuning"
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=2)
        ax.set_ylabel("validation R² (%)")
        same = " (same December 1989 tuning as the 15-year window)" if w == "expanding" else (
            " (same December 1989 tuning as the expanding window)" if w == "15y" else "")
        fallback = ("\nFallback used: no grid point passed the near-constant rule (most distinct taken)"
                    if bool(c["near_constant_rule_failed"]) else "")
        ax.set_title(f"{title}: {w} window, validation share {v:.0%}{fallback}", loc="left", fontsize=10, color=INK)
        n_dq = int((~t["passes_near_constant"].astype(bool)).sum())
        fig.text(0.01, 0.01, f"Fit of December 1989{same}. Validation R²: predictions demeaned within each validation "
                 f"month plus the training average return, against the raw target. Grey X: {n_dq} of {len(t)} grid "
                 f"points with fewer than 50 distinct validation predictions in some month (not eligible).{rule}",
                 fontsize=6.5, color=INK_2, wrap=True)
        ax.legend(frameon=False, fontsize=8, loc="lower left" if model == "lgbm" else "upper left")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        path = out / f"07_{model}_tuning_{w}_val{round(v * 100)}.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="enet")
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--test-end", type=int, default=2025, help="last test year of the 05 run (default 2025)")
    args = parser.parse_args()
    res = Path(args.results)

    panel = pd.read_parquet(args.panel, columns=["id", "eom", "ret_exc_lead1m"])
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel["ret_month"] = (panel["eom"] + pd.offsets.MonthEnd(1)).dt.to_period("M")
    setups, builds = audit(args.model, res, panel, args.test_end)
    cols = ["window", "freq", "age", "val_share", "train", "val", "test", "train_months", "val_months", "test_months",
            "n_train", "n_val", "n_test", "n_test_with_target", "fits", "last_fit_val_end", "last_fit_test",
            "matches_05_log", "no_overlap", "val_before_test", "test_as_designed", "settings_fixed",
            "settings_match_tuning", "all_fits_ok"]
    setups[cols].to_csv(res / "tables" / f"07_{args.model}_audit.csv", index=False)
    builds.to_csv(res / "tables" / f"07_{args.model}_audit_builds.csv", index=False)

    print(f"Sample splits, first fit (December 1989) of each setup; checks over all {len(builds)} fits")
    print(f"  {'window':<10}{'freq':<7}{'age':>4}{'val':>5}  {'training':<19}{'validation':<19}{'test':<19}"
          f"{'n train':>9}{'n val':>8}{'n test':>9}{'w/ target':>10}{'fits':>5}  checks")
    for r in setups.itertuples():
        print(f"  {r.window:<10}{r.freq:<7}{str(r.age):>4}{r.val_share:>5.2f}  {r.train:<19}{r.val:<19}{r.test:<19}"
              f"{r.n_train:>9,}{r.n_val:>8,}{r.n_test:>9,}{r.n_test_with_target:>10,}{r.fits:>5}  "
              + ("all ok" if r.all_fits_ok else "FAILED"))
    print("  checks per fit: months and rows equal 05's fit log; training, validation and test months disjoint; "
          "validation ends before the first test month (from the saved predictions); test months as designed; "
          "settings fixed at the first fit's tuning and equal to the chosen row of 05's tuning table; yearly fits at "
          "age 0 (the year after the build) and age 1 (the year after that, Part 2)")
    paths = figures(args.model, res)
    print(f"\nSaved {res / 'tables'}/07_{args.model}_audit.csv, 07_{args.model}_audit_builds.csv and "
          f"{len(paths)} figures in {res / 'tables' / 'figures'}/")


if __name__ == "__main__":
    main()
