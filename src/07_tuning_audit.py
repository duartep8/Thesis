"""Step 7: audit of the elastic-net tuning (reads saved outputs; changes no model).

Usage:
    python src/07_tuning_audit.py [--model enet] [--panel data/panel.parquet] [--results results]

1. Sample-split table. For every setup, the training, validation and test months of
   its fits, rederived from the design (windows by return month; validation = the last
   share of the window's months, rounded half up), with the rows in each (training and
   validation: rows with a target, as in 05; test: rows predicted, and with a target).
   Checks, for every fit: the rederived months and row counts equal 05's fit log; the
   training, validation and test months do not overlap; validation ends before the
   first test month, taken from the saved predictions; the settings stay those tuned
   at the first fit. results/tables/07_<model>_audit.csv has one row per setup (its
   first fit, December 1989, plus the result of the checks over all its fits);
   07_<model>_audit_builds.csv has every fit.
2. Tuning figures. Validation R^2 against the penalty (log scale), one line per
   l1_ratio, chosen point marked, per window and validation share (from
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
TEST_START, TEST_END = 1990, 2025
WINDOW_ORDER = ["5y", "10y", "15y", "expanding"]
# Reference categorical palette (validated: scripts/validate_palette.js, light surface); each line also
# has its own marker shape, so identity never rests on colour alone
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


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


def audit(model, res, panel):
    rows_per_month = panel[panel["ret_exc_lead1m"].notna()].groupby("ret_month").size()
    fits = pd.read_csv(res / "tables" / f"05_{model}_fits.csv")
    fits["val_share"] = fits["val_share"].round(2)
    summary = pd.read_csv(res / "tables" / f"05_{model}_summary.csv")
    summary["val_share"] = summary["val_share"].round(2)
    target = panel.set_index(["id", "eom"])["ret_exc_lead1m"]
    builds_out, setups_out = [], []
    for w in WINDOW_ORDER:
        for v in sorted(summary["val_share"].unique()):
            preds = pd.read_parquet(res / "predictions" / model / f"{w}_val{round(v * 100)}.parquet",
                                    columns=["id", "eom", "ret_month", "freq", "build_year", "age", "settings"])
            preds["has_target"] = target.reindex(pd.MultiIndex.from_frame(preds[["id", "eom"]])).notna().to_numpy()
            preds["rm"] = pd.to_datetime(preds["ret_month"]).dt.to_period("M")
            tuned = json.loads(summary[(summary["window"] == w) & (summary["val_share"] == v)]["settings"].iloc[0])
            for freq in ("once", "yearly"):
                p = preds[preds["freq"] == freq]
                if freq == "yearly":
                    p = p[p["age"] == 0]
                b_rows = []
                for b, pb in p.groupby("build_year"):
                    tr, va = split(w, v, b)
                    test = pd.PeriodIndex(sorted(pb["rm"].unique()))
                    expected = pd.period_range(f"{b + 1}-01", f"{b + 1 if freq == 'yearly' else TEST_END}-12", freq="M")
                    log = fits[(fits["window"] == w) & (fits["val_share"] == v) & (fits["build_year"] == b)].iloc[0]
                    n_tr, n_va = int(rows_per_month.reindex(tr).sum()), int(rows_per_month.reindex(va).sum())
                    r = {"window": w, "freq": freq, "val_share": v, "build_year": b,
                         "train": fmt(tr), "val": fmt(va), "test": fmt(test),
                         "train_months": len(tr), "val_months": len(va), "test_months": len(test),
                         "n_train": n_tr, "n_val": n_va, "n_test": len(pb), "n_test_with_target": int(pb["has_target"].sum()),
                         "matches_05_log": bool(log["train_months"] == len(tr) and log["val_months"] == len(va)
                                                and log["n_train"] == n_tr and log["n_val"] == n_va),
                         "no_overlap": bool(len(tr.intersection(va)) == 0 and len(tr.union(va).intersection(test)) == 0),
                         "val_before_test": bool(tr.max() < va.min() and va.max() < test.min()),
                         "test_as_designed": bool(test.equals(expected)),
                         "settings_fixed": bool((pb["settings"].map(json.loads) == tuned).all())}
                    b_rows.append(r)
                    check(r["matches_05_log"] and r["no_overlap"] and r["val_before_test"] and r["test_as_designed"]
                          and r["settings_fixed"], f"audit failed: {r}")
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
        for i, (l1, g) in enumerate(t.groupby("l1_ratio")):
            g = g.sort_values("alpha")
            ax.plot(g["alpha"], g["val_r2"] * 100, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, label=f"l1_ratio = {l1:g}", zorder=3)
        c = t[t["chosen"]].iloc[0]
        ax.plot(c["alpha"], c["val_r2"] * 100, marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.6, zorder=4, linestyle="none", label="chosen")
        ax.text(0.98, 0.04, f"chosen (ring): penalty {c['alpha']:.2e}, l1_ratio {c['l1_ratio']:g}\n"
                f"validation R² {c['val_r2']:.2%}, {int(c['n_nonzero'])} nonzero coefficients",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK)
        ax.axhline(0, color=INK_2, linewidth=0.8, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("penalty (alpha), log scale")
        ax.set_ylabel("validation R² (%)")
        same = " (same December 1989 tuning as the 15-year window)" if w == "expanding" else (
            " (same December 1989 tuning as the expanding window)" if w == "15y" else "")
        ax.set_title(f"Elastic net tuning: {w} window, validation share {v:.0%}", loc="left", fontsize=10, color=INK)
        fig.text(0.01, 0.01, f"Fit of December 1989; validation R² without demeaning{same}.", fontsize=7, color=INK_2)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
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
    args = parser.parse_args()
    res = Path(args.results)

    panel = pd.read_parquet(args.panel, columns=["id", "eom", "ret_exc_lead1m"])
    panel["eom"] = pd.to_datetime(panel["eom"])
    panel["ret_month"] = (panel["eom"] + pd.offsets.MonthEnd(1)).dt.to_period("M")
    setups, builds = audit(args.model, res, panel)
    cols = ["window", "freq", "val_share", "train", "val", "test", "train_months", "val_months", "test_months",
            "n_train", "n_val", "n_test", "n_test_with_target", "fits", "last_fit_val_end", "last_fit_test",
            "matches_05_log", "no_overlap", "val_before_test", "test_as_designed", "settings_fixed", "all_fits_ok"]
    setups[cols].to_csv(res / "tables" / f"07_{args.model}_audit.csv", index=False)
    builds.to_csv(res / "tables" / f"07_{args.model}_audit_builds.csv", index=False)

    print(f"Sample splits, first fit (December 1989) of each setup; checks over all {len(builds)} fits")
    print(f"  {'window':<10}{'freq':<7}{'val':>5}  {'training':<19}{'validation':<19}{'test':<19}"
          f"{'n train':>9}{'n val':>8}{'n test':>9}{'w/ target':>10}{'fits':>5}  checks")
    for r in setups.itertuples():
        print(f"  {r.window:<10}{r.freq:<7}{r.val_share:>5.2f}  {r.train:<19}{r.val:<19}{r.test:<19}"
              f"{r.n_train:>9,}{r.n_val:>8,}{r.n_test:>9,}{r.n_test_with_target:>10,}{r.fits:>5}  "
              + ("all ok" if r.all_fits_ok else "FAILED"))
    print("  checks per fit: months and rows equal 05's fit log; training, validation and test months disjoint; "
          "validation ends before the first test month (from the saved predictions); test months as designed; "
          "settings fixed at the first fit's tuning")
    paths = figures(args.model, res)
    print(f"\nSaved results/tables/07_{args.model}_audit.csv, 07_{args.model}_audit_builds.csv and "
          f"{len(paths)} figures in results/tables/figures/")


if __name__ == "__main__":
    main()
