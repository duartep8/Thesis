"""Step 5: train return models over the grid of window x validation share x frequency.

Usage:
    python src/05_train.py --model enet [--windows 5 10 15 expanding] [--val 0.30 0.25 0.35]
                           [--test-start 1990] [--test-end 2025] [--cross-fit]
                           [--panel data/panel.parquet] [--clean data/clean.parquet] [--out results]

Inputs: the 153 ranked characteristics; target: ret_exc_lead1m. Every row is labelled by
the month of the return it predicts (ret_month = eom + 1 month). Rows with a missing
target are dropped from training, validation and evaluation, but every row is predicted.

For test year Y the model is built in December Y-1 on a window of return months:
rolling W years = January Y-W to December Y-1; expanding = January 1975 to December Y-1.
The last val share of the window's months (rounded to whole months) is validation; the
model is fit on the rest only. Every return used is realized by December Y-1.
  - yearly: a model is built every December from Y_start-1 to Y_end-1 and predicts the
    next year (age 0) and, for Part 2 (A vs C), the year after (age 1)
  - once: the first model (December Y_start-1) predicts every test year unchanged; it is
    the same fit as yearly's first, not refit
Settings are tuned by validation R^2 (Gu, Kelly & Xiu 2020, no demeaning) at a setup's
first fit and kept for its later refits. Fits with identical data and settings are run
once (e.g. the 15-year and expanding windows in December 1989).

Each model class has tune(), fit(), predict() and save(); add LightGBM and the neural
network to MODELS. Outputs (an existing predictions file is not refit, only evaluated):
  results/predictions/<model>/<window>_val<share>.parquet  both frequencies, ages 0 and 1
  results/models/<model>/<window>_val<share>_<build year>.*
  results/portfolios/<model>_<window>_val<share>_<freq>_{monthly,holdings}.parquet
  results/tables/05_<model>_{summary,tuning,fits}.csv      aggregates only
The report covers return months Y_start to Y_end: out-of-sample R^2, average monthly
rank IC, and the long-short portfolio from 04_portfolios.py (missing returns = 0).

--cross-fit (diagnostic only, after the grid): for each window and each pair of validation
shares (a, b), refit every build on a's training part with b's tuned settings (from
05_<model>_summary.csv). This separates the effect of less training data from that of
different tuning; no reported setup changes. 06_evaluate.py evaluates these:
  results/predictions/<model>/diagnostic/<window>_val<a>_set<b>.parquet
  results/models/<model>/diagnostic/, results/tables/05_<model>_crossfit_fits.csv
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHARS_153_FILE = ROOT / "src" / "characteristics_153.txt"
SEED = 42
WINDOWS = ["5", "10", "15", "expanding"]
VAL_SHARES = [0.30, 0.25, 0.35]


def load_src(name):
    """Import a numbered script from src/ (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def r2_oos(y, f):
    """Gu, Kelly & Xiu (2020): 1 - sum (y - f)^2 / sum y^2, with no demeaning."""
    return 1.0 - np.sum((y - f) ** 2) / np.sum(y ** 2)


class ElasticNet:
    """Elastic net with an intercept (scikit-learn's objective). Settings: l1_ratio, alpha."""
    name = "enet"
    L1_RATIOS = [0.1, 0.5, 0.9]
    N_ALPHAS, ALPHA_RANGE = 30, 1e-4  # alphas from the all-zero value down to 1e-4 of it, log-spaced
    MAX_ITER, TOL = 100_000, 1e-6

    def tune(self, X_tr, y_tr, X_va, y_va, seed):
        from sklearn.linear_model import enet_path

        x_mean, y_mean = X_tr.mean(axis=0), y_tr.mean()
        Xc, yc = X_tr - x_mean, y_tr - y_mean
        gram, xy = Xc.T @ Xc, Xc.T @ yc
        rows = []
        for l1 in self.L1_RATIOS:
            alpha_max = np.abs(xy).max() / (len(yc) * l1)
            alphas = alpha_max * np.logspace(0, np.log10(self.ALPHA_RANGE), self.N_ALPHAS)
            alphas, coefs, _ = enet_path(Xc, yc, l1_ratio=l1, alphas=alphas, precompute=gram, Xy=xy,
                                         max_iter=self.MAX_ITER, tol=self.TOL)
            for a, c in zip(alphas, coefs.T):
                f = y_mean + (X_va - x_mean) @ c
                rows.append({"l1_ratio": l1, "alpha": float(a), "val_r2": r2_oos(y_va, f), "n_nonzero": int((c != 0).sum())})
        best = max(rows, key=lambda r: r["val_r2"])
        return {"l1_ratio": best["l1_ratio"], "alpha": best["alpha"]}, rows

    def fit(self, X, y, settings, seed):
        from sklearn.linear_model import ElasticNet as SkElasticNet

        model = SkElasticNet(alpha=settings["alpha"], l1_ratio=settings["l1_ratio"], precompute=True,
                             max_iter=self.MAX_ITER, tol=self.TOL, random_state=seed)
        return model.fit(X, y)

    def predict(self, model, X):
        return model.predict(X)

    def n_nonzero(self, model):
        return int((model.coef_ != 0).sum())

    def save(self, model, path, chars):
        coef = pd.DataFrame({"name": ["intercept"] + chars, "coef": [model.intercept_] + list(model.coef_)})
        coef.to_csv(path.with_suffix(".csv"), index=False)


MODELS = {"enet": ElasticNet}


def window_label(window):
    return "expanding" if window == "expanding" else f"{window}y"


def setup_name(window, val):
    return f"{window_label(window)}_val{round(val * 100)}"


def month_index(dates):
    return (dates.dt.year * 12 + dates.dt.month - 1).to_numpy()


def run_setup(model, data, window, val, builds, test_end, out, cache, fit_log, tuning_log, fixed_settings=None, tag=""):
    """Fit one (window, validation share) over all build years; returns its predictions.

    fixed_settings: use these settings instead of tuning (diagnostic refits, saved with tag).
    """
    X, y, has_y, m_idx, ret_year, first_idx = (data[k] for k in ("X", "y", "has_y", "m_idx", "ret_year", "first_idx"))
    name = setup_name(window, val) + tag
    model_dir = out / "models" / model.name / ("diagnostic" if fixed_settings is not None else "")
    settings, parts = fixed_settings, []
    for b in builds:
        t0 = time.time()
        lo = first_idx if window == "expanding" else (b + 1 - int(window)) * 12
        hi = b * 12 + 11
        check(lo >= first_idx, f"{name}: window for build {b} starts before the data")
        n_val = int(np.floor(val * (hi - lo + 1) + 0.5))
        split = hi - n_val
        tr = has_y & (m_idx >= lo) & (m_idx <= split)
        va = has_y & (m_idx > split) & (m_idx <= hi)
        check(m_idx[tr | va].max() <= hi, "look-ahead: a training return after the build month")

        if settings is None:
            key = ("tune", lo, split, hi)
            if key not in cache:
                cache[key] = model.tune(X[tr], y[tr], X[va], y[va], SEED)
            settings, rows = cache[key]
            best = max(rows, key=lambda r: r["val_r2"])
            tuning_log += [{"window": window_label(window), "val_share": val, **r, "chosen": r is best} for r in rows]
        key = ("fit", lo, split, hi, json.dumps(settings, sort_keys=True))
        if key not in cache:
            cache[key] = model.fit(X[tr], y[tr], settings, SEED)
        fitted = cache[key]
        model.save(fitted, model_dir / f"{name}_{b}", data["chars"])

        pred = model.predict(fitted, X).astype("float32")
        settings_json = json.dumps(settings, sort_keys=True)

        def frame(sel, freq):
            return pd.DataFrame({"id": data["id"][sel], "eom": data["eom"][sel], "ret_month": data["ret_month"][sel],
                                 "pred": pred[sel], "model": model.name, "window": window_label(window),
                                 "val_share": val, "freq": freq, "build_year": b,
                                 "age": (ret_year[sel] - b - 1).astype("int16"), "settings": settings_json, "seed": SEED})

        for age in (0, 1):
            if b + 1 + age <= test_end:
                parts.append(frame(ret_year == b + 1 + age, "yearly"))
        if b == builds[0]:
            parts.append(frame((ret_year > b) & (ret_year <= test_end), "once"))
        fit_log.append({"window": window_label(window), "val_share": val, "build_year": b,
                        "train_months": split - lo + 1, "val_months": n_val, "n_train": int(tr.sum()), "n_val": int(va.sum()),
                        "val_r2": r2_oos(y[va], pred[va]), "n_nonzero": model.n_nonzero(fitted), "seconds": time.time() - t0})
        f = fit_log[-1]
        print(f"  {name} build {b}: train {f['train_months']} months ({f['n_train']:,} rows), val {n_val} months, "
              f"val R2 {f['val_r2']:.4f}, {f['n_nonzero']} nonzero, {f['seconds']:.1f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def evaluate(p04, pf_panel, preds, target, model_name, name, freq, out):
    """R^2, rank IC and the long-short portfolio of one setup's predictions."""
    p = preds[preds["freq"] == freq]
    if freq == "yearly":
        p = p[p["age"] == 0]
    p = p.merge(target, on=["id", "eom"], how="left")
    ok = p["ret_exc_lead1m"].notna()
    y, f = p.loc[ok, "ret_exc_lead1m"].to_numpy(float), p.loc[ok, "pred"].to_numpy(float)
    ic = p[ok].groupby("ret_month")[["pred", "ret_exc_lead1m"]].apply(
        lambda g: g["pred"].rank().corr(g["ret_exc_lead1m"].rank()))
    monthly, hold = p04.portfolio(p[["id", "eom", "pred"]], pf_panel)
    stem = out / "portfolios" / f"{model_name}_{name}_{freq}"
    monthly.to_parquet(f"{stem}_monthly.parquet")
    hold.to_parquet(f"{stem}_holdings.parquet", index=False)
    s, _ = p04.summary(monthly)
    ls = s.loc["long_short"]
    return {"months": int(p["ret_month"].nunique()), "oos_r2": r2_oos(y, f), "ic_mean": ic.mean(),
            "ic_t": ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))), "ls_mean": ls["mean"], "ls_sd": ls["sd"],
            "sharpe": ls["sharpe"], "t": ls["t"], "turnover": monthly["turnover"].mean(),
            "first_ret_month": p["ret_month"].min(), "last_ret_month": p["ret_month"].max()}


def cross_fit(model, data, args, builds, out):
    """Diagnostic refits: each validation share's training part with each other share's settings."""
    summary = pd.read_csv(out / "tables" / f"05_{model.name}_summary.csv")
    diag = out / "predictions" / model.name / "diagnostic"
    diag.mkdir(parents=True, exist_ok=True)
    (out / "models" / model.name / "diagnostic").mkdir(parents=True, exist_ok=True)
    cache, fit_log = {}, []
    for window in args.windows:
        settings = {}
        for val in args.val:
            rows = summary[(summary["window"] == window_label(window)) & np.isclose(summary["val_share"], val)]
            check(rows["settings"].nunique() == 1, f"no unique tuned settings for {setup_name(window, val)} in the summary")
            settings[val] = json.loads(rows["settings"].iloc[0])
        for val in args.val:
            for other in args.val:
                if other == val:
                    continue
                tag = f"_set{round(other * 100)}"
                path = diag / f"{setup_name(window, val)}{tag}.parquet"
                if path.exists():
                    print(f"{path.name}: exists, skipped")
                    continue
                print(f"{setup_name(window, val)}{tag}: training part of {val:.0%}, settings tuned with {other:.0%}", flush=True)
                n_log = len(fit_log)
                preds = run_setup(model, data, window, val, builds, args.test_end, out, cache, fit_log, [],
                                  fixed_settings=settings[other], tag=tag)
                for row in fit_log[n_log:]:
                    row["settings_from"] = other
                preds["settings_from"] = other
                preds.to_parquet(path, index=False)
    if fit_log:
        pd.DataFrame(fit_log).to_csv(out / "tables" / f"05_{model.name}_crossfit_fits.csv", index=False, float_format="%.6g")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(MODELS), default="enet")
    parser.add_argument("--windows", nargs="+", default=WINDOWS, choices=WINDOWS)
    parser.add_argument("--val", nargs="+", type=float, default=VAL_SHARES)
    parser.add_argument("--test-start", type=int, default=1990)
    parser.add_argument("--test-end", type=int, default=2025)
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--clean", default=str(ROOT / "data" / "clean.parquet"), help="for the portfolio distress flag")
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--cross-fit", action="store_true", help="diagnostic refits with another share's settings")
    args = parser.parse_args()
    t_start = time.time()
    model = MODELS[args.model]()
    out = Path(args.out)
    for sub in ("predictions/" + model.name, "models/" + model.name, "portfolios", "tables"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    chars = [l.strip() for l in CHARS_153_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]
    df = pd.read_parquet(args.panel, columns=["id", "eom", "ret_exc_lead1m"] + chars)
    df["eom"] = pd.to_datetime(df["eom"])
    df = df.sort_values(["eom", "id"], ignore_index=True)
    df["ret_month"] = df["eom"] + pd.offsets.MonthEnd(1)
    m_idx = month_index(df["ret_month"])
    data = {"X": df[chars].to_numpy(np.float64), "y": df["ret_exc_lead1m"].to_numpy(np.float64),
            "has_y": df["ret_exc_lead1m"].notna().to_numpy(), "m_idx": m_idx, "first_idx": int(m_idx.min()),
            "ret_year": df["ret_month"].dt.year.to_numpy(), "id": df["id"].to_numpy(), "eom": df["eom"].to_numpy(),
            "ret_month": df["ret_month"].to_numpy(), "chars": chars}
    target = df[["id", "eom", "ret_exc_lead1m"]]
    print(f"Panel: {len(df):,} rows, return months {df['ret_month'].min():%Y-%m} to {df['ret_month'].max():%Y-%m}; "
          f"model {model.name}; test years {args.test_start}-{args.test_end}")

    builds = list(range(args.test_start - 1, args.test_end))
    if args.cross_fit:
        cross_fit(model, data, args, builds, out)
        print(f"Cross-fits done in {time.time() - t_start:,.0f}s")
        return
    p04 = load_src("04_portfolios")
    pf_panel = p04.load_panel(args.panel, args.clean)
    cache, fit_log, tuning_log, summary = {}, [], [], []
    once_preds = {}
    for window in args.windows:
        for val in args.val:
            name = setup_name(window, val)
            path = out / "predictions" / model.name / f"{name}.parquet"
            if path.exists():
                print(f"{name}: predictions exist, evaluating only")
                preds = pd.read_parquet(path)
                preds["eom"] = pd.to_datetime(preds["eom"])
            else:
                print(f"{name}: fitting {len(builds)} build years", flush=True)
                preds = run_setup(model, data, window, val, builds, args.test_end, out, cache, fit_log, tuning_log)
                preds.to_parquet(path, index=False)
            first = preds[preds["build_year"] == builds[0]].iloc[0]
            constant = preds.groupby(["freq", "build_year"])["pred"].nunique().eq(1)
            for freq in ("once", "yearly"):
                row = {"model": model.name, "window": window_label(window), "val_share": val, "freq": freq,
                       "n_fits": 1 if freq == "once" else preds["build_year"].nunique(), "settings": first["settings"],
                       "constant_fits": int(constant.loc[freq].sum())}
                row.update(evaluate(p04, pf_panel, preds, target, model.name, name, freq, out))
                summary.append(row)
            once = preds[preds["freq"] == "once"].sort_values(["eom", "id"])
            once_preds[(window_label(window), val)] = once["pred"].to_numpy()

    summary = pd.DataFrame(summary)
    summary["note"] = np.where(summary["constant_fits"] > 0,
                               summary["constant_fits"].astype(str) + " fit(s) predict a constant (all coefficients 0); ", "")
    for val in args.val:
        a, b = once_preds.get(("15y", val)), once_preds.get(("expanding", val))
        if a is not None and b is not None and np.array_equal(a, b):
            summary.loc[(summary["freq"] == "once") & (summary["val_share"] == val)
                        & summary["window"].isin(["15y", "expanding"]), "note"] += "15y = expanding (same Dec 1989 fit)"
    tables = out / "tables"
    summary.to_csv(tables / f"05_{model.name}_summary.csv", index=False, float_format="%.6g")
    if fit_log:
        pd.DataFrame(fit_log).to_csv(tables / f"05_{model.name}_fits.csv", index=False, float_format="%.6g")
        pd.DataFrame(tuning_log).to_csv(tables / f"05_{model.name}_tuning.csv", index=False, float_format="%.6g")

    s0 = summary.iloc[0]
    print(f"\nSummary, return months {s0['first_ret_month']:%Y-%m} to {s0['last_ret_month']:%Y-%m}; long-short decile "
          "portfolio, capped value weights, missing returns = 0")
    print(f"  {'window':<10}{'freq':<7}{'val':>4} {'l1':>4}{'alpha':>9}{'R2 oos':>8}{'IC':>7}{'IC t':>6}"
          f"{'LS mean':>8}{'Sharpe':>7}{'t':>6}{'turnover':>9}  note")
    for r in summary.itertuples():
        st = json.loads(r.settings)
        print(f"  {r.window:<10}{r.freq:<7}{r.val_share:>4.2f} {st.get('l1_ratio', ''):>4}{st.get('alpha', 0):>9.2e}"
              f"{r.oos_r2:>8.2%}{r.ic_mean:>7.3f}{r.ic_t:>6.1f}{r.ls_mean:>8.2%}{r.sharpe:>7.2f}{r.t:>6.2f}{r.turnover:>9.0%}  {r.note}")
    print(f"\nSaved {tables}/05_{model.name}_summary.csv; done in {time.time() - t_start:,.0f}s")


if __name__ == "__main__":
    main()
