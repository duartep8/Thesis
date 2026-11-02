"""Step 6: evaluate any model's saved predictions and portfolios (nothing is refit).

Usage:
    python src/06_evaluate.py --model enet [--download-ff] [--self-test]
                              [--panel data/panel.parquet] [--clean data/clean.parquet]
                              [--results results] [--ff-dir data/ff]

Reads what 05_train.py saved for every setup (window x frequency x validation share):
results/portfolios/<model>_<setup>_<freq>_{monthly,holdings}.parquet, the predictions in
results/predictions/<model>/ and results/tables/05_<model>_{summary,fits}.csv. Every
setup is reported equally; nothing is ranked and no best setup is picked.

Fama-French: the monthly 5 factors (2x3) and momentum from Kenneth French's data library,
in data/ff/ (downloaded if missing, or with --download-ff). Only the monthly rows are read
(the annual block below them is skipped), percent -> decimals, every test month must be
present. The file versions (CRSP vintage, SHA-256) are written to data/ff/README.txt.
The long-short formed at month-end t earns its return in t+1 and is matched to the
factors of that month (ret_month).

Trading costs: cost_t = sum |trade| x bidaskhl_21d / 2 over both sides, with trade = the
new weight minus the drifted previous weight (04_portfolios.py), charged against the
return earned in t+1; the first month pays for the initial purchase. A sold-out stock
missing from the panel at t takes its unranked bidaskhl_21d from clean.parquet at t, else
its own spread at t-1. bidaskhl_21d (Corwin-Schultz high-low) overstates the spreads of
large stocks (Chen & Velikov 2023: 25-50 bp too high after 2003), so every net result is
an upper bound on costs. Costs are built from COST_COMPONENTS; price impact can be added
there. Short-selling fees are not included. Break-even: the multiple k of these costs at
which the net Sharpe ratio is zero (= gross mean / mean cost) and, for yearly vs once,
the k at which their net Sharpe ratios are equal (searched in [0, K_MAX]).

Sharpe ratio tests: Ledoit & Wolf (2008), difference of monthly Sharpe ratios. Main
p-value: studentized circular block bootstrap (block 5, 5,000 draws, seed 42); HAC
p-value (prewhitened QS kernel, Andrews automatic bandwidth) alongside; Holm-adjusted
bootstrap p-values within each group (gross and net separately). Groups: yearly vs once
for each window and validation share; the validation shares against each other for each
window and frequency; every setup against the reference (expanding, yearly, 30%), fixed
in advance as the main design. Identical series (15-year and expanding trained once) are
marked and not tested.

Alphas: long-short returns (gross and net) on Mkt-RF, SMB, HML, RMW, CMA and Mom, OLS
with Newey-West standard errors (6 lags).

Missing returns: our JKP file has no delisting returns, so every result is labelled
provisional, pending delisting returns (04_portfolios.RESULT_STATUS). Next to the baseline
(a held stock with a missing return counts as 0), two sensitivities rebuild each setup's
portfolio from its saved predictions with held missing returns at -30% for all stocks, or
for distressed stocks only (04_portfolios.py); the weights do not change. For each:
gross and net Sharpe ratio, alphas and the yearly-vs-once tests.

Validation-share diagnostics: chosen settings and model size per setup; for each pair of
shares within a window and frequency, the average monthly Pearson and Spearman correlation
of predictions, the overlap of the long and short sides and the correlation of long-short
returns. If 05_train.py --cross-fit was run, a's training part with b's settings
separates training data from tuning: correlations at equal settings or equal data, and the
Sharpe difference split both ways (tuning + data).

Outputs: results/tables/06_<model>_{summary,sharpe_tests,alphas,valshare_diag,crossfit,
missing_sensitivity}.csv (aggregates only); monthly net returns in
results/portfolios/<model>_<setup>_<freq>[_m30_all|_m30_distressed]_net.parquet
and cross-fit portfolios in results/portfolios/diagnostic/ (not committed).
"""

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FF_FILES = {"F-F_Research_Data_5_Factors_2x3": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
            "F-F_Momentum_Factor": ["Mom"]}
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
NW_LAGS = 6
BLOCK, N_BOOT, SEED = 5, 5000, 42
K_MAX = 5.0
WINDOW_ORDER = ["5y", "10y", "15y", "expanding"]
FREQS = ["once", "yearly"]
VAL_PAIRS = [(0.25, 0.30), (0.30, 0.35), (0.25, 0.35)]
REFERENCE = ("expanding", "yearly", 0.30)
SIZE_COLUMN = {"enet": "n_nonzero"}  # model size logged by 05_train.py in 05_<model>_fits.csv
COST_LABEL = "net = after half-spread costs from bidaskhl_21d, an upper bound on costs"
# Held missing returns: (label, value, scope) for 04_portfolios.portfolio; the first is the baseline
MISSING_VERSIONS = [("baseline", 0.0, "all"), ("m30_all", -0.30, "all"), ("m30_distressed", -0.30, "distressed")]
MISSING_LABELS = {"baseline": "0 (baseline)", "m30_all": "-30%, all", "m30_distressed": "-30%, distressed only"}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def load_src(name):
    """Import a numbered script from src/ (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stem(model, window, freq, val):
    return f"{model}_{window}_val{round(val * 100)}_{freq}"


# ---------------------------------------------------------------- Fama-French factors

def download_ff(ff_dir):
    ff_dir.mkdir(parents=True, exist_ok=True)
    for name in FF_FILES:
        zpath = ff_dir / f"{name}_CSV.zip"
        urllib.request.urlretrieve(FF_URL.format(name), zpath)
        with zipfile.ZipFile(zpath) as z:
            check(f"{name}.csv" in z.namelist(), f"{zpath.name} lacks {name}.csv: {z.namelist()}")
            (ff_dir / f"{name}.csv").write_bytes(z.read(f"{name}.csv"))  # only this member, by name
        print(f"Downloaded {zpath.name}")


def ff_provenance(ff_dir):
    """Download date, CRSP vintage and SHA-256 of each file, written to data/ff/README.txt."""
    lines = ["Fama-French factors from Kenneth French's data library (" + FF_URL.format("<name>") + ")."]
    for name in FF_FILES:
        csv, zpath = ff_dir / f"{name}.csv", ff_dir / f"{name}_CSV.zip"
        text = csv.read_text(encoding="latin-1").splitlines()
        vintage = re.search(r"(\d{6}) CRSP database", text[0])
        lines += [f"{name}.csv: downloaded {date.fromtimestamp(zpath.stat().st_mtime)}; "
                  f"built from the {vintage.group(1) if vintage else '?'} CRSP database",
                  f"  {[l for l in text if 'Copyright' in l][-1].strip()}",
                  f"  sha256 zip {hashlib.sha256(zpath.read_bytes()).hexdigest()}",
                  f"  sha256 csv {hashlib.sha256(csv.read_bytes()).hexdigest()}"]
    (ff_dir / "README.txt").write_text("\n".join(lines) + "\n")
    return lines


def read_ff(path, cols):
    """Monthly rows of a French CSV (YYYYMM, under the first header), in decimals."""
    lines = path.read_text(encoding="latin-1").splitlines()
    heads = [i for i, l in enumerate(lines) if [c.strip() for c in l.split(",")] == [""] + cols]
    check(heads, f"{path.name}: header ,{','.join(cols)} not found")
    rows = []
    for l in lines[heads[0] + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if not re.fullmatch(r"\d{6}", parts[0]):
            break
        check(len(parts) == len(cols) + 1, f"{path.name}: bad row {l!r}")
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["yyyymm"] + cols)
    vals = df[cols].astype(float)
    check(not vals.isin([-99.99, -999.0]).any().any(), f"{path.name}: missing-value code in the monthly data")
    vals.index = pd.to_datetime(df["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    vals.index.name = "ret_month"
    check(vals.index.is_unique and vals.index.is_monotonic_increasing, f"{path.name}: months not unique and sorted")
    expected = pd.date_range(vals.index[0], vals.index[-1], freq="ME")
    check(len(expected) == len(vals), f"{path.name}: gaps in the monthly data")
    return vals / 100.0


def load_factors(ff_dir, months):
    ff = pd.concat([read_ff(ff_dir / f"{n}.csv", c) for n, c in FF_FILES.items()], axis=1, join="inner")
    missing = months.difference(ff.index)
    check(len(missing) == 0, f"factor months missing: {list(missing[:5])}")
    return ff


# ---------------------------------------------------------------- trading costs

def fill_spread(hold, panel_spread, clean_spread):
    """bidaskhl_21d for every holdings row: panel at t, else clean at t, else own panel value at t-1."""
    s = hold["bidaskhl_21d_raw"].astype(float).to_numpy().copy()
    counts = {"panel": int(np.isfinite(s).sum())}
    miss = ~np.isfinite(s)
    s[miss] = hold.loc[miss, ["id", "eom"]].merge(clean_spread, on=["id", "eom"], how="left")["bidaskhl_21d"].to_numpy()
    counts["clean_t"] = int(miss.sum() - (~np.isfinite(s)).sum())
    miss = ~np.isfinite(s)
    prev = hold.loc[miss, ["id", "eom"]].assign(eom=lambda d: d["eom"] - pd.offsets.MonthEnd(1))
    s[miss] = prev.merge(panel_spread, on=["id", "eom"], how="left")["bidaskhl_21d_raw"].to_numpy()
    counts["own_t-1"] = int(miss.sum() - (~np.isfinite(s)).sum())
    check(np.isfinite(s).all(), f"{(~np.isfinite(s)).sum()} holdings rows still lack a spread")
    check((s >= 0).all(), "negative spread")
    return s, counts


def spread_cost(hold, spread):
    """Half-spread cost per month and side. Trades are split by side, so a stock switching
    sides pays for selling one position and buying the other."""
    w, wp = hold["weight"], hold["weight_prev"]
    trades = pd.DataFrame({"long": (w.clip(lower=0) - wp.clip(lower=0)).abs(),
                           "short": (w.clip(upper=0) - wp.clip(upper=0)).abs()})
    return trades.mul(spread / 2, axis=0).groupby(hold["eom"]).sum(), trades.groupby(hold["eom"]).sum()


# Each component returns (cost per month and side, traded weight per month and side); add price impact here.
COST_COMPONENTS = {"spread": spread_cost}


def net_returns(monthly, hold, spread):
    """Monthly gross and net long-short returns indexed by ret_month, with costs by component and side."""
    out = pd.DataFrame({"gross": monthly["long_short"]})
    for name, fn in COST_COMPONENTS.items():
        cost, traded = fn(hold, spread)
        cost = cost.reindex(monthly.index)
        check(cost.notna().all().all(), f"{name}: a month without trades")
        out[f"cost_{name}_long"], out[f"cost_{name}_short"] = cost["long"], cost["short"]
        out[f"cost_{name}"] = cost["long"] + cost["short"]
        traded = traded.reindex(monthly.index)
        has = monthly["turnover"].notna()
        check(np.allclose(traded.loc[has, "long"], monthly.loc[has, "turnover_long"], atol=1e-9)
              and np.allclose(traded.loc[has, "short"], monthly.loc[has, "turnover_short"], atol=1e-9),
              "traded weight differs from 04's turnover")
        check(np.allclose(traded.loc[~has].to_numpy(), 1.0), "first month: initial purchase is not 100% per side")
    out["cost_total"] = out[[f"cost_{n}" for n in COST_COMPONENTS]].sum(axis=1)
    out["net"] = out["gross"] - out["cost_total"]
    out.index = out.index + pd.offsets.MonthEnd(1)
    out.index.name = "ret_month"
    return out


def sharpe(x):
    x = np.asarray(x, float)
    return x.mean() / x.std(ddof=1) * np.sqrt(12)


def breakeven_zero(gross, cost):
    """Multiple of costs at which the net mean (and Sharpe ratio) is zero; NaN if the gross mean is not positive."""
    return gross.mean() / cost.mean() if gross.mean() > 0 else np.nan


def breakeven_equal(g1, c1, g2, c2):
    """Smallest k in [0, K_MAX] at which SR(g1 - k c1) = SR(g2 - k c2); NaN if none."""
    if np.array_equal(g1, g2) and np.array_equal(c1, c2):
        return np.nan
    f = lambda k: sharpe(g1 - k * c1) - sharpe(g2 - k * c2)
    grid = np.linspace(0, K_MAX, 501)
    vals = np.array([f(k) for k in grid])
    if vals[0] == 0:
        return 0.0
    idx = np.nonzero(np.sign(vals[1:]) != np.sign(vals[:-1]))[0]
    return brentq(f, grid[idx[0]], grid[idx[0] + 1], xtol=1e-10) if len(idx) else np.nan


# ---------------------------------------------------------------- Ledoit & Wolf (2008)

def _lw_parts(r1, r2):
    mu1, mu2, g1, g2 = r1.mean(), r2.mean(), (r1 ** 2).mean(), (r2 ** 2).mean()
    v1, v2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    delta = mu1 / np.sqrt(v1) - mu2 / np.sqrt(v2)
    grad = np.array([g1 / v1 ** 1.5, -g2 / v2 ** 1.5, -mu1 / (2 * v1 ** 1.5), mu2 / (2 * v2 ** 1.5)])
    y = np.column_stack([r1 - mu1, r2 - mu2, r1 ** 2 - g1, r2 ** 2 - g2])
    return delta, grad, y


def qs_kernel(x):
    x = np.asarray(x, float)
    z = 6 * np.pi * x / 5
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 25 / (12 * np.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))
    return np.where(x == 0, 1.0, k)


def hac_psi(y):
    """Long-run covariance of y: VAR(1) prewhitening, QS kernel, Andrews (1991) AR(1) bandwidth,
    recolored (Andrews & Monahan 1992), with the T/(T-4) small-sample factor (Ledoit & Wolf 2008)."""
    Y, Z = y[1:], y[:-1]
    A = np.linalg.lstsq(Z, Y, rcond=None)[0].T
    u, s, vt = np.linalg.svd(A)
    if s.max() > 0.97:
        A = u @ np.diag(np.minimum(s, 0.97)) @ vt
    v = Y - Z @ A.T
    T, k = v.shape
    rho = (v[1:] * v[:-1]).sum(0) / (v[:-1] ** 2).sum(0)
    sig2 = ((v[1:] - rho * v[:-1]) ** 2).mean(0)
    a2 = (4 * rho ** 2 * sig2 ** 2 / (1 - rho) ** 8).sum() / (sig2 ** 2 / (1 - rho) ** 4).sum()
    bw = 1.3221 * (a2 * T) ** 0.2
    psi = v.T @ v / T
    weights = qs_kernel(np.arange(1, T) / bw)
    for j, wj in enumerate(weights, start=1):
        g = v[j:].T @ v[:-j] / T
        psi += wj * (g + g.T)
    psi *= T / (T - k)
    D = np.linalg.inv(np.eye(k) - A)
    return D @ psi @ D.T


def lw_boot_p(r1, r2, delta, se, b=BLOCK, m=N_BOOT, seed=SEED):
    """Studentized circular block bootstrap p-value (Ledoit & Wolf 2008, Section 3.2)."""
    T = len(r1)
    l = T // b
    rng = np.random.default_rng(seed)
    idx = ((rng.integers(0, T, size=(m, l))[:, :, None] + np.arange(b)) % T).reshape(m, l * b)
    x1, x2 = r1[idx], r2[idx]
    mu1, mu2, g1, g2 = x1.mean(1), x2.mean(1), (x1 ** 2).mean(1), (x2 ** 2).mean(1)
    v1, v2 = g1 - mu1 ** 2, g2 - mu2 ** 2
    d = mu1 / np.sqrt(v1) - mu2 / np.sqrt(v2)
    grad = np.stack([g1 / v1 ** 1.5, -g2 / v2 ** 1.5, -mu1 / (2 * v1 ** 1.5), mu2 / (2 * v2 ** 1.5)], 1)
    y = np.stack([x1 - mu1[:, None], x2 - mu2[:, None], x1 ** 2 - g1[:, None], x2 ** 2 - g2[:, None]], 2)
    zeta = y.reshape(m, l, b, 4).sum(2) / np.sqrt(b)
    psi = np.einsum("mli,mlj->mij", zeta, zeta) / l
    se_b = np.sqrt(np.einsum("mi,mij,mj->m", grad, psi, grad) / (l * b))
    return (np.sum(np.abs(d - delta) / se_b >= abs(delta) / se) + 1) / (m + 1)


def lw_test(r1, r2, boot=True):
    """Ledoit-Wolf test of equal Sharpe ratios; delta is the monthly difference."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    if np.array_equal(r1, r2):
        return {"identical": True, "delta_m": 0.0, "se_m": 0.0, "p_hac": 1.0, "p_boot": 1.0}
    delta, grad, y = _lw_parts(r1, r2)
    se = np.sqrt(grad @ hac_psi(y) @ grad / len(r1))
    return {"identical": False, "delta_m": delta, "se_m": se, "p_hac": 2 * norm.sf(abs(delta) / se),
            "p_boot": lw_boot_p(r1, r2, delta, se) if boot else np.nan}


def self_test():
    """Checks of the Sharpe test and the break-even solver on simulated data."""
    rng = np.random.default_rng(SEED)
    print("Self-test")
    x = rng.normal(0.01, 0.05, 432)
    r = lw_test(x, x.copy())
    check(r["identical"] and r["p_boot"] == 1.0, "identical series")
    print("  identical series: delta 0, p = 1  ok")

    T, s1, s2, rho = 200_000, 0.20, 0.10, 0.5
    e = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], T)
    r1, r2 = s1 + e[:, 0], s2 + e[:, 1]
    se_lw = lw_test(r1, r2, boot=False)["se_m"]
    se_jk = np.sqrt((2 - 2 * rho + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / T)
    print(f"  iid normal, T = {T:,}: HAC se / Jobson-Korkie-Memmel se = {se_lw / se_jk:.3f}")
    check(abs(se_lw / se_jk - 1) < 0.05, "HAC se far from the closed form")

    def sim(n):  # equal true Sharpe ratios: mean 1%, sd 5% a month, correlation 0.7
        e = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], (n, 432))
        return 0.01 + 0.05 * e[..., 0], 0.01 + 0.05 * e[..., 1]
    a, b = sim(1000)
    rej = np.mean([lw_test(a[i], b[i], boot=False)["p_hac"] < 0.05 for i in range(1000)])
    a, b = sim(200)
    rej_b = np.mean([lw_boot_p_only(a[i], b[i]) < 0.05 for i in range(200)])
    print(f"  equal Sharpe ratios, T = 432: HAC rejects {rej:.1%} of 1,000; bootstrap (999 draws) {rej_b:.1%} of 200 at 5%")
    check(0.03 <= rej <= 0.075 and 0.01 <= rej_b <= 0.10, "test size far from 5%")

    g1, c1 = rng.normal(0.010, 0.04, 432), np.full(432, 0.004)
    g2, c2 = g1 * 0.8 + rng.normal(0, 0.01, 432), np.full(432, 0.002)
    k = breakeven_equal(g1, c1, g2, c2)
    check(abs(sharpe(g1 - k * c1) - sharpe(g2 - k * c2)) < 1e-8, "break-even solver")
    check(abs(sharpe(g1 - breakeven_zero(g1, c1) * c1)) < 1e-12, "zero break-even")
    print(f"  break-even solvers: ok (equal Sharpe at k = {k:.3f})")


def lw_boot_p_only(r1, r2):
    delta, grad, y = _lw_parts(r1, r2)
    se = np.sqrt(grad @ hac_psi(y) @ grad / len(r1))
    return lw_boot_p(r1, r2, delta, se, m=999)


# ---------------------------------------------------------------- alphas

def alpha_reg(y, ff):
    X = sm.add_constant(ff.loc[y.index, FACTORS])
    res = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    out = {"months": int(res.nobs), "alpha_m": res.params["const"], "alpha_ann": 12 * res.params["const"],
           "alpha_t": res.tvalues["const"], "r2": res.rsquared}
    for f in FACTORS:
        out[f"b_{f}"], out[f"t_{f}"] = res.params[f], res.tvalues[f]
    return out


# ---------------------------------------------------------------- validation-share diagnostics

def monthly_corr(df, a, b):
    """Average over months of the cross-sectional correlation of columns a and b (NaN months skipped)."""
    g = df.groupby("eom")
    da, db = df[a] - g[a].transform("mean"), df[b] - g[b].transform("mean")
    num = (da * db).groupby(df["eom"]).sum()
    den = np.sqrt((da ** 2).groupby(df["eom"]).sum() * (db ** 2).groupby(df["eom"]).sum())
    c = (num / den).replace([np.inf, -np.inf], np.nan)
    return c.mean(), int(c.isna().sum())


def pred_corrs(pa, pb):
    """Pearson and Spearman correlation of two prediction sets on the same (id, eom)."""
    d = pa.merge(pb, on=["id", "eom"], suffixes=("_a", "_b"))
    check(len(d) == len(pa) == len(pb), "prediction sets do not cover the same rows")
    d["rank_a"] = d.groupby("eom")["pred_a"].rank()
    d["rank_b"] = d.groupby("eom")["pred_b"].rank()
    pearson, n_nan = monthly_corr(d, "pred_a", "pred_b")
    spearman, _ = monthly_corr(d, "rank_a", "rank_b")
    return pearson, spearman, n_nan


def side_overlap(ha, hb, positive):
    """Average share of a side's stocks that are on the same side in the other portfolio."""
    sel = lambda h: h.loc[h["weight"] > 0 if positive else h["weight"] < 0, ["id", "eom"]]
    a, b = sel(ha), sel(hb)
    common = a.merge(b, on=["id", "eom"]).groupby("eom").size()
    return (common.reindex(a["eom"].unique(), fill_value=0) / a.groupby("eom").size()).mean()


def predictions(path, freq):
    p = pd.read_parquet(path, columns=["id", "eom", "pred", "freq", "age"])
    p = p[(p["freq"] == freq) & ((p["age"] == 0) if freq == "yearly" else True)]
    p["eom"] = pd.to_datetime(p["eom"])
    return p[["id", "eom", "pred"]].sort_values(["eom", "id"], ignore_index=True)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="enet")
    parser.add_argument("--download-ff", action="store_true", help="download the French files even if present")
    parser.add_argument("--self-test", action="store_true", help="check the Sharpe test on simulated data first")
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--clean", default=str(ROOT / "data" / "clean.parquet"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--ff-dir", default=str(ROOT / "data" / "ff"))
    args = parser.parse_args()
    t0 = time.time()
    pd.set_option("display.width", 250)
    if args.self_test:
        self_test()

    res, model = Path(args.results), args.model
    tables, pf_dir = res / "tables", res / "portfolios"
    (pf_dir / "diagnostic").mkdir(parents=True, exist_ok=True)

    # Fama-French factors
    ff_dir = Path(args.ff_dir)
    if args.download_ff or not all((ff_dir / f"{n}.csv").exists() for n in FF_FILES):
        download_ff(ff_dir)
    print("\nFama-French factors")
    for line in ff_provenance(ff_dir):
        print("  " + line)

    # Setups and gross results from 05
    s05 = pd.read_csv(tables / f"05_{model}_summary.csv")
    s05["val_share"] = s05["val_share"].round(2)
    setups = [(w, f, v) for w in WINDOW_ORDER for f in FREQS for v in sorted(s05["val_share"].unique())
              if ((s05["window"] == w) & (s05["freq"] == f) & (s05["val_share"] == v)).any()]
    print(f"\nModel {model}: {len(setups)} setups")

    p04 = load_src("04_portfolios")
    pf_panel = p04.load_panel(args.panel, args.clean)
    panel_spread = pf_panel[["id", "eom", "bidaskhl_21d_raw"]]
    clean_spread = pd.read_parquet(args.clean, columns=["id", "eom", "bidaskhl_21d"])
    clean_spread["eom"] = pd.to_datetime(clean_spread["eom"])

    # Costs and net returns
    series, holds, fill_counts, missing = {}, {}, {}, {}
    for s in setups:
        name = stem(model, *s)
        monthly = pd.read_parquet(pf_dir / f"{name}_monthly.parquet")
        hold = pd.read_parquet(pf_dir / f"{name}_holdings.parquet")
        hold["eom"] = pd.to_datetime(hold["eom"])
        spread, fill_counts[s] = fill_spread(hold, panel_spread, clean_spread)
        net = net_returns(monthly, hold, spread)
        net.to_parquet(pf_dir / f"{name}_net.parquet")
        series[s], holds[s] = net, hold
        missing[("baseline", s)] = (int(monthly["n_missing_ret"].sum()), int(monthly["n_missing_filled"].sum()))

    # Missing-return sensitivities: the same portfolios rebuilt with held missing returns at -30%
    sens = {"baseline": series}
    for version, value, scope in MISSING_VERSIONS[1:]:
        sens[version] = {}
    for s in setups:
        sig = predictions(res / "predictions" / model / f"{s[0]}_val{round(s[2] * 100)}.parquet", s[1])
        for version, value, scope in MISSING_VERSIONS[1:]:
            monthly, hold = p04.portfolio(sig, pf_panel, missing_return=value, missing_scope=scope)
            base = holds[s]
            check(len(hold) == len(base) and (hold["id"].to_numpy() == base["id"].to_numpy()).all()
                  and (hold["eom"].to_numpy().astype("datetime64[ns]") == base["eom"].to_numpy().astype("datetime64[ns]")).all()
                  and np.allclose(hold["weight"].to_numpy(), base["weight"].to_numpy(), rtol=0, atol=1e-12),
                  f"{s} {version}: weights differ from the baseline portfolio")
            spread, _ = fill_spread(hold, panel_spread, clean_spread)
            net = net_returns(monthly, hold, spread)
            net.to_parquet(pf_dir / f"{stem(model, *s)}_{version}_net.parquet")
            sens[version][s] = net
            missing[(version, s)] = (int(monthly["n_missing_ret"].sum()), int(monthly["n_missing_filled"].sum()))
    print(f"Missing-return sensitivities rebuilt for {len(setups)} setups x {len(MISSING_VERSIONS) - 1} versions "
          "(weights identical to the baseline)")
    months = series[setups[0]].index
    check(all(series[s].index.equals(months) for s in setups), "setups cover different months")
    c = pd.DataFrame(fill_counts).T.sum()
    print(f"Return months {months[0]:%Y-%m} to {months[-1]:%Y-%m} ({len(months)}). Spreads of holdings rows over all "
          f"setups: {c['panel']:,} from the panel at t, sold-out stocks {c['clean_t']:,} from clean at t and "
          f"{c['own_t-1']:,} from their own value at t-1")
    ff = load_factors(ff_dir, months)
    print(f"Factors: {ff.index[0]:%Y-%m} to {ff.index[-1]:%Y-%m}; all {len(months)} test months present; mean per month, "
          + ", ".join(f"{f} {ff.loc[months, f].mean():.2%}" for f in FACTORS))

    # Summary rows: Sharpe ratios, costs, break-even, alphas
    rows, alpha_rows = [], []
    for s in setups:
        w, f, v = s
        d = series[s]
        ref = s05[(s05["window"] == w) & (s05["freq"] == f) & (s05["val_share"] == v)].iloc[0]
        check(abs(sharpe(d["gross"]) - ref["sharpe"]) <= 1e-5 * max(1, abs(ref["sharpe"])),
              f"{s}: gross Sharpe differs from 05's summary")
        row = {"model": model, "window": w, "freq": f, "val_share": v, "months": len(d),
               "gross_mean": d["gross"].mean(), "gross_sharpe": sharpe(d["gross"]),
               "net_mean": d["net"].mean(), "net_sharpe": sharpe(d["net"]),
               "cost_mean": d["cost_total"].mean(), "cost_long": d["cost_spread_long"].mean(),
               "cost_short": d["cost_spread_short"].mean(), "turnover": ref["turnover"],
               "breakeven_k": breakeven_zero(d["gross"], d["cost_total"])}
        for kind in ("gross", "net"):
            a = alpha_reg(d[kind], ff)
            alpha_rows.append({"model": model, "window": w, "freq": f, "val_share": v, "returns": kind, **a})
            row.update({f"alpha_{kind}_m": a["alpha_m"], f"alpha_{kind}_ann": a["alpha_ann"], f"alpha_{kind}_t": a["alpha_t"]})
        rows.append(row)
    summary = pd.DataFrame(rows).set_index(["window", "freq", "val_share"])
    alphas = pd.DataFrame(alpha_rows)

    # Sharpe ratio tests
    tests = []

    def add_test(group, a, b, version="baseline"):
        for kind in ("gross", "net"):
            ra, rb = sens[version][a][kind].to_numpy(), sens[version][b][kind].to_numpy()
            t = lw_test(ra, rb)
            tests.append({"group": group, "missing_version": version, "returns": kind,
                          "a_window": a[0], "a_freq": a[1], "a_val": a[2],
                          "b_window": b[0], "b_freq": b[1], "b_val": b[2], "sharpe_a": sharpe(ra), "sharpe_b": sharpe(rb),
                          "diff": sharpe(ra) - sharpe(rb), "delta_lw_ann": t["delta_m"] * np.sqrt(12),
                          "se_lw_ann": t["se_m"] * np.sqrt(12), "p_boot": t["p_boot"], "p_hac": t["p_hac"],
                          "identical": t["identical"]})

    have = set(setups)
    for w in WINDOW_ORDER:
        for v in sorted(s05["val_share"].unique()):
            if (w, "yearly", v) in have and (w, "once", v) in have:
                for version, _, _ in MISSING_VERSIONS:
                    add_test("yearly_vs_once", (w, "yearly", v), (w, "once", v), version)
    for w in WINDOW_ORDER:
        for f in FREQS:
            for va, vb in VAL_PAIRS:
                if (w, f, va) in have and (w, f, vb) in have:
                    add_test("val_shares", (w, f, va), (w, f, vb))
    if REFERENCE in have:
        for s in setups:
            if s != REFERENCE:
                add_test("vs_reference", s, REFERENCE)
    tests = pd.DataFrame(tests)
    tests["p_holm"] = np.nan
    for _, idx in tests[~tests["identical"]].groupby(["group", "returns", "missing_version"]).groups.items():
        tests.loc[idx, "p_holm"] = multipletests(tests.loc[idx, "p_boot"], method="holm")[1]
    tests["note"] = np.where(tests["identical"], "identical return series; not tested", "")

    # Break-even multiple at which yearly and once have equal net Sharpe ratios; p-values into the summary
    summary["breakeven_k_equal"] = np.nan
    for w in WINDOW_ORDER:
        for v in sorted(s05["val_share"].unique()):
            y, o = (w, "yearly", v), (w, "once", v)
            if y in have and o in have:
                k = breakeven_equal(series[y]["gross"].to_numpy(), series[y]["cost_total"].to_numpy(),
                                    series[o]["gross"].to_numpy(), series[o]["cost_total"].to_numpy())
                summary.loc[[y, o], "breakeven_k_equal"] = k
    for kind in ("gross", "net"):
        t = tests[(tests["returns"] == kind) & (tests["missing_version"] == "baseline")]
        yo = t[t["group"] == "yearly_vs_once"]
        for r in yo.itertuples():
            summary.loc[[(r.a_window, r.a_freq, r.a_val), (r.b_window, r.b_freq, r.b_val)], f"p_yearly_vs_once_{kind}"] = r.p_boot
        vr = t[t["group"] == "vs_reference"]
        for r in vr.itertuples():
            summary.loc[(r.a_window, r.a_freq, r.a_val), f"p_vs_ref_{kind}"] = r.p_boot
    identical = [s for s in setups if s[1] == "once" and s[0] in ("15y", "expanding")]
    summary["note"] = ""
    for s in identical:
        other = ("expanding" if s[0] == "15y" else "15y", s[1], s[2])
        if other in have and series[s]["gross"].equals(series[other]["gross"]):
            summary.loc[s, "note"] = "15y = expanding (same Dec 1989 fit)"
    if REFERENCE in have:
        summary.loc[REFERENCE, "note"] = (summary.loc[REFERENCE, "note"] + " reference").strip()
    summary["cost_measure"] = "bidaskhl_21d half-spread: upper bound on costs"
    summary = summary.reset_index()

    # Sensitivity table: every setup under each missing-return version
    sens_rows = []
    for version, value, scope in MISSING_VERSIONS:
        yo = tests[(tests["group"] == "yearly_vs_once") & (tests["missing_version"] == version)]
        for s in setups:
            d = sens[version][s]
            r = {"window": s[0], "freq": s[1], "val_share": s[2], "missing_version": version,
                 "missing_return": MISSING_LABELS[version], "held_missing": missing[(version, s)][0],
                 "filled_with_m30": missing[(version, s)][1], "gross_mean": d["gross"].mean(),
                 "gross_sharpe": sharpe(d["gross"]), "net_mean": d["net"].mean(), "net_sharpe": sharpe(d["net"]),
                 "cost_mean": d["cost_total"].mean()}
            for kind in ("gross", "net"):
                a = alpha_reg(d[kind], ff)
                r.update({f"alpha_{kind}_m": a["alpha_m"], f"alpha_{kind}_t": a["alpha_t"]})
                pair = yo[(yo["returns"] == kind) & (yo["a_window"] == s[0]) & (yo["a_val"] == s[2])]
                if len(pair):
                    r[f"yearly_minus_once_{kind}"] = pair["diff"].iloc[0]
                    r[f"p_yearly_vs_once_{kind}"] = pair["p_boot"].iloc[0]
            sens_rows.append(r)
    sensitivity = pd.DataFrame(sens_rows)

    # Validation-share diagnostics
    fits = pd.read_csv(tables / f"05_{model}_fits.csv")
    fits["val_share"] = fits["val_share"].round(2)
    size_col = SIZE_COLUMN.get(model)
    diag_rows = []
    for w in WINDOW_ORDER:
        for f in FREQS:
            for v in sorted(s05["val_share"].unique()):
                if (w, f, v) not in have:
                    continue
                st = json.loads(s05[(s05["window"] == w) & (s05["val_share"] == v)]["settings"].iloc[0])
                r = {"kind": "setup", "window": w, "freq": f, "val_a": v, **{f"setting_{k}": x for k, x in st.items()}}
                fw = fits[(fits["window"] == w) & (fits["val_share"] == v)].sort_values("build_year")
                if size_col and len(fw):
                    sz = fw[size_col].iloc[:1] if f == "once" else fw[size_col]
                    r.update({f"{size_col}_median": sz.median(), f"{size_col}_min": sz.min(), f"{size_col}_max": sz.max()})
                    r["train_months_first"] = fw["train_months"].iloc[0]
                diag_rows.append(r)
            for va, vb in VAL_PAIRS:
                a, b = (w, f, va), (w, f, vb)
                if a not in have or b not in have:
                    continue
                pa = predictions(res / "predictions" / model / f"{w}_val{round(va * 100)}.parquet", f)
                pb = predictions(res / "predictions" / model / f"{w}_val{round(vb * 100)}.parquet", f)
                pear, spear, n_nan = pred_corrs(pa, pb)
                diag_rows.append({"kind": "pair", "window": w, "freq": f, "val_a": va, "val_b": vb,
                                  "pred_pearson": pear, "pred_spearman": spear, "months_constant": n_nan,
                                  "long_overlap": side_overlap(holds[a], holds[b], True),
                                  "short_overlap": side_overlap(holds[a], holds[b], False),
                                  "ls_return_corr": series[a]["gross"].corr(series[b]["gross"])})
    diag = pd.DataFrame(diag_rows)

    # Cross-fit decomposition (if 05_train.py --cross-fit was run)
    xdir = res / "predictions" / model / "diagnostic"
    cross_rows, cross_sr = [], {}
    for w in WINDOW_ORDER:
        for f in FREQS:
            for va, vb in VAL_PAIRS:
                a, b = (w, f, va), (w, f, vb)
                f_ab = xdir / f"{w}_val{round(va * 100)}_set{round(vb * 100)}.parquet"
                f_ba = xdir / f"{w}_val{round(vb * 100)}_set{round(va * 100)}.parquet"
                if a not in have or b not in have or not (f_ab.exists() and f_ba.exists()):
                    continue
                P = {"aa": predictions(res / "predictions" / model / f"{w}_val{round(va * 100)}.parquet", f),
                     "bb": predictions(res / "predictions" / model / f"{w}_val{round(vb * 100)}.parquet", f),
                     "ab": predictions(f_ab, f), "ba": predictions(f_ba, f)}
                sr = {"aa": sharpe(series[a]["gross"]), "bb": sharpe(series[b]["gross"])}
                for key, path in (("ab", f_ab), ("ba", f_ba)):
                    if (path, f) not in cross_sr:
                        m, _ = p04.portfolio(P[key].rename(columns={"pred": "signal"}), pf_panel)
                        m.to_parquet(pf_dir / "diagnostic" / f"{model}_{path.stem}_{f}_monthly.parquet")
                        check(m.index.equals(months - pd.offsets.MonthEnd(1)), "cross-fit portfolio months differ")
                        cross_sr[(path, f)] = sharpe(m["long_short"])
                    sr[key] = cross_sr[(path, f)]
                corr = lambda x, y: pred_corrs(P[x], P[y])[1]
                cross_rows.append({
                    "window": w, "freq": f, "val_a": va, "val_b": vb,
                    "spearman_original": corr("aa", "bb"),
                    "spearman_same_settings_b": corr("ab", "bb"), "spearman_same_settings_a": corr("aa", "ba"),
                    "spearman_same_data_a": corr("aa", "ab"), "spearman_same_data_b": corr("ba", "bb"),
                    "sharpe_aa": sr["aa"], "sharpe_ab": sr["ab"], "sharpe_ba": sr["ba"], "sharpe_bb": sr["bb"],
                    "diff_total": sr["aa"] - sr["bb"],
                    "path1_tuning_on_a_data": sr["aa"] - sr["ab"], "path1_data_at_b_settings": sr["ab"] - sr["bb"],
                    "path2_data_at_a_settings": sr["aa"] - sr["ba"], "path2_tuning_on_b_data": sr["ba"] - sr["bb"]})
    cross = pd.DataFrame(cross_rows)

    # Save (every table carries the provisional label)
    for t in (summary, tests, alphas, diag, cross, sensitivity):
        if len(t):
            t.insert(0, "status", p04.RESULT_STATUS)
    sensitivity.to_csv(tables / f"06_{model}_missing_sensitivity.csv", index=False, float_format="%.6g")
    summary.to_csv(tables / f"06_{model}_summary.csv", index=False, float_format="%.6g")
    tests.to_csv(tables / f"06_{model}_sharpe_tests.csv", index=False, float_format="%.6g")
    alphas.to_csv(tables / f"06_{model}_alphas.csv", index=False, float_format="%.6g")
    diag.to_csv(tables / f"06_{model}_valshare_diag.csv", index=False, float_format="%.6g")
    if len(cross):
        cross.to_csv(tables / f"06_{model}_crossfit.csv", index=False, float_format="%.6g")

    # Report
    lab = lambda w, f, v: f"{w:<10}{f:<7}{v:>4.2f}"
    num = lambda x, spec, width: f"{'':>{width}}" if pd.isna(x) else f"{x:>{width}{spec}}"
    print(f"\nAll results {p04.RESULT_STATUS.upper()}.")
    print(f"\nSummary, return months {months[0]:%Y-%m} to {months[-1]:%Y-%m}; long-short decile, capped value weights, "
          f"missing returns = 0\n  {COST_LABEL}. Alphas: FF5 + Mom, Newey-West {NW_LAGS} lags. p: LW bootstrap.")
    print(f"  {'window':<10}{'freq':<7}{'val':>4}{'SR gross':>9}{'SR net':>8}{'mean g':>8}{'mean n':>8}{'cost':>7}"
          f"{'turn':>6}{'k=0':>6}{'k eq':>6}{'alpha g':>8}{'t':>6}{'alpha n':>8}{'t':>6}{'p y/o g':>8}{'p y/o n':>8}"
          f"{'p ref g':>8}{'p ref n':>8}  note")
    for r in summary.itertuples():
        print(f"  {lab(r.window, r.freq, r.val_share)}{r.gross_sharpe:>9.2f}{r.net_sharpe:>8.2f}{r.gross_mean:>8.2%}"
              f"{r.net_mean:>8.2%}{r.cost_mean:>7.2%}{r.turnover:>6.0%}{num(r.breakeven_k, '.2f', 6)}{num(r.breakeven_k_equal, '.2f', 6)}"
              f"{r.alpha_gross_m:>8.2%}{r.alpha_gross_t:>6.2f}{r.alpha_net_m:>8.2%}{r.alpha_net_t:>6.2f}"
              f"{num(r.p_yearly_vs_once_gross, '.3f', 8)}{num(r.p_yearly_vs_once_net, '.3f', 8)}"
              f"{num(r.p_vs_ref_gross, '.3f', 8)}{num(r.p_vs_ref_net, '.3f', 8)}"
              f"  {r.note}")
    print("  k=0: multiple of costs at which the net Sharpe ratio is zero (blank: gross mean <= 0); k eq: multiple at "
          f"which yearly and once have equal net Sharpe ratios (blank: none in [0, {K_MAX:g}])")
    print(f"  average cost per month: long side {summary['cost_long'].mean():.2%}, short side {summary['cost_short'].mean():.2%} "
          "(mean over setups)")

    vs = ["baseline", "m30_all", "m30_distressed"]
    piv = sensitivity.set_index(["window", "freq", "val_share", "missing_version"])
    print("\nMissing-return sensitivity: held missing returns = 0 (baseline) | -30% all | -30% distressed only")
    print(f"  {'window':<10}{'freq':<7}{'val':>4}{'held':>6}{'filled':>12}{'SR gross':>21}{'SR net':>22}"
          f"{'alpha gross (t)':>36}{'alpha net (t)':>40}")
    for s in setups:
        g = [piv.loc[(*s, v)] for v in vs]
        print(f"  {lab(*s)}{g[0]['held_missing']:>6}{g[1]['filled_with_m30']:>6}{g[2]['filled_with_m30']:>6}"
              + "".join(f"{x['gross_sharpe']:>7.2f}" for x in g) + " "
              + "".join(f"{x['net_sharpe']:>7.2f}" for x in g) + " "
              + "".join(f"{x['alpha_gross_m']:>6.2%} ({x['alpha_gross_t']:>5.2f})" for x in g) + " "
              + "".join(f"{x['alpha_net_m']:>6.2%} ({x['alpha_net_t']:>5.2f})" for x in g))
    print("  held: held stock-months with a missing return; filled: of these, set to -30% (all | distressed)")
    print("\nYearly vs once under each missing-return version: Sharpe difference (yearly - once), LW bootstrap p")
    print(f"  {'window':<10}{'val':>5}" + "".join(f"{MISSING_LABELS[v]:>24}" for v in vs) + "   (gross | net)")
    for (w, f, v) in [x for x in setups if x[1] == "yearly"]:
        cells = []
        for ver in vs:
            x = piv.loc[(w, f, v, ver)]
            cells.append(f"{x['yearly_minus_once_gross']:>+6.2f} {x['p_yearly_vs_once_gross']:.2f}|"
                         f"{x['yearly_minus_once_net']:>+6.2f} {x['p_yearly_vs_once_net']:.2f}")
        print(f"  {w:<10}{v:>5.2f}" + "".join(f"{c:>24}" for c in cells))

    for g, title in (("yearly_vs_once", "Yearly vs once"), ("val_shares", "Validation shares"),
                     ("vs_reference", "Each setup vs the reference (expanding, yearly, 0.30)")):
        t = tests[(tests["group"] == g) & (tests["missing_version"] == "baseline")]
        print(f"\n{title} (missing returns = 0): Ledoit-Wolf, difference of annualized Sharpe ratios (A - B); p bootstrap / HAC / Holm")
        print(f"  {'A':<21}  {'B':<21}{'gross diff':>11}{'p boot':>7}{'p hac':>7}{'p holm':>7}"
              f"{'net diff':>11}{'p boot':>7}{'p hac':>7}{'p holm':>7}")
        for (key, grp) in t.groupby(["a_window", "a_freq", "a_val", "b_window", "b_freq", "b_val"], sort=False):
            gr, nt = grp[grp["returns"] == "gross"].iloc[0], grp[grp["returns"] == "net"].iloc[0]
            fmt = lambda r: (f"{r['diff']:>+11.2f}{r['p_boot']:>7.3f}{r['p_hac']:>7.3f}{r['p_holm']:>7.3f}" if not r['identical']
                             else f"{'identical':>11}{'':>21}")
            print(f"  {lab(*key[:3]):<21}  {lab(*key[3:]):<21}{fmt(gr)}{fmt(nt)}")

    print(f"\nAlphas (monthly) and factor loadings, FF5 + Mom, Newey-West {NW_LAGS} lags")
    print(f"  {'window':<10}{'freq':<7}{'val':>4} {'ret':<6}{'alpha':>7}{'t':>6}{'ann.':>7}"
          + "".join(f"{f:>8}" for f in FACTORS) + f"{'R2':>6}")
    for _, r in alphas.iterrows():
        print(f"  {lab(r['window'], r['freq'], r['val_share'])} {r['returns']:<6}{r['alpha_m']:>7.2%}{r['alpha_t']:>6.2f}"
              f"{r['alpha_ann']:>7.1%}" + "".join(f"{r['b_' + f]:>8.2f}" for f in FACTORS) + f"{r['r2']:>6.2f}")

    print("\nValidation-share diagnostics: chosen settings and model size per setup")
    print(diag[diag["kind"] == "setup"].drop(columns=["kind", "val_b"], errors="ignore").dropna(axis=1, how="all")
          .to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("\nPairs of validation shares: average monthly correlation of predictions, side overlap, LS return correlation")
    print(diag[diag["kind"] == "pair"].dropna(axis=1, how="all").drop(columns=["kind"])
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(cross):
        print("\nCross-fit diagnostic (a's training part with b's settings; diagnostic only, not a reported setup).")
        print("  Spearman correlation of predictions: original pair; same settings, different training data; "
              "same training data, different settings")
        print(f"  {'window':<10}{'freq':<7}{'a':>5}{'b':>5}{'orig':>7}{'set b':>7}{'set a':>7}{'data a':>7}{'data b':>7}"
              f"  | Sharpe {'aa':>6}{'ab':>6}{'ba':>6}{'bb':>6} | diff  = tuning + data (path 1) = data + tuning (path 2)")
        for r in cross.itertuples():
            print(f"  {r.window:<10}{r.freq:<7}{r.val_a:>5.2f}{r.val_b:>5.2f}{r.spearman_original:>7.3f}"
                  f"{r.spearman_same_settings_b:>7.3f}{r.spearman_same_settings_a:>7.3f}{r.spearman_same_data_a:>7.3f}"
                  f"{r.spearman_same_data_b:>7.3f}  |        {r.sharpe_aa:>6.2f}{r.sharpe_ab:>6.2f}{r.sharpe_ba:>6.2f}"
                  f"{r.sharpe_bb:>6.2f} | {r.diff_total:>+5.2f} = {r.path1_tuning_on_a_data:>+5.2f} {r.path1_data_at_b_settings:>+5.2f}"
                  f"          = {r.path2_data_at_a_settings:>+5.2f} {r.path2_tuning_on_b_data:>+5.2f}")
    else:
        print("\nNo cross-fit predictions found (run 05_train.py --cross-fit); decomposition skipped.")
    print(f"\nSaved results/tables/06_{model}_*.csv; done in {time.time() - t0:,.0f}s")


if __name__ == "__main__":
    main()
