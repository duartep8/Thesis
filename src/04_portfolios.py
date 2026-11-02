"""Step 4: decile long-short portfolios from any signal; the function every model uses.

Usage:
    python src/04_portfolios.py --signal FILE --column COL --name NAME [--start 1990] [--end 2025]
    python src/04_portfolios.py --sanity
    common options: [--panel data/panel.parquet] [--clean data/clean.parquet] [--out-dir data/portfolios]

Other scripts load this file with importlib (its name starts with a digit) and call
portfolio(signal, panel) and summary(monthly, start, end).

Each month t, using only month-t data:
  - eligible: universe stocks with a non-missing signal (a missing ret_exc_lead1m is not
    known at t, so those stocks stay eligible)
  - sides: sorted by signal, ties broken by id; long the top k, short the bottom k,
    k = round(10% of eligible stocks)
  - weights: within each side proportional to min(me, cap), cap = the NYSE 80th percentile
    = the smallest me among size_grp = 'mega' stocks that month; long +1, short -1
Returns are realized from t to t+1 (ret_exc_lead1m): long = sum(w r) on the long side,
short = the return of the short position = -sum(w r) on the short side, long-short =
long + short. A held stock with a missing return counts as 0 and is logged. Months with
no returns at all (the last month) are left out.
Turnover per side = sum |w_t - w~_t-1| over stocks held in either month (buys and sells:
a full replacement is 200%), where w~ = w (1 + r) / (1 + side return) is last month's
weight after drift, renormalized to sum to 1 on each side (excess returns). Long-short
turnover = long + short. It is missing when the previous month has no portfolio.
Trading costs can be added later from the holdings table: it has every trade
(weight - drifted previous weight), including sales of stocks that left, and the
stock's bidaskhl_21d_raw.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PANEL_COLS = ["id", "eom", "me", "size_grp", "ret_exc_lead1m", "ret_exc_lead1m_suspect", "bidaskhl_21d_raw"]
FRAC = 0.10
SEED = 42
PERIODS = [(1975, 1989), (1990, 2025)]


def load_panel(path):
    panel = pd.read_parquet(path, columns=PANEL_COLS)
    panel["eom"] = pd.to_datetime(panel["eom"])
    return panel.sort_values(["eom", "id"], ignore_index=True)


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def portfolio(signal, panel, frac=FRAC):
    """signal: DataFrame with id, eom and one signal column. Returns (monthly, holdings)."""
    cols = [c for c in signal.columns if c not in ("id", "eom")]
    if len(cols) != 1:
        raise ValueError(f"signal needs id, eom and exactly one signal column, got {list(signal.columns)}")
    sig = signal[["id", "eom", cols[0]]].rename(columns={cols[0]: "signal"})
    sig["eom"] = pd.to_datetime(sig["eom"])
    if sig.duplicated(["id", "eom"]).any():
        raise ValueError("duplicate (id, eom) in the signal")

    cap = panel.loc[panel["size_grp"] == "mega"].groupby("eom")["me"].min().rename("cap")
    n_ret = panel.groupby("eom")["ret_exc_lead1m"].count()
    d = panel.merge(sig, on=["id", "eom"], how="inner")
    d = d[d["signal"].notna() & d["eom"].isin(n_ret.index[n_ret > 0])].join(cap, on="eom")
    check(d["cap"].notna().all(), "a month has no mega stock, so no cap")

    # Sides: position in the month's sort by signal, ties broken by id
    d = d.sort_values(["eom", "signal", "id"], kind="mergesort")
    by_month = d.groupby("eom")
    n = by_month["id"].transform("size")
    k = np.floor(frac * n + 0.5)
    pos = by_month.cumcount()
    d["side"] = np.where(pos >= n - k, "long", np.where(pos < k, "short", ""))
    h = d[d["side"] != ""].copy()

    # Capped value weights a (summing to 1 on each side), returns, and drifted weights
    h["capped"] = np.minimum(h["me"], h["cap"])
    by_side = h.groupby(["eom", "side"])
    h["a"] = h["capped"] / by_side["capped"].transform("sum")
    h["ret_missing"] = h["ret_exc_lead1m"].isna()
    h["r"] = h["ret_exc_lead1m"].fillna(0.0)
    h["side_ret"] = (h["a"] * h["r"]).groupby([h["eom"], h["side"]]).transform("sum")
    h["a_drift"] = h["a"] * (1 + h["r"]) / (1 + h["side_ret"])

    sums = h.groupby(["eom", "side"])[["a", "a_drift"]].sum()
    check(np.allclose(sums.to_numpy(), 1.0, rtol=0, atol=1e-9), "weights or drifted weights do not sum to 1 on a side")
    check(not h.duplicated(["id", "eom"]).any(), "a stock is on both sides in the same month")
    check((h["a"] <= h["cap"] / by_side["capped"].transform("sum") * (1 + 1e-12)).all(), "a weight exceeds its capped share")
    counts = h.groupby(["eom", "side"]).size().unstack("side")
    check((counts["long"] == counts["short"]).all(), "long and short sides differ in size")

    # Turnover per side: |a_t - drifted a_t-1| over stocks held on that side in either month
    months = pd.DatetimeIndex(sorted(h["eom"].unique()))
    has_prev = months.isin(months + pd.offsets.MonthEnd(1))
    prev = h[["id", "eom", "side", "a_drift"]].assign(eom=h["eom"] + pd.offsets.MonthEnd(1))
    tr = h[["id", "eom", "side", "a"]].merge(prev, on=["id", "eom", "side"], how="outer").fillna({"a": 0.0, "a_drift": 0.0})
    tr = tr[tr["eom"].isin(months)]
    turnover = (tr["a"] - tr["a_drift"]).abs().groupby([tr["eom"], tr["side"]]).sum().unstack("side").reindex(months)
    turnover.loc[~has_prev] = np.nan

    side_ret = h.groupby(["eom", "side"])["side_ret"].first().unstack("side")
    monthly = pd.DataFrame({"long": side_ret["long"], "short": -side_ret["short"]})
    monthly["long_short"] = monthly["long"] + monthly["short"]
    monthly["n_long"], monthly["n_short"] = counts["long"], counts["short"]
    monthly["turnover_long"], monthly["turnover_short"] = turnover["long"], turnover["short"]
    monthly["turnover"] = monthly["turnover_long"] + monthly["turnover_short"]
    monthly["n_missing_ret"] = h.groupby("eom")["ret_missing"].sum()
    monthly.index.name = "eom"

    # Holdings with signed weights, plus rows for stocks sold out of the portfolio
    sign = np.where(h["side"] == "long", 1.0, -1.0)
    cur = h[["id", "eom", "side", "ret_exc_lead1m", "ret_missing", "ret_exc_lead1m_suspect"]].assign(weight=sign * h["a"])
    prv = pd.DataFrame({"id": h["id"], "eom": h["eom"] + pd.offsets.MonthEnd(1), "weight_prev": sign * h["a_drift"]})
    hold = cur.merge(prv, on=["id", "eom"], how="outer")
    hold = hold[hold["eom"].isin(months)].merge(panel[["id", "eom", "me", "bidaskhl_21d_raw"]], on=["id", "eom"], how="left")
    hold["weight"] = hold["weight"].fillna(0.0)
    hold["weight_prev"] = hold["weight_prev"].fillna(0.0)
    hold["trade"] = (hold["weight"] - hold["weight_prev"]).where(hold["eom"].isin(months[has_prev]))
    hold = hold.sort_values(["eom", "id"], ignore_index=True)
    return monthly, hold


def summary(monthly, start=None, end=None):
    """Average monthly return, volatility, annualized Sharpe ratio and t-statistic per portfolio."""
    m = monthly[(monthly.index.year >= (start or 0)) & (monthly.index.year <= (end or 9999))]
    rows = {}
    for c in ("long", "short", "long_short"):
        x = m[c]
        T, mu, sd = len(x), x.mean(), x.std(ddof=1)
        rows[c] = {"months": T, "mean": mu, "sd": sd, "sharpe": mu / sd * np.sqrt(12), "t": mu / (sd / np.sqrt(T))}
    return pd.DataFrame(rows).T, m


def report(name, monthly, start=None, end=None):
    s, m = summary(monthly, start, end)
    if m.empty:
        print(f"\n{name}: no months in {start}-{end}")
        return
    yearly = (1 + m["long_short"]).groupby(m.index.year).prod() - 1
    print(f"\n{name}, portfolios formed {m.index.min():%Y-%m} to {m.index.max():%Y-%m} ({len(m)} months)")
    print(f"  {'':<11}{'mean/month':>11}{'sd/month':>10}{'Sharpe (ann.)':>15}{'t-stat':>8}")
    for c, r in s.iterrows():
        print(f"  {c:<11}{r['mean']:>11.2%}{r['sd']:>10.2%}{r['sharpe']:>15.2f}{r['t']:>8.2f}")
    print(f"  turnover/month (buys + sells): long {m['turnover_long'].mean():.0%}, short {m['turnover_short'].mean():.0%}, "
          f"long-short {m['turnover'].mean():.0%}; stocks per side {m['n_long'].mean():.0f}")
    print(f"  long-short positive in {(yearly > 0).sum()} of {len(yearly)} years; "
          f"held stocks with a missing return (counted as 0): {int(m['n_missing_ret'].sum()):,}")


def save(out_dir, name, monthly, hold):
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(out_dir / f"{name}_monthly.parquet")
    hold.to_parquet(out_dir / f"{name}_holdings.parquet", index=False)
    missing = hold.loc[hold["ret_missing"].fillna(False).astype(bool), ["id", "eom", "side", "weight", "ret_exc_lead1m_suspect"]]
    missing.to_csv(out_dir / f"{name}_missing_returns.csv", index=False)
    print(f"Saved {name}: monthly, holdings, and {len(missing):,} missing returns to {out_dir}/{name}_missing_returns.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", help="parquet file with id, eom and the signal column")
    parser.add_argument("--column", help="signal column in --signal")
    parser.add_argument("--name", help="name for the output files")
    parser.add_argument("--start", type=int, help="first formation year in the report")
    parser.add_argument("--end", type=int, help="last formation year in the report")
    parser.add_argument("--sanity", action="store_true", help="momentum and random-signal checks")
    parser.add_argument("--panel", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--clean", default=str(ROOT / "data" / "clean.parquet"), help="unranked ret_12_1 for --sanity")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "portfolios"))
    args = parser.parse_args()
    t0 = time.time()
    out_dir = Path(args.out_dir)
    panel = load_panel(args.panel)

    if args.sanity:
        mom = pd.read_parquet(args.clean, columns=["id", "eom", "ret_12_1"])
        rng = np.random.default_rng(SEED)
        rnd = panel[["id", "eom"]].assign(random=rng.standard_normal(len(panel)))
        for name, sig in [("sanity_momentum", mom), ("sanity_random", rnd)]:
            monthly, hold = portfolio(sig, panel)
            for start, end in PERIODS:
                report(name, monthly, start, end)
            save(out_dir, name, monthly, hold)
    else:
        if not (args.signal and args.column and args.name):
            sys.exit("Give --signal, --column and --name, or --sanity")
        sig = pd.read_parquet(args.signal, columns=["id", "eom", args.column])
        monthly, hold = portfolio(sig, panel)
        report(args.name, monthly, args.start, args.end)
        save(out_dir, args.name, monthly, hold)
    print(f"\nDone in {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    main()
