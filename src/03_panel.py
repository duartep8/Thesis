"""Step 3: build the modelling panel, data/panel.parquet.

Usage:
    python src/03_panel.py [--input data/clean.parquet] [--out data/panel.parquet] [--universe 1000]
                           [--start 1974-12] [--reports data/reports]

Steps, each using only the month-t cross-section (no look-ahead):
  1. universe: from the start month (December 1974, so the first return predicted is
     January 1975), the 1,000 largest stocks by me in each month (ties broken by id); rows with a missing ret_exc_lead1m stay in (they are
     dropped only at training)
  1b. suspect returns: ret_exc_lead1m <= -0.90 while the stock's me in t+1 is at least 50%
     of its me in t (a near-total loss that the market value does not show). The original
     is kept in ret_exc_lead1m_raw, ret_exc_lead1m_suspect flags the row, and
     ret_exc_lead1m is set to missing (dropped from training; 0 and logged in portfolios).
     This touches only the target, which is month t+1 information anyway; me in t+1 is
     not kept in the panel.
  2. size_decile (1-10): me deciles within the universe each month
  3. raw liquidity for trading costs: bidaskhl_21d_raw and dolvol_126d_raw, copied before
     ranking; missing values filled with the eom x size_decile median, else the month
     median; *_filled flags mark them
  4. the 153 JKP characteristics ranked each month to [-1, 1]: x = 2(r - 1)/(n - 1) - 1,
     with average ranks r among the n stocks that have a value; missing values = 0
     (Gu, Kelly & Xiu 2020). No winsorizing.
The report also gives stocks per month, the missing-target share by year, the 20 largest
and smallest ret_exc_lead1m in the universe, and each characteristic's first month and
missing share by decade within the universe (before filling), saved to
<reports>/coverage_by_decade.csv.
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHARS_153_FILE = ROOT / "src" / "characteristics_153.txt"
KEYS = ["id", "eom", "size_grp", "me", "ret_exc", "ret_exc_lead1m"]
LIQUIDITY = ["bidaskhl_21d", "dolvol_126d"]


def sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(ROOT / "data" / "clean.parquet"))
    parser.add_argument("--out", default=str(ROOT / "data" / "panel.parquet"))
    parser.add_argument("--universe", type=int, default=1000, help="largest stocks by me per month (default 1000)")
    parser.add_argument("--start", default="1974-12", help="first month of the panel, YYYY-MM (default 1974-12: "
                        "the first return predicted is January 1975)")
    parser.add_argument("--reports", default=str(ROOT / "data" / "reports"), help="folder for coverage_by_decade.csv")
    args = parser.parse_args()
    t0 = time.time()
    N = args.universe

    chars = [l.strip() for l in CHARS_153_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]
    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    src = f"read_parquet({sql_str(args.input)})"
    cols_in = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    missing_cols = [c for c in KEYS + chars if c not in cols_in]
    if missing_cols:
        sys.exit(f"{args.input} lacks {len(missing_cols)} columns, e.g. {missing_cols[:10]}")

    # 1. Universe
    n_in = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    start = f"DATE '{args.start}-01'"
    n_from_start = con.execute(f"SELECT count(*) FROM {src} WHERE eom >= {start}").fetchone()[0]
    available = con.execute(f"SELECT eom, count(*) AS n FROM {src} WHERE eom >= {start} "
                            "GROUP BY eom").df().set_index("eom")["n"]
    select = ", ".join(f'u."{c}"' for c in KEYS + chars)
    # me_lead1m: the same stock's me at the next month end, if it is in the input (for step 1b only)
    df = con.execute(f"""
        SELECT {select}, n.me AS me_lead1m FROM (
            SELECT *, row_number() OVER (PARTITION BY eom ORDER BY me DESC, id) AS rank_me
            FROM {src} WHERE eom >= {start}
        ) u
        LEFT JOIN (SELECT id, eom, me FROM {src}) n ON n.id = u.id AND n.eom = last_day(u.eom + INTERVAL 1 MONTH)
        WHERE u.rank_me <= {N}
        ORDER BY u.eom, u.id""").df()
    months = df.groupby("eom").size()
    if df.duplicated(["id", "eom"]).any():
        sys.exit("Duplicate (id, eom) rows in the universe")
    expected = available.clip(upper=N).reindex(months.index)
    if len(months) != len(available) or (months != expected).any():
        sys.exit("Some months do not have min(universe, available) stocks")
    short = months[months < N]
    no_target_raw = int(df["ret_exc_lead1m"].isna().sum())

    # 1b. Suspect returns (NaN me_lead1m compares False, so stocks absent in t+1 are not flagged)
    suspect = (df["ret_exc_lead1m"] <= -0.90) & (df["me_lead1m"] >= 0.5 * df["me"])
    flagged = df.loc[suspect, ["id", "eom", "me", "me_lead1m", "ret_exc_lead1m"]].copy()
    df["ret_exc_lead1m_raw"] = df["ret_exc_lead1m"]
    df["ret_exc_lead1m_suspect"] = suspect
    df.loc[suspect, "ret_exc_lead1m"] = np.nan
    no_target = int(df["ret_exc_lead1m"].isna().sum())

    # 2. Size deciles within the universe (df is sorted by eom, id, so ties in me go by id)
    by_month = df.groupby("eom")
    me_rank = by_month["me"].rank(method="first")
    df["size_decile"] = np.ceil(10 * me_rank / by_month["me"].transform("size")).astype("int8")

    # 3. Raw liquidity, filled with the eom x size_decile median, else the month median
    fill_report = []
    for c in LIQUIDITY:
        raw = df[c].astype("float64")
        miss = raw.isna()
        decile_median = df.groupby(["eom", "size_decile"])[c].transform("median")
        month_median = df.groupby("eom")[c].transform("median")
        filled = raw.fillna(decile_median).fillna(month_median)
        fill_report.append((c, int(miss.sum()), int((miss & decile_median.notna()).sum()),
                            int((miss & decile_median.isna() & month_median.notna()).sum()),
                            int(filled.isna().sum())))
        df[f"{c}_raw"] = filled.astype("float32")
        df[f"{c}_filled"] = miss

    # 4. Rank the characteristics to [-1, 1]; missing = 0
    miss_share = df[chars].isna().mean()
    has_data = by_month[chars].count() > 0
    first_month = has_data.idxmax().where(has_data.any())
    # Missing share by decade within the universe, before missing values become 0
    coverage = df[chars].isna().groupby(df["eom"].dt.year // 10 * 10).mean().T
    decades = [f"{d}s" for d in coverage.columns]
    coverage.columns = decades
    coverage.insert(0, "first_month", first_month.reindex(chars))
    ranks = by_month[chars].rank(method="average")
    n_obs = by_month[chars].transform("count")
    ranked = (2 * (ranks - 1) / (n_obs - 1) - 1).fillna(0)  # n = 1 gives 0/0 -> 0
    if ranked.isna().any().any() or (ranked.abs() > 1 + 1e-9).any().any():
        sys.exit("Ranked characteristics outside [-1, 1] or NaN")
    df[chars] = ranked.astype("float32")

    # Write
    extra = ["ret_exc_lead1m_raw", "ret_exc_lead1m_suspect", "size_decile"] + [f"{c}_raw" for c in LIQUIDITY] + [f"{c}_filled" for c in LIQUIDITY]
    panel = df[KEYS + extra + chars]
    out = Path(args.out)
    tmp_out = out.with_suffix(".parquet.tmp")
    con.register("panel", panel)
    con.execute(f"COPY (SELECT * REPLACE (CAST(eom AS DATE) AS eom) FROM panel ORDER BY eom, id) "
                f"TO {sql_str(tmp_out)} (FORMAT parquet, COMPRESSION zstd)")
    tmp_out.replace(out)

    # Report
    print(f"Input: {args.input} ({n_in:,} rows; from {args.start}: {n_from_start:,} rows, {len(available)} months)")
    print(f"Universe: {N:,} largest by me per month -> {len(df):,} rows, "
          f"{months.index.min().date()} to {months.index.max().date()}")
    print(f"Months with fewer than {N:,} stocks: {len(short)}"
          + (f" (min {short.min():,}, e.g. {[d.date().isoformat() for d in short.index[:3]]})" if len(short) else ""))
    print(f"Rows with missing ret_exc_lead1m (kept): {no_target:,} = {no_target_raw:,} missing in the data "
          f"+ {len(flagged):,} suspect, set to missing")
    print(f"Stocks per month: " + (f"exactly {N:,} in every month" if (months == N).all() else f"not always {N:,}")
          + "; by decade (min-max): " + ", ".join(f"{d}s {g.min():,}-{g.max():,}"
                                                for d, g in months.groupby(months.index.year // 10 * 10)))
    last = months.index.max()
    miss_target = df["ret_exc_lead1m"].isna().groupby(df["eom"].dt.year).mean()
    print(f"Missing ret_exc_lead1m: {no_target / len(df):.2%} of rows; in the last month ({last.date()}) "
          f"{df.loc[df['eom'] == last, 'ret_exc_lead1m'].isna().mean():.0%}. By year:")
    items = [f"{y} {v:6.2%}" for y, v in miss_target.items()]
    for i in range(0, len(items), 8):
        print("  " + "   ".join(items[i:i + 8]))

    def show(rows, side):
        for r in rows.itertuples():
            nxt = f"{r.me_lead1m:>12,.1f} {r.me_lead1m / r.me:>10.3f}" if pd.notna(r.me_lead1m) else ""
            print(f"  {side:<9} {r.id:>7} {str(r.eom)[:10]:<10} {r.ret_exc_lead1m:>15,.4f} {r.me:>12,.1f} {nxt}")

    header = f"  {'':9} {'id':>7} {'eom':<10} {'ret_exc_lead1m':>15} {'me':>12} {'me_next':>12} {'me_next/me':>10}"
    print(f"\nSuspect returns (ret_exc_lead1m <= -0.90 and me_next >= 50% of me), set to missing: {len(flagged):,}"
          f"\n(me in USD million; me_next = the stock's me in month t+1; original values shown)")
    print(header)
    show(flagged, "suspect")

    ret = df[["id", "eom", "me", "me_lead1m", "ret_exc_lead1m"]]
    print(f"\nExtreme ret_exc_lead1m in the universe after removing suspect returns "
          f"(me_next blank if the stock is absent in t+1):")
    print(header)
    show(ret.nlargest(20, "ret_exc_lead1m"), "largest")
    show(ret.nsmallest(20, "ret_exc_lead1m"), "smallest")

    print(f"\nMissing characteristics before ranking: {miss_share.mean():.1%} of all values")
    print("Most missing: " + ", ".join(f"{c} {s:.1%}" for c, s in miss_share.nlargest(10).items()))
    never = first_month[first_month.isna()].index.tolist()
    late = first_month.dropna()
    late = late[late > months.index.min()].sort_values()
    print(f"Characteristics with no data in the first month: {len(late) + len(never)}"
          + (f"; never any data: {never}" if never else ""))

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(reports / "coverage_by_decade.csv", index_label="characteristic", float_format="%.4f")
    print("Missing share of the 153 characteristics by decade, within the universe, before filling:")
    print(f"  {'decade':<8}{'mean':>7}{'> 20%':>7}{'> 50%':>7}")
    for d in decades:
        print(f"  {d:<8}{coverage[d].mean():>7.1%}{(coverage[d] > 0.2).sum():>7}{(coverage[d] > 0.5).sum():>7}")
    print(f"\n  {'characteristic':<22}{'first data':<11}" + "".join(f"{d:>7}" for d in decades))
    for c, r in coverage.sort_values("first_month", kind="stable").iterrows():
        print(f"  {c:<22}{str(r['first_month'])[:7]:<11}" + "".join(f"{r[d]:>7.0%}" for d in decades))
    print(f"  (saved to {reports / 'coverage_by_decade.csv'})")

    print("\nLiquidity fill (missing -> eom x size_decile median, month median, still missing):")
    for c, n_miss, n_dec, n_mon, n_left in fill_report:
        print(f"  {c:<14} missing {n_miss:,}: decile {n_dec:,}, month {n_mon:,}, still missing {n_left:,}")

    print(f"\nWrote {out} ({out.stat().st_size / 1e6:,.1f} MB, {panel.shape[1]} columns) in {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    main()
