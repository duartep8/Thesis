"""Step 3: build the modelling panel, data/panel.parquet.

Usage:
    python src/03_panel.py [--input data/clean.parquet] [--out data/panel.parquet] [--universe 1000]

Steps, each using only the month-t cross-section (no look-ahead):
  1. universe: the 1,000 largest stocks by me in each month (ties broken by id);
     rows with a missing ret_exc_lead1m stay in (they are dropped only at training)
  2. size_decile (1-10): me deciles within the universe each month
  3. raw liquidity for trading costs: bidaskhl_21d_raw and dolvol_126d_raw, copied before
     ranking; missing values filled with the eom x size_decile median, else the month
     median; *_filled flags mark them
  4. the 153 JKP characteristics ranked each month to [-1, 1]: x = 2(r - 1)/(n - 1) - 1,
     with average ranks r among the n stocks that have a value; missing values = 0
     (Gu, Kelly & Xiu 2020). No winsorizing.
"""

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

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
    available = con.execute(f"SELECT eom, count(*) AS n FROM {src} GROUP BY eom").df().set_index("eom")["n"]
    select = ", ".join(f'"{c}"' for c in KEYS + chars)
    df = con.execute(f"""
        SELECT {select} FROM (
            SELECT *, row_number() OVER (PARTITION BY eom ORDER BY me DESC, id) AS rank_me FROM {src}
        )
        WHERE rank_me <= {N}
        ORDER BY eom, id""").df()
    months = df.groupby("eom").size()
    if df.duplicated(["id", "eom"]).any():
        sys.exit("Duplicate (id, eom) rows in the universe")
    expected = available.clip(upper=N).reindex(months.index)
    if len(months) != len(available) or (months != expected).any():
        sys.exit("Some months do not have min(universe, available) stocks")
    short = months[months < N]
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
    ranks = by_month[chars].rank(method="average")
    n_obs = by_month[chars].transform("count")
    ranked = (2 * (ranks - 1) / (n_obs - 1) - 1).fillna(0)  # n = 1 gives 0/0 -> 0
    if ranked.isna().any().any() or (ranked.abs() > 1 + 1e-9).any().any():
        sys.exit("Ranked characteristics outside [-1, 1] or NaN")
    df[chars] = ranked.astype("float32")

    # Write
    extra = ["size_decile"] + [f"{c}_raw" for c in LIQUIDITY] + [f"{c}_filled" for c in LIQUIDITY]
    panel = df[KEYS + extra + chars]
    out = Path(args.out)
    tmp_out = out.with_suffix(".parquet.tmp")
    con.register("panel", panel)
    con.execute(f"COPY (SELECT * REPLACE (CAST(eom AS DATE) AS eom) FROM panel ORDER BY eom, id) "
                f"TO {sql_str(tmp_out)} (FORMAT parquet, COMPRESSION zstd)")
    tmp_out.replace(out)

    # Report
    print(f"Input: {args.input} ({n_in:,} rows, {len(available)} months)")
    print(f"Universe: {N:,} largest by me per month -> {len(df):,} rows, "
          f"{months.index.min().date()} to {months.index.max().date()}")
    print(f"Months with fewer than {N:,} stocks: {len(short)}"
          + (f" (min {short.min():,}, e.g. {[d.date().isoformat() for d in short.index[:3]]})" if len(short) else ""))
    print(f"Rows with missing ret_exc_lead1m (kept): {no_target:,}")

    print(f"\nMissing characteristics before ranking: {miss_share.mean():.1%} of all values")
    print("Most missing: " + ", ".join(f"{c} {s:.1%}" for c, s in miss_share.nlargest(10).items()))
    never = first_month[first_month.isna()].index.tolist()
    late = first_month.dropna()
    late = late[late > months.index.min()].sort_values()
    print(f"Characteristics with no data in the first month: {len(late) + len(never)}"
          + (f"; never any data: {never}" if never else ""))
    for c, m in late.items():
        print(f"  {c:<22} first data {m.date()}")

    print("\nLiquidity fill (missing -> eom x size_decile median, month median, still missing):")
    for c, n_miss, n_dec, n_mon, n_left in fill_report:
        print(f"  {c:<14} missing {n_miss:,}: decile {n_dec:,}, month {n_mon:,}, still missing {n_left:,}")

    print(f"\nWrote {out} ({out.stat().st_size / 1e6:,.1f} MB, {panel.shape[1]} columns) in {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    main()
