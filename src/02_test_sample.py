"""Step 2: data/test_sample.parquet, the 200 largest stocks by me in each month.

Usage:
    python src/02_test_sample.py [--input data/clean.parquet] [--out data/test_sample.parquet] [--n 200]

Same columns and full date range as clean.parquet, for quick tests of later scripts.
Ties in me are broken by id, so the sample is the same on every run.
"""

import argparse
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(ROOT / "data" / "clean.parquet"))
    parser.add_argument("--out", default=str(ROOT / "data" / "test_sample.parquet"))
    parser.add_argument("--n", type=int, default=200, help="stocks per month (default 200)")
    args = parser.parse_args()
    t0 = time.time()

    con = duckdb.connect()
    con.execute("SET memory_limit = '4GB'")
    src = f"read_parquet({sql_str(args.input)})"
    out = Path(args.out)
    con.execute(f"""
        COPY (
            SELECT * EXCLUDE (rank_me) FROM (
                SELECT *, row_number() OVER (PARTITION BY eom ORDER BY me DESC, id) AS rank_me FROM {src}
            )
            WHERE rank_me <= {args.n}
            ORDER BY eom, id
        ) TO {sql_str(out)} (FORMAT parquet, COMPRESSION zstd)""")

    sample = f"read_parquet({sql_str(out)})"
    in_cols = [r[:2] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
    out_cols = [r[:2] for r in con.execute(f"DESCRIBE SELECT * FROM {sample}").fetchall()]
    assert in_cols == out_cols, "test sample columns differ from the input"
    n_in, months_in = con.execute(f"SELECT count(*), count(DISTINCT eom) FROM {src}").fetchone()
    n, months, first, last = con.execute(
        f"SELECT count(*), count(DISTINCT eom), min(eom), max(eom) FROM {sample}").fetchone()
    short = con.execute(f"SELECT eom, count(*) AS n FROM {sample} GROUP BY eom HAVING n < {args.n} "
                        "ORDER BY eom").fetchall()
    me_min = con.execute(f"SELECT min(me) FROM {sample} WHERE eom = (SELECT max(eom) FROM {sample})").fetchone()[0]

    print(f"Input: {args.input} ({n_in:,} rows, {months_in} months)")
    print(f"Kept the {args.n} largest stocks by me per month: {n:,} rows, {months} months, {first} to {last}")
    print(f"Months with fewer than {args.n} stocks: {len(short)}" + (f", e.g. {short[:5]}" if short else ""))
    print(f"Smallest me kept in the last month: {me_min:,.0f} (USD million)")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:,.2f} MB, {len(out_cols)} columns) in {time.time() - t0:,.1f}s")


if __name__ == "__main__":
    main()
