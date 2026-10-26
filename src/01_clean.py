"""Step 1: clean the JKP data into data/clean.parquet.

Usage:
    python src/01_clean.py data/*.csv [--chars 153|all] [--out data/clean.parquet]

The input may be one or several CSV or parquet files (e.g. split by decade). They are
read with DuckDB, which streams them; the full data is never loaded into memory.

Steps (rows are counted at each one):
  1. keep excntry = 'USA'
  2. check that common, exch_main, primary_sec and obs_main are all 1 (stop if not)
  3. keep source_crsp = 1, so id is the PERMNO and returns come from CRSP
  4. check that (id, eom) is unique (stop if not)
  5. drop rows with missing me
Rows with a missing ret_exc_lead1m are kept: dropping them would use information
from month t+1. They are dropped only at training time.
Characteristics are stored as float32 and are not winsorized or filled.
"""

import argparse
import csv
import glob
import gzip
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
CHARS_153_FILE = ROOT / "src" / "characteristics_153.txt"
# DuckDB spills to disk here when memory runs short; inside data/ so licensed data stays there.
TMP_DIR = ROOT / "data" / ".duckdb_tmp"

# Identifier and metadata columns of the JKP files. Every other column is a characteristic.
META = [
    "obs_main", "exch_main", "common", "primary_sec", "permno", "date", "permco", "gvkey",
    "iid", "id", "excntry", "eom", "size_grp", "me", "me_company", "ret_exc_lead1m",
    "ret_exc", "ret", "ret_local", "ret_lag_dif", "prc_local", "prc_high", "prc_low",
    "bidask", "curcd", "fx", "gics", "naics", "sic", "ff49", "dolvol", "shares", "tvol",
    "adjfct", "comp_tpci", "crsp_shrcd", "comp_exchg", "crsp_exchcd", "source_crsp",
]
SCREENS = ["common", "exch_main", "primary_sec", "obs_main"]
# Columns kept besides the characteristics, with their types.
KEYS = {
    "id": "BIGINT", "eom": "DATE", "excntry": "VARCHAR", "size_grp": "VARCHAR",
    "me": "DOUBLE", "ret_exc": "DOUBLE", "ret_exc_lead1m": "DOUBLE",
}
# Liquidity measures always kept (for trading costs later), even if not in the chosen set.
ALWAYS_KEEP = ["dolvol_126d", "bidaskhl_21d"]


def sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


def expand_inputs(patterns):
    files = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches:
            sys.exit(f"No input file matches {p}")
        files += matches
    files = list(dict.fromkeys(files))
    for f in files:
        if not f.endswith((".csv", ".csv.gz", ".parquet")):
            sys.exit(f"Unsupported input (need .csv, .csv.gz or .parquet): {f}")
    return files


def columns_of(con, f):
    if f.endswith(".parquet"):
        return [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_str(f)})").fetchall()]
    opener = gzip.open if f.endswith(".gz") else open
    with opener(f, "rt", newline="") as fh:
        return next(csv.reader(fh))


def file_query(f, cols, needed):
    """SELECT of the needed columns from one file, cast to fixed types."""
    missing = [c for c in needed if c not in cols]
    if missing:
        sys.exit(f"{f} lacks {len(missing)} needed columns, e.g. {missing[:10]}")
    if f.endswith(".parquet"):
        source = f"read_parquet({sql_str(f)})"
    else:
        # Give every column a type so DuckDB never has to guess; unused ones are read as text.
        types = ", ".join(f"{sql_str(c)}: {sql_str(needed.get(c, 'VARCHAR'))}" for c in cols)
        source = f"read_csv({sql_str(f)}, header = true, types = {{{types}}})"
    select = ", ".join(f'CAST("{c}" AS {t}) AS "{c}"' for c, t in needed.items())
    return f"SELECT {select} FROM {source}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="input files or glob patterns (CSV or parquet)")
    parser.add_argument("--chars", choices=["153", "all"], default="153",
                        help="the 153 JKP characteristics (default) or all characteristic columns")
    parser.add_argument("--out", default=str(ROOT / "data" / "clean.parquet"))
    args = parser.parse_args()
    t0 = time.time()

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '4GB'; SET temp_directory = {sql_str(TMP_DIR)}; "
                "SET preserve_insertion_order = false")

    files = expand_inputs(args.inputs)
    file_cols = {f: columns_of(con, f) for f in files}
    available = set.intersection(*(set(c) for c in file_cols.values()))
    if args.chars == "153":
        chars = [l.strip() for l in CHARS_153_FILE.read_text().splitlines() if l.strip() and not l.startswith("#")]
    else:
        chars = sorted(available - set(META))
    chars += [c for c in ALWAYS_KEEP if c not in chars]

    needed = dict(KEYS)
    needed.update({c: "INTEGER" for c in SCREENS + ["source_crsp"]})
    needed.update({c: "FLOAT" for c in chars})

    print(f"Reading {len(files)} file(s): {', '.join(Path(f).name for f in files)}")
    print(f"Characteristics: {args.chars} ({len(chars)} columns incl. {', '.join(ALWAYS_KEEP)})")
    con.execute("CREATE TABLE d AS " + " UNION ALL ".join(file_query(f, file_cols[f], needed) for f in files))

    count = lambda: con.execute("SELECT count(*) FROM d").fetchone()[0]
    report = []
    n_in = n = count()

    def drop(label, where):
        nonlocal n
        con.execute(f"DELETE FROM d WHERE {where}")
        new = count()
        report.append((label, f"-{n - new:,}", f"{new:,}"))
        n = new

    drop("keep excntry = 'USA'", "excntry IS DISTINCT FROM 'USA'")

    bad = con.execute("SELECT " + ", ".join(f"count(*) FILTER ({c} IS DISTINCT FROM 1)" for c in SCREENS)
                      + " FROM d").fetchone()
    if any(bad):
        sys.exit("Screen check failed, rows not equal to 1: "
                 + ", ".join(f"{c}={b:,}" for c, b in zip(SCREENS, bad)))
    report.append(("check screens all = 1", "ok", f"{n:,}"))

    drop("keep source_crsp = 1", "source_crsp IS DISTINCT FROM 1")

    dups = con.execute("SELECT id, eom, count(*) AS n FROM d GROUP BY id, eom HAVING n > 1 "
                       "ORDER BY eom, id").fetchall()
    if dups:
        sys.exit(f"{len(dups):,} duplicate (id, eom) pairs, e.g. {dups[:5]}")
    report.append(("check (id, eom) unique", "ok", f"{n:,}"))

    drop("drop missing me", "me IS NULL")
    no_target = con.execute("SELECT count(*) FROM d WHERE ret_exc_lead1m IS NULL").fetchone()[0]

    out = Path(args.out)
    tmp_out = out.with_suffix(".parquet.tmp")
    cols = ", ".join(f'"{c}"' for c in list(KEYS) + chars)
    con.execute(f"COPY (SELECT {cols} FROM d ORDER BY eom, id) TO {sql_str(tmp_out)} "
                "(FORMAT parquet, COMPRESSION zstd)")
    tmp_out.replace(out)

    per_month = con.execute("SELECT eom, count(*) FROM d GROUP BY eom ORDER BY eom").fetchall()
    per_year = con.execute("SELECT year(eom), count(DISTINCT eom), count(*) / count(DISTINCT eom) "
                           "FROM d GROUP BY 1 ORDER BY 1").fetchall()
    counts = sorted(c for _, c in per_month)

    print(f"\nRows in{n_in:>38,}")
    for label, change, rows in report:
        print(f"  {label:<26}{change:>10}{rows:>12}")
    print(f"Rows out{n:>37,}")
    print(f"  of which ret_exc_lead1m missing (kept): {no_target:,}")
    print(f"\nDate range: {per_month[0][0]} to {per_month[-1][0]} ({len(per_month)} months)")
    print(f"Stocks per month: min {counts[0]:,}, median {counts[len(counts) // 2]:,}, max {counts[-1]:,}; "
          f"first month {per_month[0][1]:,}, last month {per_month[-1][1]:,}")
    print("Year  months  avg stocks/month")
    for year, months, avg in per_year:
        print(f"{year}  {months:>6}  {avg:>16,.0f}")
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:,.1f} MB, {len(KEYS) + len(chars)} columns) "
          f"in {time.time() - t0:,.0f}s")

    con.close()
    if not any(TMP_DIR.iterdir()):
        TMP_DIR.rmdir()


if __name__ == "__main__":
    main()
