"""Step 1: clean the JKP data into data/clean.parquet.

Usage:
    python src/01_clean.py data/*.csv [--chars 153|all] [--out data/clean.parquet]
                           [--delist data/crsp_delist/Data_Delisting.csv]

The input may be one or several CSV or parquet files (e.g. split by decade). They are
read with DuckDB, which streams them; the full data is never loaded into memory.

Steps (rows are counted at each one):
  1. keep excntry = 'USA' (applied while reading, so non-US rows are never loaded)
  2. check that common, exch_main, primary_sec and obs_main are all 1 (stop if not)
  3. keep source_crsp = 1, so id is the PERMNO and returns come from CRSP
  4. check that (id, eom) is unique (stop if not)
  5. drop rows with missing me
  6. CRSP delisting returns (--delist, the CRSP CIZ delisting file, one row per PERMNO):
     JKP already compounds delret into the return of the delisting month, but our
     vintage counts it twice when CRSP's MthRet already includes it (fixed in jkp-data on
     02-09-2026). For each delisting row t (the month of delistingdt) whose previous row
     is month t-1, ratio = (1 + ret) / price relative (prices adjusted with adjfct when
     available) is compared with (1 + delret)^k:
       - k = 2 (counted twice): ret_exc of row t and ret_exc_lead1m of row t-1 become
         (1 + ret) / (1 + delret) - 1 - rf, rf = ret - ret_exc of row t  -> 'fixed'
       - k = 1 (counted once), delret = 0 or -1 (both readings agree)  -> 'unchanged'
       - delret missing: ratio 0.70 is JKP's Shumway -30%; on Nasdaq (primaryexch Q) it
         becomes x 0.45 (Shumway & Warther -55%)  -> 'nasdaq55'; ratio 1 -> 'unchanged'
       Exact match first (|ratio - m| <= TOL); if neither or both readings match, a dividend
       in the delisting month is allowed: the reading whose implied yield ratio / m - 1 lies
       in [0, 5%] wins (delist_fix_how = 'dividend'). Neither or both -> 'ambiguous', left as is.
     New columns: ret_exc_lead1m_jkp (JKP's target, untouched), delist_fix and
     delist_fix_how on row t-1 (empty if the target month is not a delisting month), and
     delisted on row t (the stock delists during month t; known at t, so it is no
     look-ahead: such stocks are left out of the month-t portfolios in 04).
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
import numpy as np

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
# Read for the delisting step only (not written out).
PRICE_COLS = {"ret": "DOUBLE", "prc": "DOUBLE", "adjfct": "DOUBLE"}
# Delisting step: exact-match tolerance on the ratio, largest implied dividend yield,
# JKP's imputed -30% and the Nasdaq value that replaces it.
TOL, DIV_MAX, JKP_IMPUTED, NASDAQ_IMPUTED = 1e-4, 0.05, 0.70, 0.45
NEW_COLS = {"ret_exc_lead1m_jkp": "DOUBLE", "delist_fix": "VARCHAR", "delist_fix_how": "VARCHAR",
            "delisted": "BOOLEAN"}


def load_delistings(con, path):
    """The CRSP CIZ delisting file as table dl (one row per PERMNO)."""
    con.execute(f"""CREATE TABLE dl AS SELECT permno, delistingdt, delret, delactiontype, delreasontype, primaryexch
                    FROM read_csv({sql_str(path)}, header = true, types = {{'permno': 'BIGINT', 'delistingdt': 'DATE',
                    'delret': 'DOUBLE', 'delactiontype': 'VARCHAR', 'delreasontype': 'VARCHAR', 'primaryexch': 'VARCHAR'}})""")
    n, perm, lo, hi = con.execute("SELECT count(*), count(DISTINCT permno), min(delistingdt), max(delistingdt) FROM dl").fetchone()
    if n != perm:
        sys.exit(f"{path}: {n - perm:,} PERMNOs appear more than once")
    return n, lo, hi


def classify_delisting(r):
    """(flag, how, multiplier m that JKP applied) for one delisting row; see step 6."""
    d, ratio = r.delret, r.ratio
    if not np.isfinite(ratio):
        return "ambiguous", "no ratio", np.nan
    if np.isfinite(d) and d in (0.0, -1.0):
        return "unchanged", "readings agree", 1 + d
    if np.isfinite(d):
        readings = {"fixed": (1 + d) ** 2, "unchanged": 1 + d}
    else:
        readings = {"nasdaq55" if r.primaryexch == "Q" else "unchanged": JKP_IMPUTED, "unchanged_none": 1.0}
    exact = [k for k, m in readings.items() if abs(ratio - m) <= TOL]
    if len(exact) == 1:
        return exact[0].replace("_none", ""), "exact", readings[exact[0]]
    div = [k for k, m in readings.items() if -TOL <= ratio / m - 1 <= DIV_MAX]
    if len(div) == 1:
        return div[0].replace("_none", ""), "dividend", readings[div[0]]
    return "ambiguous", "both readings fit" if len(div) == 2 else "neither reading fits", np.nan


def fix_delistings(con, path):
    """Step 6 on table d; returns report lines."""
    n_file, lo, hi = load_delistings(con, path)
    for c, t in NEW_COLS.items():
        con.execute(f"ALTER TABLE d ADD COLUMN {c} {t}")
    con.execute("UPDATE d SET ret_exc_lead1m_jkp = ret_exc_lead1m, delisted = false")
    con.execute("""UPDATE d SET delisted = true FROM dl
                   WHERE dl.permno = d.id AND date_trunc('month', dl.delistingdt) = date_trunc('month', d.eom)""")
    rows = con.execute("""
        SELECT t.id, t.eom, t.ret, t.ret_exc, t.prc, t.adjfct, dl.delret, dl.delactiontype, dl.delreasontype,
               dl.primaryexch, p.eom AS eom_prev, p.prc AS prc_prev, p.adjfct AS adj_prev, p.ret_exc_lead1m AS y_prev
        FROM d t JOIN dl ON dl.permno = t.id AND date_trunc('month', dl.delistingdt) = date_trunc('month', t.eom)
        LEFT JOIN d p ON p.id = t.id AND p.eom = last_day(t.eom - INTERVAL 1 MONTH)""").df()
    n_in_data = len(rows)
    rows = rows[rows["eom_prev"].notna()].copy()
    no_prev = n_in_data - len(rows)
    rel = (rows["prc"] / rows["adjfct"]) / (rows["prc_prev"] / rows["adj_prev"])
    rel = rel.where(np.isfinite(rel) & (rel > 0), rows["prc"] / rows["prc_prev"])
    rows["ratio"] = ((1 + rows["ret"]) / rel.where(np.isfinite(rel) & (rel > 0))).astype(float)
    rows["delret"] = rows["delret"].astype(float)
    out = [classify_delisting(r) for r in rows.itertuples()]
    rows["delist_fix"], rows["delist_fix_how"], rows["m"] = [o[0] for o in out], [o[1] for o in out], [o[2] for o in out]
    # The target of row t-1 must be JKP's return of month t
    mismatch = ~np.isclose(rows["y_prev"], rows["ret_exc"], rtol=0, atol=1e-9) & rows["y_prev"].notna()
    rows.loc[mismatch, ["delist_fix", "delist_fix_how"]] = ["ambiguous", "target differs from next ret_exc"]
    rf = rows["ret"] - rows["ret_exc"]
    rows["new"] = rows["ret_exc"]
    fixed = rows["delist_fix"] == "fixed"
    rows.loc[fixed, "new"] = (1 + rows["ret"]) / (1 + rows["delret"]) - 1 - rf
    nas = rows["delist_fix"] == "nasdaq55"
    rows.loc[nas, "new"] = (1 + rows["ret"]) * NASDAQ_IMPUTED / JKP_IMPUTED - 1 - rf
    if not np.isfinite(rows.loc[fixed | nas, "new"]).all():
        sys.exit("Delisting step: a corrected return is not finite")

    upd = rows[["id", "eom", "eom_prev", "new", "delist_fix", "delist_fix_how"]]
    con.register("upd", upd)
    con.execute("""UPDATE d SET ret_exc = upd.new FROM upd
                   WHERE d.id = upd.id AND d.eom = upd.eom AND upd.delist_fix IN ('fixed', 'nasdaq55')""")
    con.execute("""UPDATE d SET delist_fix = upd.delist_fix, delist_fix_how = upd.delist_fix_how,
                   ret_exc_lead1m = CASE WHEN upd.delist_fix IN ('fixed', 'nasdaq55') AND d.ret_exc_lead1m IS NOT NULL
                                         THEN upd.new ELSE d.ret_exc_lead1m END
                   FROM upd WHERE d.id = upd.id AND d.eom = CAST(upd.eom_prev AS DATE)""")
    con.unregister("upd")

    chg = (rows["new"] - rows["ret_exc"])
    lines = [f"Delisting file: {n_file:,} PERMNOs, delistings {lo} to {hi}",
             f"Delisting months in the data: {n_in_data:,}; with a row in month t-1: {len(rows):,} "
             f"(no previous month, nothing to fix: {no_prev:,}); target of t-1 differs from ret_exc of t: {int(mismatch.sum()):,}",
             f"  {'flag':<11}{'how':<34}{'rows':>8}{'mean change':>13}{'mean |change|':>15}{'max |delret|':>14}"]
    for (f, h), g in rows.groupby(["delist_fix", "delist_fix_how"]):
        c = chg[g.index]
        lines.append(f"  {f:<11}{h:<34}{len(g):>8,}{c.mean():>13.4f}{c.abs().mean():>15.4f}{g['delret'].abs().max():>14.4f}")
    amb = rows[rows["delist_fix"] == "ambiguous"]
    big = amb[amb["delret"].abs() > 0.20].sort_values("eom")
    lines.append(f"Ambiguous (left unchanged): {len(amb):,}; |delret| > 5%: {int((amb['delret'].abs() > 0.05).sum()):,}; "
                 f"> 20%: {len(big):,}" + (" (all US rows; the panel subset is reported later)" if len(big) else ""))
    for r in big.head(50).itertuples():
        lines.append(f"    {r.id:>7} {r.eom:%Y-%m} delret {r.delret:>8.4f} {r.delactiontype}/{r.delreasontype} "
                     f"ratio {r.ratio:.4f} ({r.delist_fix_how})")
    return lines


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


def file_query(f, cols, needed, where=None):
    """SELECT of the needed columns from one file, cast to fixed types, optionally filtered."""
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
    return f"SELECT {select} FROM {source}" + (f" WHERE {where}" if where else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="input files or glob patterns (CSV or parquet)")
    parser.add_argument("--chars", choices=["153", "all"], default="153",
                        help="the 153 JKP characteristics (default) or all characteristic columns")
    parser.add_argument("--out", default=str(ROOT / "data" / "clean.parquet"))
    parser.add_argument("--delist", default=str(ROOT / "data" / "crsp_delist" / "Data_Delisting.csv"),
                        help="CRSP CIZ delisting file (step 6)")
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
    needed.update(PRICE_COLS)
    needed.update({c: "FLOAT" for c in chars})

    print(f"Reading {len(files)} file(s): {', '.join(Path(f).name for f in files)}")
    print(f"Characteristics: {args.chars} ({len(chars)} columns incl. {', '.join(ALWAYS_KEEP)})")
    n_in = con.execute("SELECT count(*) FROM (" + " UNION ALL ".join(
        file_query(f, file_cols[f], needed) for f in files) + ")").fetchone()[0]
    con.execute("CREATE TABLE d AS " + " UNION ALL ".join(
        file_query(f, file_cols[f], needed, where="excntry = 'USA'") for f in files))

    count = lambda: con.execute("SELECT count(*) FROM d").fetchone()[0]
    n = count()
    report = [("keep excntry = 'USA'", f"-{n_in - n:,}", f"{n:,}")]

    def drop(label, where):
        nonlocal n
        con.execute(f"DELETE FROM d WHERE {where}")
        new = count()
        report.append((label, f"-{n - new:,}", f"{new:,}"))
        n = new

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
    delist_report = fix_delistings(con, args.delist)
    report.append(("CRSP delisting returns", "fixed", f"{n:,}"))
    no_target = con.execute("SELECT count(*) FROM d WHERE ret_exc_lead1m IS NULL").fetchone()[0]

    out = Path(args.out)
    tmp_out = out.with_suffix(".parquet.tmp")
    cols = ", ".join(f'"{c}"' for c in list(KEYS) + list(NEW_COLS) + chars)
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
    print("\n" + "\n".join(delist_report))
    print(f"\nDate range: {per_month[0][0]} to {per_month[-1][0]} ({len(per_month)} months)")
    print(f"Stocks per month: min {counts[0]:,}, median {counts[len(counts) // 2]:,}, max {counts[-1]:,}; "
          f"first month {per_month[0][1]:,}, last month {per_month[-1][1]:,}")
    print("Year  months  avg stocks/month")
    for year, months, avg in per_year:
        print(f"{year}  {months:>6}  {avg:>16,.0f}")
    print(f"\nWrote {out} ({out.stat().st_size / 1e6:,.1f} MB, {len(KEYS) + len(NEW_COLS) + len(chars)} columns) "
          f"in {time.time() - t0:,.0f}s")

    con.close()
    if not any(TMP_DIR.iterdir()):
        TMP_DIR.rmdir()


if __name__ == "__main__":
    main()
