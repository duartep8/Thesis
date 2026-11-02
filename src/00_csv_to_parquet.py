"""Step 0: convert the raw JKP CSVs to parquet (the raw archive) and verify each one.

Usage:
    python src/00_csv_to_parquet.py data/*.csv [--outdir data/raw_parquet] [--delete-csv]

Each CSV becomes <outdir>/<name>.parquet with every column at full precision (zstd).
Types are fixed, never guessed: gvkey, iid, excntry, size_grp, curcd as text; eom, date
as dates; identifier and code columns as BIGINT; every other column as DOUBLE. BIGINT
columns are read as text and must be whole numbers (DuckDB would silently round 1.5 to 2);
any other value stops the run.

Files are done one at a time, smallest first. Each is written to a .tmp file, checked
against its CSV, and renamed only if every check passes:
  - rows (DuckDB, plus a separate Python line count of the CSV);
  - column names, order and types;
  - numeric columns: non-missing count, sum, min, max; text columns: non-missing and
    distinct counts; gvkey leading zeros; date ranges and distinct ids;
  - 1,000 random rows compared value by value, the CSV side parsed by Python itself;
  - a fresh pyarrow process (no DuckDB): rows, column names and types, min/max of
    eom, me and ret_exc_lead1m, against the CSV figures.
A failed group of checks is logged with both sets of figures and recomputed once on a
single thread; the recomputed result decides. With --delete-csv a CSV is deleted only
after its parquet passes every check. On any remaining failure the run stops and the CSV (and the .tmp parquet) are kept. After the files,
cross-file checks run on every verified parquet in outdir. Results are kept in
<outdir>/verification_state.json (so a stopped run resumes) and the report is written
to <outdir>/verification_report.txt.
"""

import argparse
import calendar
import csv
import datetime as dt
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# DuckDB spills to disk here when memory runs short; inside data/ so licensed data stays there.
TMP_DIR = ROOT / "data" / ".duckdb_tmp"

TEXT = ["gvkey", "iid", "excntry", "size_grp", "curcd"]
DATES = ["eom", "date"]
INTS = [
    "obs_main", "exch_main", "common", "primary_sec", "permno", "permco", "id", "gics",
    "naics", "sic", "ff49", "comp_tpci", "crsp_shrcd", "crsp_exchcd", "comp_exchg", "source_crsp",
]
PYARROW_COLS = ["eom", "me", "ret_exc_lead1m"]
ARROW_TYPES = {"VARCHAR": ("string", "large_string"), "DATE": ("date32[day]",),
               "BIGINT": ("int64",), "DOUBLE": ("double",)}
N_SAMPLE = 1000
# Float sums may differ in the last bits with the order of addition; tolerance relative to sum(|x|).
SUM_TOL = 1e-12
# Space guard: parquet is about 21-29% of the CSV (measured); require 1.3 x 30% plus headroom.
PARQUET_RATIO = 0.30
HEADROOM = 5e9
# Spot check: Apple at the end of 2024, me in USD millions (about $3.8 trillion).
# Parquet-side queries read this many columns at a time; all 443 at once exhausts memory.
PARQUET_BATCH = 60
APPLE_ID, APPLE_EOM, APPLE_ME = 14593, dt.date(2024, 12, 31), (3.5e6, 4.1e6)


def sql_str(s):
    return "'" + str(s).replace("'", "''") + "'"


def kind(c):
    return "VARCHAR" if c in TEXT else "DATE" if c in DATES else "BIGINT" if c in INTS else "DOUBLE"


def csv_header(f):
    with open(f, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def csv_source(f, cols):
    """SELECT of every CSV column with its fixed type; nothing is left for DuckDB to guess."""
    read_types = ", ".join(f"{sql_str(c)}: {sql_str('VARCHAR' if kind(c) == 'BIGINT' else kind(c))}"
                           for c in cols)
    select = []
    for c in cols:
        if kind(c) == "BIGINT":
            select.append(f"CASE WHEN \"{c}\" IS NULL THEN NULL "
                          f"WHEN regexp_full_match(\"{c}\", '-?[0-9]{{1,18}}') THEN CAST(\"{c}\" AS BIGINT) "
                          f"ELSE error('not a whole number in {c}: ' || \"{c}\") END AS \"{c}\"")
        else:
            select.append(f'"{c}"')
    return (f"SELECT {', '.join(select)} FROM read_csv({sql_str(f)}, header = true, auto_detect = false, "
            f"delim = ',', quote = '\"', escape = '\"', dateformat = '%Y-%m-%d', "
            f"columns = {{{read_types}}})")


def stats(con, source, cols, batch=None):
    """Row count, distinct ids, and per-column summaries. In one pass, or with batch, a few
    columns per query (for parquet, which then reads only those columns)."""
    exprs = {"rows": "count(*)", "ids": 'count(DISTINCT "id")',
             "gvkey|zero": "count(*) FILTER (gvkey LIKE '0%')",
             "gvkey|not6": "count(*) FILTER (length(gvkey) <> 6)"}
    for c in cols:
        k, e = kind(c), f'"{c}"'
        exprs[f"{c}|n"] = f"count({e})"
        if k == "VARCHAR":
            exprs[f"{c}|distinct"] = f"count(DISTINCT {e})"
            continue
        exprs[f"{c}|min"], exprs[f"{c}|max"] = f"min({e})", f"max({e})"
        if k == "DATE":
            exprs[f"{c}|distinct"] = f"count(DISTINCT {e})"
        elif k == "BIGINT":
            exprs[f"{c}|sum"] = f"sum({e})"
        else:
            exprs[f"{c}|sum"], exprs[f"{c}|abssum"] = f"fsum({e})", f"fsum(abs({e}))"
    owner = lambda k: None if k == "rows" else "id" if k == "ids" else k.split("|")[0]
    groups = [cols] if batch is None else [cols[i:i + batch] for i in range(0, len(cols), batch)]
    out = {}
    for i, group in enumerate(groups):
        keys = [k for k in exprs if owner(k) in group or (i == 0 and owner(k) is None)]
        proj = ", ".join(f'"{c}"' for c in group)
        row = con.execute(f"SELECT {', '.join(exprs[k] for k in keys)} FROM (SELECT {proj} FROM ({source}))").fetchone()
        out.update(zip(keys, row))
    return {k: v.isoformat() if isinstance(v, dt.date) else v for k, v in out.items()}


def same(x, y):
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def compare_stats(a, b, cols):
    """a = CSV figures, b = parquet figures. Returns [(check, passed, detail)]."""
    checks = [("rows", a["rows"] == b["rows"], f"CSV {a['rows']:,}, parquet {b['rows']:,}")]

    bad, exact_sums, worst = [], 0, 0.0
    num = [c for c in cols if kind(c) in ("BIGINT", "DOUBLE")]
    for c in num:
        for s in ("n", "min", "max"):
            if not same(a[f"{c}|{s}"], b[f"{c}|{s}"]):
                bad.append(f"{c} {s}: CSV {a[f'{c}|{s}']!r}, parquet {b[f'{c}|{s}']!r}")
        x, y = a[f"{c}|sum"], b[f"{c}|sum"]
        if same(x, y):
            exact_sums += 1
        elif kind(c) == "DOUBLE" and x is not None and y is not None:
            gap = abs(x - y) / max(1.0, a[f"{c}|abssum"])
            worst = max(worst, gap)
            if not gap <= SUM_TOL:
                bad.append(f"{c} sum: CSV {x!r}, parquet {y!r} (gap {gap:.1e})")
        else:
            bad.append(f"{c} sum: CSV {x!r}, parquet {y!r}")
    checks.append(("numeric columns: count, sum, min, max", not bad,
                   f"{len(num)} columns; sums exactly equal {exact_sums}/{len(num)}, largest gap "
                   f"{worst:.1e} of sum|x| (tolerance {SUM_TOL:.0e})" + (f"; differ: {bad[:10]}" if bad else "")))

    bad = [f"{c} {s}: CSV {a[f'{c}|{s}']}, parquet {b[f'{c}|{s}']}"
           for c in TEXT for s in ("n", "distinct") if a[f"{c}|{s}"] != b[f"{c}|{s}"]]
    checks.append(("text columns: count, distinct values", not bad,
                   ", ".join(f"{c} {b[c + '|distinct']:,}" for c in TEXT) + (f"; differ: {bad}" if bad else "")))

    ok = a["gvkey|zero"] == b["gvkey|zero"] and a["gvkey|not6"] == b["gvkey|not6"] == 0
    checks.append(("gvkey leading zeros", ok,
                   f"starting with 0: CSV {a['gvkey|zero']:,}, parquet {b['gvkey|zero']:,}; "
                   f"not 6 characters: CSV {a['gvkey|not6']:,}, parquet {b['gvkey|not6']:,}"))

    keys = [f"{c}|{s}" for c in DATES for s in ("n", "min", "max", "distinct")] + ["ids"]
    bad = [f"{k}: CSV {a[k]}, parquet {b[k]}" for k in keys if a[k] != b[k]]
    checks.append(("date ranges and distinct ids", not bad,
                   f"eom {b['eom|min']} to {b['eom|max']} ({b['eom|distinct']} months), "
                   f"date {b['date|min']} to {b['date|max']}, ids {b['ids']:,}" + (f"; differ: {bad}" if bad else "")))
    return checks


def schema_check(con, pq_path, cols):
    got = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_str(pq_path)})").fetchall()
    names, types = [r[0] for r in got], [r[1] for r in got]
    bad_types = [c for c, t in zip(names, types) if t != kind(c)]
    ok = names == cols and not bad_types
    return [("columns: names, order, types", ok,
             f"{len(names)} columns" + ("" if names == cols else "; names or order differ")
             + (f"; wrong types: {bad_types[:10]}" if bad_types else ""))]


def value_matches(k, text, val):
    if text == "":
        return val is None
    if val is None:
        return False
    try:
        if k == "VARCHAR":
            return text == val
        if k == "DATE":
            return dt.date.fromisoformat(text) == val
        if k == "BIGINT":
            return int(text) == val
        return same(float(text), val)
    except ValueError:
        return False


def sample_check(con, csv_path, pq_path, cols, n_rows):
    """Count CSV lines and compare N_SAMPLE random rows value by value, with Python's own parsing."""
    rng = random.Random(Path(csv_path).name)
    picks = sorted(rng.sample(range(n_rows), min(N_SAMPLE, n_rows)))
    want, raw, lines = set(picks), {}, 0
    with open(csv_path, "rb") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            if i in want:
                raw[i] = line.decode("utf-8")
            lines = i + 1
    checks = [("rows (Python line count of the CSV)", lines == n_rows, f"{lines:,} lines after the header")]

    pq_rows = {i: [] for i in picks}
    for j in range(0, len(cols), PARQUET_BATCH):
        col_list = ", ".join(f'"{c}"' for c in cols[j:j + PARQUET_BATCH])
        got = con.execute(f"SELECT file_row_number, {col_list} "
                          f"FROM read_parquet({sql_str(pq_path)}, file_row_number = true) "
                          f"WHERE file_row_number IN ({', '.join(map(str, picks))})").fetchall()
        for r in got:
            pq_rows[r[0]].extend(r[1:])
    kinds = [kind(c) for c in cols]
    bad, n_values = [], 0
    for i in picks:
        fields = next(csv.reader([raw[i]])) if i in raw else None
        if fields is None or len(pq_rows[i]) != len(cols) or len(fields) != len(cols):
            bad.append(f"row {i}: missing or wrong field count")
            continue
        for c, k, text, val in zip(cols, kinds, fields, pq_rows[i]):
            n_values += 1
            if not value_matches(k, text, val):
                bad.append(f"row {i} {c}")
    checks.append((f"{len(picks):,} random rows, value by value", not bad,
                   f"{n_values:,} values compared" + (f"; {len(bad)} differ, e.g. {bad[:5]}" if bad else "")))
    return checks


def pyarrow_figures(pq_path, single=False):
    """Run in a fresh process: read the parquet with pyarrow only and print its figures as JSON."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if single:
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
    f = pq.ParquetFile(pq_path)
    md = f.metadata
    out = {"rows_meta": md.num_rows, "rows_groups": sum(md.row_group(i).num_rows for i in range(md.num_row_groups)),
           "names": f.schema_arrow.names, "types": [str(t) for t in f.schema_arrow.types]}
    rows_read, lo, hi = 0, {}, {}
    for batch in f.iter_batches(columns=PYARROW_COLS, batch_size=1_000_000, use_threads=not single):
        rows_read += batch.num_rows
        for c in PYARROW_COLS:
            mm = pc.min_max(batch.column(c))
            bmin, bmax = mm["min"].as_py(), mm["max"].as_py()
            if bmin is not None:
                lo[c] = bmin if c not in lo else min(lo[c], bmin)
                hi[c] = bmax if c not in hi else max(hi[c], bmax)
    out["rows_read"] = rows_read
    out["minmax"] = {c: [lo.get(c), hi.get(c)] for c in PYARROW_COLS}
    print(json.dumps(out, default=lambda v: v.isoformat()))


def pyarrow_check(pq_path, cols, a, single=False):
    """Independent check in a new process with pyarrow, against the CSV figures a."""
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--pyarrow-figures", str(pq_path)]
    res = subprocess.run(cmd + ["--single-thread"] * single, capture_output=True, text=True)
    if res.returncode != 0:
        return [("pyarrow, fresh process", False, f"failed: {res.stderr.strip()[-300:]}")]
    p = json.loads(res.stdout)
    bad = []
    if not p["rows_meta"] == p["rows_groups"] == p["rows_read"] == a["rows"]:
        bad.append(f"rows {p['rows_meta']:,}/{p['rows_groups']:,}/{p['rows_read']:,} vs CSV {a['rows']:,}")
    if p["names"] != cols:
        bad.append("column names or order")
    wrong = [c for c, t in zip(p["names"], p["types"]) if t not in ARROW_TYPES[kind(c)]]
    if wrong:
        bad.append(f"types of {wrong[:10]}")
    for c in PYARROW_COLS:
        if not (same(p["minmax"][c][0], a[f"{c}|min"]) and same(p["minmax"][c][1], a[f"{c}|max"])):
            bad.append(f"{c} min/max {p['minmax'][c]} vs CSV [{a[f'{c}|min']}, {a[f'{c}|max']}]")
    detail = (f"rows {p['rows_read']:,}, {len(p['names'])} columns, "
              + ", ".join(f"{c} {p['minmax'][c][0]} to {p['minmax'][c][1]}" for c in PYARROW_COLS))
    return [("pyarrow, fresh process: rows, columns, types, min/max", not bad,
             detail + (f"; differ: {bad}" if bad else ""))]


def verify(con, csv_path, pq_path, cols):
    """Every check of one parquet against its CSV. Returns (checks, CSV figures).

    If a group of checks fails, its figures are logged and the whole group is recomputed
    once on a single thread; the recomputed result decides. The report keeps both.
    """
    fig = {}

    def summaries(single):
        fig["csv"] = stats(con, csv_source(csv_path, cols), cols)
        fig["parquet"] = stats(con, f"SELECT * FROM read_parquet({sql_str(pq_path)})", cols, PARQUET_BATCH)
        return compare_stats(fig["csv"], fig["parquet"], cols)

    groups = [summaries,
              lambda single: schema_check(con, pq_path, cols),
              lambda single: sample_check(con, csv_path, pq_path, cols, fig["parquet"]["rows"]),
              lambda single: pyarrow_check(pq_path, cols, fig["csv"], single)]
    checks = []
    for group in groups:
        first = group(False)
        failed = {name: detail for name, ok, detail in first if not ok}
        if not failed:
            checks += first
            continue
        for name, detail in failed.items():
            print(f"  [FAIL] {name}: {detail}", flush=True)
        print("  Recomputing on a single thread...", flush=True)
        threads = con.execute("SELECT current_setting('threads')").fetchone()[0]
        con.execute("SET threads = 1")
        try:
            again = group(True)
        finally:
            con.execute(f"SET threads = {threads}")
        for name, ok, detail in again:
            if name in failed:
                detail += f" | first pass failed, recomputed on 1 thread; first pass: {failed[name]}"
            checks.append((name, ok, detail))
    return checks, fig["csv"]


def month_end_after(d):
    y, m = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def cross_checks(con, entries, outdir):
    """Checks across all verified parquets. Returns [(check, passed or None for n/a, detail)]."""
    paths = [str(outdir / e["parquet"]) for e in entries]
    src = f"read_parquet([{', '.join(map(sql_str, paths))}], filename = true)"
    checks = []

    pq_rows = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    csv_rows = sum(e["rows"] for e in entries)
    checks.append(("total rows: parquets = CSVs", pq_rows == csv_rows,
                   f"parquets {pq_rows:,}, CSVs {csv_rows:,} ({len(entries)} files)"))

    shared = con.execute(f"SELECT count(*) FROM (SELECT id, eom FROM {src} GROUP BY id, eom "
                         "HAVING count(DISTINCT filename) > 1)").fetchone()[0]
    checks.append(("no (id, eom) in two files", shared == 0, f"{shared:,} pairs in more than one file"))

    by_month = con.execute(f"SELECT eom, list(DISTINCT filename), count(*) FROM {src} GROUP BY eom ORDER BY eom").fetchall()
    missing_eom = sum(n for eom, _, n in by_month if eom is None)
    by_month = [(eom, files) for eom, files, _ in by_month if eom is not None]
    not_month_end = [eom for eom, _ in by_month if eom != dt.date(eom.year, eom.month, calendar.monthrange(eom.year, eom.month)[1])]
    in_two = [eom for eom, files in by_month if len(files) > 1]
    checks.append(("eom never missing, always a month end", missing_eom == 0 and not not_month_end,
                   f"missing {missing_eom:,}, not month end {len(not_month_end)}" + (f", e.g. {not_month_end[:3]}" if not_month_end else "")))

    ranges = sorted((e["eom_min"], e["eom_max"], e["parquet"]) for e in entries)
    joins = []
    for (_, prev_max, prev_name), (next_min, _, next_name) in zip(ranges, ranges[1:]):
        if dt.date.fromisoformat(next_min) != month_end_after(dt.date.fromisoformat(prev_max)):
            joins.append(f"{prev_name} ends {prev_max}, {next_name} starts {next_min}")
    checks.append(("date ranges join, no gaps or overlaps", not joins and not in_two,
                   " | ".join(f"{n}: {lo} to {hi}" for lo, hi, n in ranges)
                   + (f"; months in two files: {in_two[:5]}" if in_two else "") + (f"; {joins}" if joins else "")))

    months = [eom for eom, _ in by_month]
    expected, d = [], months[0] if months else None
    while d is not None and d <= months[-1]:
        expected.append(d)
        d = month_end_after(d)
    gaps = sorted(set(expected) - set(months))
    checks.append(("every month present", not gaps and bool(months),
                   f"{len(months)} months, {months[0] if months else '-'} to {months[-1] if months else '-'}"
                   + (f"; missing {len(gaps)}, e.g. {gaps[:5]}" if gaps else "")))

    if not months or not months[0] <= APPLE_EOM <= months[-1]:
        checks.append(("spot check: Apple (id 14593) me at 2024-12-31", None, "n/a: 2024-12-31 not in the data"))
    else:
        rows = con.execute(f"SELECT me FROM {src} WHERE id = {APPLE_ID} AND eom = DATE '{APPLE_EOM}'").fetchall()
        ok = len(rows) == 1 and rows[0][0] is not None and APPLE_ME[0] <= rows[0][0] <= APPLE_ME[1]
        checks.append(("spot check: Apple (id 14593) me at 2024-12-31", ok,
                       f"{len(rows)} row(s), me = {', '.join(f'{r[0]:,.0f}' for r in rows) or '-'} "
                       f"(expected {APPLE_ME[0]:,.0f} to {APPLE_ME[1]:,.0f})"))
    return checks


def mark(ok):
    return "n/a " if ok is None else "PASS" if ok else "FAIL"


def write_report(outdir, state, cross):
    lines = [f"Raw CSV -> parquet verification, {dt.datetime.now():%Y-%m-%d %H:%M}", ""]
    for e in state["files"].values():
        status = "all checks passed" if e["passed"] else "FAILED"
        lines += [f"{e['csv']} -> {e['parquet']}: {status}",
                  f"  rows {e['rows']:,} | columns {e['columns']} | eom {e['eom_min']} to {e['eom_max']}",
                  f"  CSV {e['csv_bytes'] / 1e9:,.2f} GB | parquet "
                  + (f"{e['parquet_bytes'] / 1e9:,.2f} GB ({e['parquet_bytes'] / e['csv_bytes']:.1%} of CSV)"
                     if e.get("parquet_bytes") else "-")
                  + f" | CSV deleted: {'yes' if e.get('csv_deleted') else 'no'} | {e['seconds']:,.0f}s"]
        lines += [f"  [{mark(ok)}] {name}: {detail}" for name, ok, detail in e["checks"]]
        lines.append("")
    if cross is not None:
        lines.append(f"Cross-file checks ({len([e for e in state['files'].values() if e['passed']])} verified files)")
        lines += [f"  [{mark(ok)}] {name}: {detail}" for name, ok, detail in cross]
    text = "\n".join(lines) + "\n"
    tmp = outdir / "verification_report.txt.tmp"
    tmp.write_text(text)
    tmp.replace(outdir / "verification_report.txt")
    print("\n" + text)


def save_state(outdir, state):
    tmp = outdir / "verification_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(outdir / "verification_state.json")


def process(con, f, outdir, delete, state):
    """Convert and verify one CSV. Returns True if every check passed."""
    f = Path(f)
    final, tmp = outdir / f"{f.stem}.parquet", outdir / f"{f.stem}.parquet.tmp"
    csv_bytes = f.stat().st_size
    need, free = 1.3 * PARQUET_RATIO * csv_bytes + HEADROOM, shutil.disk_usage(outdir).free
    if free < need:
        print(f"Space guard: {f.name} needs {need / 1e9:,.1f} GB free, only {free / 1e9:,.1f} GB. Stopping.")
        return False

    t0 = time.time()
    cols = csv_header(f)
    print(f"\n{f.name}: {csv_bytes / 1e9:,.2f} GB, {len(cols)} columns. Converting...", flush=True)
    con.execute(f"COPY ({csv_source(str(f), cols)}) TO {sql_str(tmp)} (FORMAT parquet, COMPRESSION zstd)")
    print(f"  written in {time.time() - t0:,.0f}s ({tmp.stat().st_size / 1e9:,.2f} GB). Verifying...", flush=True)

    checks = [("unique column names in the CSV", len(set(cols)) == len(cols), f"{len(cols)} columns")]
    more, a = verify(con, str(f), str(tmp), cols)
    checks += more
    passed = all(ok for _, ok, _ in checks)
    entry = {"csv": f.name, "parquet": final.name, "csv_bytes": csv_bytes, "rows": a["rows"], "columns": len(cols),
             "eom_min": a["eom|min"], "eom_max": a["eom|max"], "passed": passed, "csv_deleted": False,
             "checks": checks, "seconds": time.time() - t0}
    if passed:
        tmp.replace(final)
        entry["parquet_bytes"] = final.stat().st_size
    state["files"][f.name] = entry
    save_state(outdir, state)
    for name, ok, detail in checks:
        print(f"  [{mark(ok)}] {name}: {detail}")
    if not passed:
        print(f"  Check failed: kept {f.name} and {tmp.name}. Stopping.")
        return False
    if delete:
        f.unlink()
        entry["csv_deleted"] = True
        save_state(outdir, state)
        print(f"  Deleted {f.name}.")
    return True


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--pyarrow-figures":
        return pyarrow_figures(sys.argv[2], "--single-thread" in sys.argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="CSV files; already verified ones are skipped")
    parser.add_argument("--outdir", default=str(ROOT / "data" / "raw_parquet"))
    parser.add_argument("--delete-csv", action="store_true", help="delete each CSV after its parquet passes every check")
    args = parser.parse_args()

    import duckdb

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state_file = outdir / "verification_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"files": {}}

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '6GB'; SET temp_directory = {sql_str(TMP_DIR)}; "
                "SET preserve_insertion_order = true")

    todo = []
    for f in args.inputs:
        done = state["files"].get(Path(f).name)
        if done and done["passed"] and (outdir / done["parquet"]).exists():
            print(f"{Path(f).name}: already verified, skipped.")
        elif not f.endswith(".csv") or not Path(f).exists():
            sys.exit(f"Not a CSV file: {f}")
        else:
            todo.append(f)
    ok = True
    for f in sorted(todo, key=lambda p: Path(p).stat().st_size):
        if not process(con, f, outdir, args.delete_csv, state):
            ok = False
            break

    verified = [e for e in state["files"].values() if e["passed"]]
    cross = cross_checks(con, verified, outdir) if ok and verified else None
    write_report(outdir, state, cross)
    con.close()
    if TMP_DIR.exists() and not any(TMP_DIR.iterdir()):
        TMP_DIR.rmdir()
    if not ok or (cross and any(c[1] is False for c in cross)):
        sys.exit(1)


if __name__ == "__main__":
    main()
