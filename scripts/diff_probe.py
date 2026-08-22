#!/usr/bin/env python3
"""Differential probe runner: compares ahiru vs duckdb over a list of queries.

Usage: python3 scripts/diff_probe.py queries.txt
Each line in the file: <file1>,<file2>,...|<SQL>   (files relative to repo root)
Lines starting with # are skipped.
"""
import csv
import io
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keep this distinct from ordinary SQL text. An empty CSV field is an empty
# string, not NULL; both CLIs can spell NULL with this marker.
NULL_TOKEN = "__AHIRU_DIFF_NULL__"

def duck_source(path):
    p = os.path.join(REPO, path)
    if path.endswith(".csv"): return "read_csv_auto('%s')" % p
    if path.endswith(".jsonl"): return "read_json_auto('%s')" % p
    if path.endswith(".json") and not path.endswith(".jsonl"): return "read_json_auto('%s')" % p
    return "'%s'" % p

def replace_tables(sql, files):
    names = {("t" if i == 0 else "t%d" % (i + 1)): f for i, f in enumerate(files)}
    def repl(m):
        w = m.group(0)
        return duck_source(names[w]) if w in names else w
    # word-boundary replace; \b handles alnum+underscore since names are t/t2..
    return re.sub(r"\bt\d*\b", repl, sql)

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return p.returncode, p.stdout, p.stderr

def normalize(text):
    """Parse CSV while preserving empty cells, row width, and string text.

    The old line/regex parser called ``strip()`` on every line and mapped an
    empty field to ``<null>``. That dropped trailing empty columns and made
    empty strings compare equal to SQL NULL. ``csv.reader`` keeps quoted
    commas/newlines and trailing empty fields intact; NULL is identified only
    by the explicit marker configured in ``main``.
    """
    rows = []
    reader = csv.reader(io.StringIO(text, newline=""))
    for cells in reader:
        # A blank CSV record has no meaning here. A record of empty fields
        # (",,,") is meaningful and is not filtered.
        if not cells:
            continue
        normalized = []
        for cell in cells:
            if cell == NULL_TOKEN:
                normalized.append("<null>")
                continue
            try:
                f = float(cell)
                if f != f or f in (float("inf"), float("-inf")):
                    normalized.append(cell)
                elif f == int(f) and abs(f) < 9e15:
                    normalized.append(str(int(f)))
                else:
                    normalized.append("%.9f" % f)
            except (ValueError, OverflowError):
                # Do not strip strings: surrounding/trailing whitespace is a
                # part of a SQL VARCHAR value.
                normalized.append(cell)
        rows.append(tuple(normalized))
    return rows[1:]  # drop the CSV header

def main():
    qfile = sys.argv[1]
    cases = []
    with open(qfile) as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            files_part, sql = line.split("|", 1)
            files = [f for f in files_part.split(",") if f]
            cases.append((ln, files, sql))

    bad = 0
    for ln, files, sql in cases:
        rc_a, out_a, err_a = run(["cargo", "run", "-q", "-p", "ahiru-cli", "--",
                                  "-no-init", "-csv", "-nullvalue", NULL_TOKEN,
                                  "query"] + files + [sql])
        dsql = replace_tables(sql, files)
        rc_d, out_d, err_d = run(["duckdb", "-csv", "-nullvalue", NULL_TOKEN, "-c", dsql])
        if rc_a != 0:
            print("FAIL[line %d] ahiru error: %s\n  SQL: %s\n  %s" % (ln, err_a.strip()[:200], sql, err_a.strip().splitlines()[-1][:160] if err_a.strip() else ""))
            bad += 1
            continue
        if rc_d != 0:
            print("SKIP[line %d] duckdb rejects (not a bug): %s | %s" % (ln, sql[:80], err_d.strip().splitlines()[0][:100] if err_d else ""))
            continue
        na, nd = normalize(out_a), normalize(out_d)
        if sorted(na) != sorted(nd) and na != nd:
            print("MISMATCH[line %d]: %s\n  files=%s\n  ahiru : %s\n  duckdb: %s" % (ln, sql, files, na[:6], nd[:6]))
            bad += 1
    print("\n%d/%d mismatched" % (bad, len(cases)))

if __name__ == "__main__":
    main()
