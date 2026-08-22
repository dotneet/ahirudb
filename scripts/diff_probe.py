#!/usr/bin/env python3
"""Differential probe runner: compares ahiru vs duckdb over a list of queries.

Usage: python3 scripts/diff_probe.py queries.txt
Each line in the file: <file1>,<file2>,...|<SQL>   (files relative to repo root)
Lines starting with # are skipped.
"""
import os, re, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.endswith("rows)") or line.endswith("row)"):
            continue
        cells = []
        for c in re.split(r"\t|,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line):
            c = c.strip().strip('"')
            if c in ("", "NULL", "\\N"): cells.append("<null>")
            else:
                try:
                    f = float(c)
                    if f != f or f in (float("inf"), float("-inf")):
                        cells.append(c)
                    elif f == int(f) and abs(f) < 9e15:
                        cells.append(str(int(f)))
                    else:
                        cells.append("%.9f" % f)
                except (ValueError, OverflowError):
                    cells.append(c)
        rows.append(tuple(cells))
    return rows[1:]  # drop header

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
                                  "query"] + files + [sql])
        dsql = replace_tables(sql, files)
        rc_d, out_d, err_d = run(["duckdb", "-csv", "-c", dsql])
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
