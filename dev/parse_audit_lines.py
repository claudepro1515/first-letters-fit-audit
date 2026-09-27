"""Turn BLOCK and RESULT lines copied from an audit notebook's log into data rows (dev tool).

Usage: python parse_audit_lines.py <log text file> <scroll> <z0> <z1> <notebook name> [--block-out F] [--result-out F]
The BLOCK line: 'BLOCK <label>: windings [a, b]; ... (radial rays) {fit} -> after phase_align {fit}; random-shift null
{null} -> {null}; gaps {gaps} -> {gaps}; surface in voids x -> y; N min'.
"""
import ast
import json
import re
import sys


def dicts(s):
    """The Python dict / None literals of a line, in order."""
    out, i = [], 0
    while i < len(s):
        if s.startswith("None", i):
            out.append(None)
            i += 4
            continue
        if s[i] == "{":
            depth = 0
            for j in range(i, len(s)):
                depth += s[j] == "{"
                depth -= s[j] == "}"
                if depth == 0:
                    out.append(ast.literal_eval(s[i:j + 1]))
                    i = j + 1
                    break
            continue
        i += 1
    return out


def main():
    src, scroll, z0, z1, nb = sys.argv[1:6]
    bo = sys.argv[sys.argv.index("--block-out") + 1] if "--block-out" in sys.argv else None
    text = open(src).read()
    rows = []
    for line in text.splitlines():
        m = re.search(r"BLOCK (\S+): windings \[(\d+), (\d+)\]", line)
        if not m:
            continue
        d = dicts(line[m.end():])
        voids = re.search(r"surface in voids ([0-9.]+|None) -> ([0-9.]+|None)", line)
        row = {"scroll": scroll, "z": [int(z0), int(z1)], "fit": m.group(1), "notebook": nb,
               "windings": [int(m.group(2)), int(m.group(3))], "period_vox": 17.0,
               "raw": d[0], "aligned": d[1]}
        if len(d) >= 6:  # with the random-shift null
            row["null_raw"], row["null_aligned"], row["gaps_raw"], row["gaps_aligned"] = d[2], d[3], d[4], d[5]
        else:
            row["gaps_raw"], row["gaps_aligned"] = d[2], d[3]
        if voids:
            row["void_raw"] = None if voids.group(1) == "None" else float(voids.group(1))
            row["void_aligned"] = None if voids.group(2) == "None" else float(voids.group(2))
        rows.append(row)
    for r in rows:
        print(json.dumps(r))
    if bo:
        with open(bo, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
