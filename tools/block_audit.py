"""Audit a block of consecutive windings of a spiral fit against the raw CT, as the audit notebooks do, on one machine.

For a block of windings around the middle of the fit (16 by default): the model-free test of tools/sheet_hits.py on
radial rays (is the CT brighter where a winding crosses a ray than half-way to its neighbours?) with its random-shift
null (the same windings at a random position along each ray), the gaps between consecutive crossings (two windings on
one sheet), and the share of the surface with no papyrus within 6 voxels (tools/surface_void.py), each as fitted and
after tools/phase_align.py has moved every winding onto the sheet it runs along. Prints one JSON line.

Usage: python block_audit.py <dir of wNNN tifxyz windings> <volume level 0 (URL or local zarr)> <work dir>
       --z Z0 Z1 --umbilicus U.json [--windings 16] [--period 17] [--radial 80] [--r0 300] [--r1 2200] [--null 200]
       [--procs 4] [--cache DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def last_json(out):
    return next((json.loads(line) for line in reversed(out.splitlines()) if line.startswith("{")), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes")
    ap.add_argument("volume")
    ap.add_argument("work")
    ap.add_argument("--z", type=float, nargs=2, required=True)
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--windings", type=int, default=16)
    ap.add_argument("--period", type=float, default=17.0, help="about the local sheet spacing (voxels)")
    ap.add_argument("--radial", type=int, default=80)
    ap.add_argument("--r0", type=float, default=300.0)
    ap.add_argument("--r1", type=float, default=2200.0)
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--procs", type=int, default=4, help="parallel phase_align processes (they share the CT cache)")
    ap.add_argument("--cache", default=None, help="directory for downloaded CT chunks (default <work>/ct_cache)")
    a = ap.parse_args()
    cache = a.cache or os.path.join(a.work, "ct_cache")
    ws = sorted(int(os.path.basename(d)[1:4]) for d in glob.glob(os.path.join(a.meshes, "w[0-9][0-9][0-9]"))
                if os.path.exists(os.path.join(d, "x.tif")))
    if not ws:
        sys.exit(f"no wNNN tifxyz windings in {a.meshes}")
    mid = len(ws) // 2
    start = max(0, mid - a.windings // 2)
    sel = ws[start:start + a.windings]
    raw, aligned = os.path.join(a.work, "raw"), os.path.join(a.work, "aligned")
    os.makedirs(raw, exist_ok=True)
    procs = []
    for k in range(max(1, a.procs)):
        part = os.path.join(a.work, f"part{k}")
        os.makedirs(part, exist_ok=True)
        for w in sel[k::max(1, a.procs)]:
            src = os.path.abspath(os.path.join(a.meshes, f"w{w:03d}"))
            for dst in (os.path.join(raw, f"w{w:03d}"), os.path.join(part, f"w{w:03d}")):
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
        if os.listdir(part):
            procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, "phase_align.py"), part, a.volume, aligned,
                                           "--period", str(a.period), "--cache", cache],
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    align_log = [p.communicate()[0] for p in procs]
    res = {"meshes": a.meshes, "block_windings": [sel[0], sel[-1]], "windings": len(sel), "period_vox": a.period,
           "phase_align_failed": [log[-400:] for log, p in zip(align_log, procs) if p.returncode != 0],
           "aligned_missing": [w for w in sel if not os.path.exists(os.path.join(aligned, f"w{w:03d}", "x.tif"))]}
    for tag, d in (("raw", raw), ("aligned", aligned)):
        out = subprocess.run([sys.executable, os.path.join(HERE, "sheet_hits.py"), d, a.volume, "--z", str(a.z[0]),
                              str(a.z[1]), "--radial", str(a.radial), "--r0", str(a.r0), "--r1", str(a.r1),
                              "--umbilicus", a.umbilicus, "--null", str(a.null), "--cache", cache, "--label", tag],
                             capture_output=True, text=True).stdout
        sh = last_json(out) or {}
        res[tag] = {"rays": sh.get("rays"), "fit": sh.get("fit"), "null_random_phase": sh.get("null_random_phase"),
                    "gaps_vox": sh.get("gaps_vox")}
        out = subprocess.run([sys.executable, os.path.join(HERE, "surface_void.py"), d, a.volume, "--z", str(a.z[0]),
                              str(a.z[1]), "--cache", cache, "--label", tag], capture_output=True, text=True).stdout
        res[tag]["void_frac"] = (last_json(out) or {}).get("void_frac")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
