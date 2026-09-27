"""The model-free test on PHerc Paris 4: the team's own spiral fit (published with its ink detections) and eight
consecutive staff windings, on level 2 (9.6 um) of the 2.4 um scan.

Downloads the meshes (tifxyz; the fit's intermediate `tifxyz_normalized` meshes are already in level-2 voxels, the staff
meshes' 2.4 um tifxyz are divided by 4), divides the umbilicus by 4, and runs tools/sheet_hits.py with 150 radial rays
and its random-shift null on each band, with the volume opened at level 2 instead of level 0; then the fit's w028-w058 on
z 9000-10000 again, as fitted (cut to the band) and after tools/phase_align.py. One JSON line per run
(data/sheet_hits_paris4_team_fit_and_staff.jsonl holds ours).

  python paris4_check.py <work dir>
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = next(p for p in (os.path.join(os.path.dirname(HERE), "tools"), os.path.join(os.path.dirname(HERE), "release", "tools"))
             if os.path.isdir(p))
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4"
VOL = S3 + "/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2"
UMB = S3 + "/representations/umbilicus/20260411134726-umbilicus-20260524235033.json"
FIT = {"w010": "20260701183124-w010-027", "w028": "20260701183125-w028-037", "w038": "20260701183126-w038-045",
       "w046": "20260701183127-w046-052", "w053": "20260701183128-w053-058"}
STAFF = {"w003": "20260602204401-5753_-7", "w004": "20260603185441-5753_-6", "w005": "20260603190005-5753_-5",
         "w006": "20260603145540-5753_-4", "w007": "20260603042357-5753_-3", "w008": "20260603024952-5753_-2",
         "w009": "20260603005223-5753_-1", "w010": "20260602225659-5753_0"}
RUNS = [("fit_w028-058", ["w028", "w038", "w046", "w053"], (5000, 6000), 300, 1800),
        ("fit_w028-058", ["w028", "w038", "w046", "w053"], (9000, 10000), 300, 1800),
        ("fit_w028-058", ["w028", "w038", "w046", "w053"], (13000, 14000), 300, 1800),
        ("staff_5753", list(STAFF), (9000, 10000), 100, 800), ("staff_5753", list(STAFF), (5000, 6000), 100, 800),
        ("fit_w010-027", ["w010"], (9000, 10000), 100, 800), ("fit_w010-027", ["w010"], (5000, 6000), 100, 800)]

# sheet_hits.py (or phase_align.py) with the volume opened at the given level (they open level 0 of an OME-Zarr)
RUNNER = """import sys
sys.path.insert(0, {tools!r})
import {module} as m
from vcz import ZArray
url = sys.argv[1]
m.open_level0 = lambda spec, cache_dir=None: ZArray(url, cache_dir=cache_dir)
sys.argv = [sys.argv[0]] + sys.argv[2:]
m.main()
"""
ALIGN_BAND, ALIGN_PERIOD = (9000, 10000), 13.0  # the fit's w028-w058 cut to this band (+-100), aligned, then scored again


def get(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["curl", "-sSf", "-o", path, url], check=True)


def main():
    work = sys.argv[1]
    os.makedirs(work, exist_ok=True)
    for w, seg in FIT.items():
        for c in "xyz":
            get(f"{S3}/segments/{seg}/mesh/intermediate/tifxyz_normalized/{c}.tif", f"{work}/fit/{w}/{c}.tif")
    for w, seg in STAFF.items():
        ts = seg.split("-")[0]
        for c in "xyz":
            raw = f"{work}/staff_2p4/{w}/{c}.tif"
            get(f"{S3}/segments/{seg}/mesh/{ts}-on-20260411134726-2.4um.tifxyz/{c}.tif", raw)
            os.makedirs(f"{work}/staff/{w}", exist_ok=True)
            a = np.array(Image.open(raw), np.float32)
            Image.fromarray(np.where(a > 0, a / 4.0, -1.0).astype(np.float32)).save(f"{work}/staff/{w}/{c}.tif")
    get(UMB, f"{work}/umbilicus_2p4.json")
    cp = json.load(open(f"{work}/umbilicus_2p4.json"))["control_points"]
    json.dump({"control_points": [{k: q[k] / 4.0 for k in ("z", "y", "x")} for q in cp]}, open(f"{work}/umbilicus_l2.json", "w"))
    open(f"{work}/run_level.py", "w").write(RUNNER.format(tools=TOOLS, module="sheet_hits"))
    open(f"{work}/align_level.py", "w").write(RUNNER.format(tools=TOOLS, module="phase_align"))
    for label, ws, (z0, z1), r0, r1 in RUNS:
        d = f"{work}/set_{label}"
        os.makedirs(d, exist_ok=True)
        src = "staff" if label.startswith("staff") else "fit"
        for w in ws:
            if not os.path.lexists(f"{d}/{w}"):
                os.symlink(os.path.abspath(f"{work}/{src}/{w}"), f"{d}/{w}")
        out = subprocess.run([sys.executable, f"{work}/run_level.py", VOL, d, VOL, "--z", str(z0), str(z1), "--radial", "150",
                              "--r0", str(r0), "--r1", str(r1), "--umbilicus", f"{work}/umbilicus_l2.json", "--null", "200",
                              "--label", f"{label}_z{z0}"], capture_output=True, text=True).stdout
        print(out.strip().splitlines()[-1], flush=True)
    # the fit's w028-w058 on one band, as fitted and moved onto the sheets by tools/phase_align.py (crest mode)
    z0, z1 = ALIGN_BAND
    for w in ["w028", "w038", "w046", "w053"]:
        arr = {c: np.array(Image.open(f"{work}/fit/{w}/{c}.tif"), np.float32) for c in "xyz"}
        ok = (arr["z"] > 0) & (arr["z"] >= z0 - 100) & (arr["z"] <= z1 + 100)
        rows = np.flatnonzero(ok.any(1))
        os.makedirs(f"{work}/crop/{w}", exist_ok=True)
        for c in "xyz":
            Image.fromarray(np.where(ok, arr[c], -1.0)[rows.min():rows.max() + 1].astype(np.float32)).save(f"{work}/crop/{w}/{c}.tif")
    subprocess.run([sys.executable, f"{work}/align_level.py", VOL, f"{work}/crop", VOL, f"{work}/aligned", "--period",
                    str(ALIGN_PERIOD), "--pattern", "w[0-9][0-9][0-9]"], check=True, capture_output=True)
    for tag in ("crop", "aligned"):
        out = subprocess.run([sys.executable, f"{work}/run_level.py", VOL, f"{work}/{tag}", VOL, "--z", str(z0), str(z1),
                              "--radial", "150", "--r0", "300", "--r1", "1800", "--umbilicus", f"{work}/umbilicus_l2.json",
                              "--null", "200", "--label", f"fit_w028-058_z{z0}_{tag}"], capture_output=True, text=True).stdout
        print(out.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
