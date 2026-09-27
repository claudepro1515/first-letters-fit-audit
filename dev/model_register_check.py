"""Where does the winding model put its crossings when a ray is registered on a sheet, in a gap, or at its centre?
(dev check on the 15 saved rays of wm/dev_model_brightness.py --save, PHerc0826 z 9328; CPU, no model run needed)

villa's exporter registers each ray's phase at its anchor (the seed point) and keeps the integer passages of the
registered phase. Here the same phase of each ray is registered at three anchors near the ray's centre: the centre
itself, the brightest point of the smoothed CT profile within +-8 voxels of it (a sheet), and the darkest (a gap). The
model-free contrast of sheet_hits.py is then scored at the other crossings (the one at the anchor is left out).
Also: every crossing moved together by -8..+8 voxels (the offset scan of dev_model_brightness.py).

Usage: VILLA=<villa checkout> python model_register_check.py <model_rays.npz> [--reach 8]
       (data/model_rays_PHerc0826_z9328.npz holds the 15 rays; decode_center_ray is read from villa's source)
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(os.path.dirname(HERE), "tools"), os.path.join(os.path.dirname(HERE), "release", "tools")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
from sheet_hits import contrast_at  # noqa: E402

VILLA = os.environ.get("VILLA", os.path.join(os.path.dirname(HERE), "villa"))  # a checkout of ScrollPrize/villa
ESS = os.path.join(VILLA, "vesuvius", "src", "vesuvius", "neural_tracing", "winding_models", "export_spiral_supervision.py")


def _load_decode(path):
    """villa's decode_center_ray, taken verbatim from its source file (the package import needs the GPU stack)."""
    src = open(path).read()
    ns = {"np": np}
    for name in ("DEFAULT_EDGE_MARGIN", "DEFAULT_EDGE_TRIM"):
        m = re.search(rf"^{name}\s*=\s*(.+)$", src, re.M)
        ns[name] = eval(m.group(1), ns)
    start = src.index("def decode_center_ray(")
    end = src.find("\ndef ", start + 10)
    exec("from __future__ import annotations\n" + src[start:end if end > 0 else None], ns)
    return ns["decode_center_ray"]


decode_center_ray = _load_decode(ESS)

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--reach", type=float, default=8.0)
a = ap.parse_args()
d = np.load(a.npz)
sp = float(d["sp"])
n = len([k for k in d.files if k.startswith("prof")])
res = {"rays": n}
same = 0
scores = {"centre": [], "bright": [], "dark": []}
for k in range(n):
    prof, inside, ph, pos0 = d[f"prof{k}"], d[f"inside{k}"].astype(bool), d[f"phase{k}"], d[f"pos{k}"]
    L = len(ph)
    ts = np.arange(L) * sp
    c = int(round((L - 1) / 2))
    win = np.arange(max(0, c - int(a.reach / sp)), min(L, c + int(a.reach / sp) + 1))
    win = win[inside[win]]
    if not len(win):
        continue
    anchors = {"centre": c, "bright": int(win[np.argmax(prof[win])]), "dark": int(win[np.argmin(prof[win])])}
    for name, an in anchors.items():
        pos, _ = decode_center_ray(ph, inside, anchor=an)
        t = np.sort(np.asarray(pos, float) * sp)
        if name == "centre":
            same += int(len(t) == len(pos0) and np.allclose(t, np.sort(pos0), atol=1e-3))
        keep = np.abs(t - an * sp) > 1.0  # leave out the crossing at the anchor itself
        cc = contrast_at(prof, inside, ts, t)
        if len(cc) == len(t):
            cc = cc[keep]
        scores[name].append(cc)
for name, v in scores.items():
    v = np.concatenate(v) if v else np.zeros(0)
    p = float((v > 0).mean()) if len(v) else float("nan")
    res[name] = {"n": int(len(v)), "frac_c_pos": round(p, 3), "se": round(float(np.sqrt(p * (1 - p) / max(len(v), 1))), 3)}
res["centre_decode_reproduces_saved_crossings"] = f"{same}/{n} rays"
scan = {}
for delta in np.arange(-8, 8.5, 1.0):
    cc_ = [contrast_at(d[f"prof{k}"], d[f"inside{k}"].astype(bool), np.arange(len(d[f"prof{k}"])) * sp, np.sort(d[f"pos{k}"]) + delta)
           for k in range(n)]
    cc_ = np.concatenate(cc_)
    scan[f"{delta:+.0f}"] = round(float((cc_ > 0).mean()), 3)
res["offset_scan_frac_c_pos"] = scan
print(json.dumps(res))
