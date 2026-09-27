"""How much of a fit's surface lies in empty space? A model-free check on the raw CT.

For every winding mesh of a fit (every --stride-th vertex with a normal), the CT is sampled along the normal within
+-R voxels (default 6) of the surface. A vertex is "in a void" when no sample reaches the papyrus threshold: the
Otsu threshold of the scan's own intensities inside the scroll (sampled at random in the band; masked volumes are 0
outside the scroll and those voxels are left out). A surface on a sheet, or off it by less than R, is not in a void;
a winding that runs through the empty space between sheet bundles is. Curated windings score about 0.

Usage: python surface_void.py <dir of wNNN meshes> <volume level 0 (URL or local zarr)> --z Z0 Z1 [--reach 6]
       [--stride 2] [--pattern 'w[0-9][0-9][0-9]'] [--cache DIR] [--label NAME]
Prints one JSON line: void fraction overall, by winding, and the threshold.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_align import Sampler, load_mesh, normals, open_level0  # noqa: E402


def otsu(values, bins=256):
    h, edges = np.histogram(values, bins=bins, range=(0, 256))
    h = h.astype(float)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1e-9)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(c[int(np.argmax(between[:-1]))])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh_dir")
    ap.add_argument("volume")
    ap.add_argument("--z", type=float, nargs=2, required=True)
    ap.add_argument("--reach", type=float, default=6.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--pattern", default="w[0-9][0-9][0-9]")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    smp = Sampler(open_level0(a.volume, a.cache))
    dirs = sorted(d for d in glob.glob(os.path.join(a.mesh_dir, a.pattern)) if os.path.exists(os.path.join(d, "x.tif")))
    # threshold: Otsu on CT samples inside the scroll, taken at the fit's own vertices +- 20 voxels along the normal
    # (the part of the scan the fit lives in)
    rng = np.random.default_rng(a.seed)
    probes, per = [], {}
    offs = np.arange(-a.reach, a.reach + 1e-6, 1.0)
    total_void, total_n = 0, 0
    samples_for_thr = []
    meshes = []
    for d in dirs:
        P, v = load_mesh(d)
        n, ok = normals(P, v)
        rr, cc = np.nonzero(ok)
        keep = (rr % a.stride == 0) & (cc % a.stride == 0)
        rr, cc = rr[keep], cc[keep]
        z = P[rr, cc, 2]
        inb = (z >= a.z[0]) & (z < a.z[1])
        rr, cc = rr[inb], cc[inb]
        if not len(rr):
            continue
        meshes.append((os.path.basename(d), P[rr, cc], n[rr, cc]))
        j = rng.choice(len(rr), min(200, len(rr)), replace=False)
        t = rng.uniform(-20, 20, len(j))
        pts = P[rr[j], cc[j]] + t[:, None] * n[rr[j], cc[j]]
        samples_for_thr.append(smp.sample(pts[:, ::-1]))
        smp.trim(300)
    if not meshes:
        print(json.dumps({"label": a.label, "error": "no vertices in the band"}))
        return
    allv = np.concatenate(samples_for_thr)
    thr = otsu(allv[allv > 0])
    for name, base, nrm in meshes:
        vmax = np.zeros(len(base), np.float32)
        inside = np.ones(len(base), bool)
        for k in range(0, len(base), 2000):
            pts = base[k:k + 2000, None, :] + offs[None, :, None] * nrm[k:k + 2000, None, :]
            vals = smp.sample(pts.reshape(-1, 3)[:, ::-1]).reshape(-1, len(offs))
            vmax[k:k + 2000] = vals.max(1)
            inside[k:k + 2000] = (vals > 0).all(1)  # all samples inside the scan mask
            smp.trim(300)
        void = (vmax < thr) & inside
        per[name] = round(float(void[inside].mean()), 4) if inside.any() else None
        total_void += int(void.sum())
        total_n += int(inside.sum())
    print(json.dumps({"label": a.label or os.path.basename(a.mesh_dir.rstrip("/")), "threshold": round(thr, 1),
                      "reach_vox": a.reach, "vertices": total_n, "void_frac": round(total_void / max(total_n, 1), 4),
                      "windings": len(per), "void_frac_by_winding": per}), flush=True)


if __name__ == "__main__":
    main()
