"""How many fitted windings lie between consecutive sheet crossings predicted by the winding model?

Input: a spiral fit's exported winding meshes (tifxyz directories wNNN, as fit_spiral writes them) and a
compact winding-inference crossing store (villa's export_spiral_supervision.py output: rays with the
model's integer-phase crossings). Each store ray is intersected with every winding mesh (two triangles per
grid quad, Moller-Trumbore); for each pair of model crossings k and k + d on a ray, the number of fitted
windings the ray passes between them is compared with d, the number of sheets the model counts there.
A fit whose windings follow the sheets passes exactly one winding per sheet (count == d); a fit with two
windings per sheet, or windings that cut across sheets, does not. With d = 1 this is the quantity
fit_spiral's winding-model density loss drives to 1, so on rays not used for fitting it is a held-out check.

Usage:
  python winding_agreement.py <mesh_dir> <store_dir> [--z Z0 Z1] [--max-rays N] [--umbilicus umbilicus.json]
Prints one JSON line: pairs, fraction with exactly d windings for d = 1 and d = 5, the distribution of the
count for d = 1, and (with --umbilicus) the d = 1 fraction by radius band.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def load_store(root):
    man = json.load(open(os.path.join(root, "manifest.json")))
    parts = {k: [] for k in ("ray_origin_zyx", "ray_step_zyx", "crossing_t", "crossing_level")}
    offsets, base = [np.zeros(1, np.int64)], 0
    for sh in man["shards"]:
        for k in parts:
            parts[k].append(np.load(os.path.join(root, sh["name"], sh["arrays"][k]["file"])))
        o = np.load(os.path.join(root, sh["name"], sh["arrays"]["crossing_offsets"]["file"]))
        offsets.append(o[1:] + base)
        base += int(o[-1])
    out = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
    out["offsets"] = np.concatenate(offsets)
    return out


def load_triangles(mesh_dir, zlo, zhi):
    """Triangles (zyx) of every unspliced winding mesh wNNN within [zlo, zhi] (+ one grid step)."""
    tris, labels = [], []
    for d in sorted(glob.glob(os.path.join(mesh_dir, "w[0-9][0-9][0-9]"))):
        w = int(re.search(r"w(\d+)$", d).group(1))
        x, y, z = (np.array(Image.open(os.path.join(d, c + ".tif")), np.float64) for c in "xyz")
        ok = (z > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x >= 0) & (y >= 0)
        rows = np.flatnonzero(((z >= zlo - 40) & (z <= zhi + 40) & ok).any(1))
        if len(rows) == 0:
            continue
        r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, z.shape[0])
        V = np.stack([z, y, x], -1)[r0:r1]
        m = ok[r0:r1]
        a, b, c, e = V[:-1, :-1], V[1:, :-1], V[:-1, 1:], V[1:, 1:]
        ma, mb, mc, me = m[:-1, :-1], m[1:, :-1], m[:-1, 1:], m[1:, 1:]
        k1 = ma & mb & mc
        k2 = mb & me & mc
        t1 = np.stack([a[k1], b[k1], c[k1]], 1)
        t2 = np.stack([b[k2], e[k2], c[k2]], 1)
        t = np.concatenate([t1, t2])
        tris.append(t)
        labels.append(np.full(len(t), w, np.int32))
    if not tris:
        return np.zeros((0, 3, 3)), np.zeros(0, np.int32)
    return np.concatenate(tris), np.concatenate(labels)


class TriIndex:
    """Candidate triangles near a ray. Triangles are indexed by their centres; the few very large ones (spliced
    seams, clipped rows: up to ~500 voxels across in real fits) are always tested, so the search radius stays
    small (a single large triangle used to make every ray test millions of triangles)."""

    def __init__(self, tri, big=64.0):
        cen = tri.mean(1)
        rad = np.sqrt(((tri - cen[:, None]) ** 2).sum(-1)).max(1) if len(tri) else np.zeros(0)
        self.big = np.flatnonzero(rad > big)
        small = np.flatnonzero(rad <= big)
        self.small = small
        self.r = float(rad[small].max()) if len(small) else 0.0
        self.tree = cKDTree(cen[small]) if len(small) else None

    def candidates(self, samples, pad=16.0):
        idx = set()
        if self.tree is not None:
            for js in self.tree.query_ball_point(samples, r=pad + self.r):
                idx.update(js)
        out = self.small[np.fromiter(idx, np.int64, len(idx))] if idx else np.zeros(0, np.int64)
        return np.unique(np.concatenate([out, self.big]))


def ray_hits(o, d, length, tri, eps=1e-9):
    """Moller-Trumbore: t (along unit d, 0..length) of the ray's hits with triangles tri [n, 3, 3]."""
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    h = np.cross(d[None], e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > eps
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = o[None] - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ d)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= length)
    return t[hit], np.flatnonzero(hit)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh_dir")
    ap.add_argument("store_dir")
    ap.add_argument("--z", type=float, nargs=2, default=None, help="only crossings inside [Z0, Z1)")
    ap.add_argument("--max-rays", type=int, default=4000)
    ap.add_argument("--umbilicus", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    st = load_store(a.store_dir)
    n_rays = len(st["offsets"]) - 1
    O, S, T = st["ray_origin_zyx"].astype(np.float64), st["ray_step_zyx"].astype(np.float64), st["crossing_t"]
    pts_z = [O[i, 0] + T[st["offsets"][i]:st["offsets"][i + 1]] * S[i, 0] for i in range(n_rays)]
    if a.z:
        keep = [i for i in range(n_rays) if len(pts_z[i]) >= 2 and pts_z[i].min() >= a.z[0] and pts_z[i].max() < a.z[1]]
    else:
        keep = [i for i in range(n_rays) if len(pts_z[i]) >= 2]
    rng = np.random.default_rng(a.seed)
    if len(keep) > a.max_rays:
        keep = sorted(rng.choice(keep, a.max_rays, replace=False).tolist())
    zlo = min(float(pts_z[i].min()) for i in keep) if keep else 0.0
    zhi = max(float(pts_z[i].max()) for i in keep) if keep else 0.0
    tri, lab = load_triangles(a.mesh_dir, zlo, zhi)
    index = TriIndex(tri)
    umb = None
    if a.umbilicus:
        cp = sorted(json.load(open(a.umbilicus))["control_points"], key=lambda q: q["z"])
        umb = (np.array([q["z"] for q in cp], float), np.array([q["y"] for q in cp], float),
               np.array([q["x"] for q in cp], float))
    c1, c5, radii, dist_hit = [], [], [], []
    for i in keep:
        o, s = O[i], S[i]
        L = float(np.linalg.norm(s))
        d = s / L
        ts = T[st["offsets"][i]:st["offsets"][i + 1]].astype(np.float64) * L
        length = float(ts.max()) + 24.0
        samples = o[None] + np.arange(0.0, length + 16, 16.0)[:, None] * d[None]
        cand = index.candidates(samples)
        if len(cand):
            th, idx = ray_hits(o, d, length, tri[cand])
            order = np.argsort(th)
            th = th[order]
            # a ray grazing a shared quad edge can hit two triangles of one winding at the same t
            wl = lab[np.array(cand)[idx[order]]]
            if len(th) > 1:
                dup = np.r_[False, (np.diff(th) < 1e-6) & (np.diff(wl) == 0)]
                th = th[~dup]
        else:
            th = np.zeros(0)
        cnt = np.searchsorted(th, ts)  # windings hit before each crossing
        c1 += (cnt[1:] - cnt[:-1]).tolist()
        c5 += (cnt[5:] - cnt[:-5]).tolist()
        if len(th):
            dist_hit += np.abs(ts[:, None] - th[None]).min(1).tolist()
        if umb is not None:
            p = o[None] + ts[:, None] * d[None]
            mid = 0.5 * (p[1:] + p[:-1])
            r = np.hypot(mid[:, 1] - np.interp(mid[:, 0], umb[0], umb[1]), mid[:, 2] - np.interp(mid[:, 0], umb[0], umb[2]))
            radii += r.tolist()
    c1, c5 = np.array(c1), np.array(c5)
    res = {"mesh_dir": a.mesh_dir, "store": a.store_dir, "rays": len(keep), "pairs_d1": int(len(c1)),
           "frac_exact_d1": round(float((c1 == 1).mean()), 4) if len(c1) else None,
           "count_d1_distribution": {str(k): round(float((c1 == k).mean()), 4) for k in range(0, 4)} | (
               {">=4": round(float((c1 >= 4).mean()), 4)} if len(c1) else {}),
           "mean_abs_err_d1": round(float(np.abs(c1 - 1).mean()), 4) if len(c1) else None,
           "pairs_d5": int(len(c5)),
           "frac_exact_d5": round(float((c5 == 5).mean()), 4) if len(c5) else None,
           "median_count_d5": float(np.median(c5)) if len(c5) else None,
           "median_crossing_to_nearest_winding_vox": round(float(np.median(dist_hit)), 2) if dist_hit else None}
    if umb is not None and len(c1):
        rr = np.array(radii)
        bands = {}
        for lo, hi in ((0, 300), (300, 800), (800, 1400), (1400, 99999)):
            k = (rr >= lo) & (rr < hi)
            if k.sum():
                bands[f"{lo}-{hi}"] = {"pairs": int(k.sum()), "frac_exact_d1": round(float((c1[k] == 1).mean()), 4)}
        res["by_radius_vox"] = bands
    print(json.dumps(res))


if __name__ == "__main__":
    main()
