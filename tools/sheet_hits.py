"""Do a spiral fit's windings lie on the papyrus sheets? A check along rays through the raw CT, no model involved.

Along each ray the CT intensity is sampled every 0.5 voxel and lightly smoothed. Where a fitted winding crosses
the ray, its intensity is compared with the intensity half-way to the neighbouring crossings:

    c = (I(hit) - mean(I(mid_left), I(mid_right))) / (p95 - p5 of the ray)

A fit whose windings follow the sheets one to one puts every crossing on a sheet and every mid-point in a gap,
so c > 0 almost everywhere (frac_c_pos ~ 1). With two windings per sheet the crossings that fall in gaps have
c <= 0, so frac_c_pos ~ 0.5; windings half a spacing off the sheets give frac_c_pos ~ 0, and positions unrelated
to the sheets ~0.5 or less (synthetic check, tests/test_sheet_hits.py: 1.0 / 0.50 / 0.0, and 0.33 for a pitch
of 0.67). Where the sheets touch (compressed regions) the CT has little contrast and c is small whatever the fit.

Rays: radial from the umbilicus (--radial N: random z in the band and random angle, radius --r0..--r1), or the
rays of a winding-inference crossing store (--store DIR: the store's own rays, near-normal to the sheets). With a
store, the model's crossings are scored the same way (are the model's sheets on bright CT?), and each model
crossing's distance to the nearest fitted winding is reported in units of the local crossing gap (phase: 0 = the
fit crosses the ray on the same sheet as the model, 0.5 = half-way between the model's sheets).

Usage:
  python sheet_hits.py <mesh_dir> <volume_url_or_path_level0> --z Z0 Z1 --umbilicus U.json --radial 200 [--r0 300 --r1 2500]
  python sheet_hits.py <mesh_dir> <volume> --z Z0 Z1 --store <winding_inference_dir> [--max-rays 1000]
Prints one JSON line (per-crossing statistics for the fit, and for the model with --store).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from winding_agreement import TriIndex, load_store, load_triangles, ray_hits  # noqa: E402


class Sampler:
    """Trilinear sampling of a uint8 zarr level (vcz.ZArray or a local zarr array exposing .chunks and
    _load_chunk / __getitem__) at arbitrary zyx points, loading each needed chunk once."""

    def __init__(self, arr, workers=32):
        self.a = arr
        self.cs = np.array(arr.chunks[-3:])
        self.shape = np.array(arr.shape[-3:])
        self.cache = {}
        self.workers = workers

    def _load(self, idx):
        if hasattr(self.a, "_load_chunk"):
            ch = self.a._load_chunk(idx)
            return np.zeros(tuple(self.cs), np.uint8) if ch is None else ch
        lo = np.array(idx) * self.cs
        hi = np.minimum(lo + self.cs, self.shape)
        out = np.zeros(tuple(self.cs), np.uint8)
        blk = np.asarray(self.a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        out[:blk.shape[0], :blk.shape[1], :blk.shape[2]] = blk
        return out

    def sample(self, pts):
        p0 = np.floor(pts).astype(np.int64)
        p0 = np.clip(p0, 0, self.shape - 2)
        f = np.clip(pts - p0, 0, 1)
        keys = set()
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    keys |= set(map(tuple, (p0 + (dz, dy, dx)) // self.cs))
        todo = [k for k in keys if k not in self.cache]
        with ThreadPoolExecutor(self.workers) as ex:
            for k, ch in zip(todo, ex.map(self._load, todo)):
                self.cache[k] = ch
        acc = np.zeros(len(pts), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    q = p0 + (dz, dy, dx)
                    ci = q // self.cs
                    li = q - ci * self.cs
                    v = np.empty(len(q), np.float32)
                    order = np.lexsort(ci.T[::-1])
                    cs_ = ci[order]
                    brk = np.r_[0, np.flatnonzero((np.diff(cs_, axis=0) != 0).any(1)) + 1, len(order)]
                    for a, b in zip(brk[:-1], brk[1:]):
                        idx = order[a:b]
                        ch = self.cache[tuple(cs_[a])]
                        l_ = li[idx]
                        v[idx] = ch[l_[:, 0], l_[:, 1], l_[:, 2]]
                    w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
                    acc += w * v
        return acc

    def trim(self, max_chunks=800):
        if len(self.cache) > max_chunks:
            self.cache.clear()


def open_level0(spec, cache_dir=None):
    if spec.startswith("http"):
        from vcz import ZArray
        return ZArray(spec.rstrip("/") + ("" if spec.rstrip("/").endswith("/0") else "/0"), cache_dir=cache_dir)
    import zarr
    g = zarr.open(spec, mode="r")
    return g["0"] if hasattr(g, "keys") and "0" in g else g


def contrast_at(prof, inside, ts, t_pos, return_t=False):
    """c for every interior position of the sorted list t_pos (profile prof sampled at ts; inside = the samples
    within the scan mask). Positions whose own sample or neighbouring mid-points fall outside the mask are dropped."""
    empty = (np.zeros(0), np.zeros(0)) if return_t else np.zeros(0)
    if len(t_pos) < 3 or inside.sum() < 20:
        return empty
    rng = np.percentile(prof[inside], 95) - np.percentile(prof[inside], 5)
    if rng <= 0:
        return empty
    I = np.interp(t_pos, ts, prof)
    mids = 0.5 * (t_pos[1:] + t_pos[:-1])
    Im = np.interp(mids, ts, prof)
    ins = np.interp(t_pos, ts, inside.astype(float)) > 0.99
    insm = np.interp(mids, ts, inside.astype(float)) > 0.99
    keep = ins[1:-1] & insm[:-1] & insm[1:]
    c = ((I[1:-1] - 0.5 * (Im[:-1] + Im[1:])) / rng)[keep]
    return (c, t_pos[1:-1][keep]) if return_t else c


def summarize(c):
    c = np.asarray(c)
    if len(c) == 0:
        return {"n": 0}
    return {"n": int(len(c)), "median_c": round(float(np.median(c)), 4), "mean_c": round(float(np.mean(c)), 4),
            "frac_c_pos": round(float((c > 0).mean()), 4),
            "frac_c_gt_0.1": round(float((c > 0.1).mean()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh_dir")
    ap.add_argument("volume", help="level-0 zarr (URL of the OME-Zarr or its /0, or a local path)")
    ap.add_argument("--z", type=float, nargs=2, required=True)
    ap.add_argument("--umbilicus", default=None)
    ap.add_argument("--radial", type=int, default=0, help="number of radial rays from the umbilicus")
    ap.add_argument("--r0", type=float, default=300.0)
    ap.add_argument("--r1", type=float, default=2500.0)
    ap.add_argument("--store", default=None)
    ap.add_argument("--max-rays", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=1.0, help="smoothing of the CT profile (voxels)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--cache", default=None, help="directory for downloaded CT chunks (shared between runs)")
    ap.add_argument("--null", type=int, default=0,
                    help="also score K random-phase copies of the fit: every ray's crossings shifted together by a random "
                         "offset within +-half their median gap on that ray (the chance level on this scan)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    samp = Sampler(open_level0(a.volume, a.cache))
    rays = []  # (origin zyx, unit direction, length, model crossing t's or None)
    if a.store:
        st = load_store(a.store)
        O, S, T, off = st["ray_origin_zyx"].astype(float), st["ray_step_zyx"].astype(float), st["crossing_t"], st["offsets"]
        idx = [i for i in range(len(off) - 1) if off[i + 1] - off[i] >= 3
               and a.z[0] <= O[i, 0] + T[off[i]:off[i + 1]].min() * S[i, 0] and O[i, 0] + T[off[i]:off[i + 1]].max() * S[i, 0] < a.z[1]]
        if len(idx) > a.max_rays:
            idx = sorted(rng.choice(idx, a.max_rays, replace=False).tolist())
        for i in idx:
            L = float(np.linalg.norm(S[i]))
            ts = np.sort(T[off[i]:off[i + 1]].astype(float) * L)
            o = O[i] + (ts[0] - 12.0) * S[i] / L
            rays.append((o, S[i] / L, float(ts[-1] - ts[0] + 24.0), ts - ts[0] + 12.0))
    if a.radial:
        cp = sorted(json.load(open(a.umbilicus))["control_points"], key=lambda q: q["z"])
        uz = np.array([q["z"] for q in cp], float)
        uy = np.array([q["y"] for q in cp], float)
        ux = np.array([q["x"] for q in cp], float)
        for _ in range(a.radial):
            z = rng.uniform(a.z[0] + 20, a.z[1] - 20)
            th = rng.uniform(0, 2 * np.pi)
            d = np.array([0.0, np.sin(th), np.cos(th)])
            o = np.array([z, np.interp(z, uz, uy), np.interp(z, uz, ux)]) + a.r0 * d
            rays.append((o, d, a.r1 - a.r0, None))
    tri, lab = load_triangles(a.mesh_dir, a.z[0], a.z[1])
    index = TriIndex(tri)
    c_fit, c_model, phase, hits_per_ray, r_fit, gaps = [], [], [], [], [], []
    same_w = []  # per gap: are the two crossings on the same winding (a fold or a graze of one winding)?
    null_rng, null_c = np.random.default_rng(a.seed + 1000), [[] for _ in range(a.null)]  # own RNG: rays unchanged
    gap_r, crest_sp, crest_r = [], [], []  # fitted gaps and CT crest spacings with the radius where they are measured
    umb = None
    if a.umbilicus:
        cp_ = sorted(json.load(open(a.umbilicus))["control_points"], key=lambda q: q["z"])
        umb = tuple(np.array([q[k] for q in cp_], float) for k in ("z", "y", "x"))
    for n, (o, d, length, tm) in enumerate(rays):
        ts = np.arange(0.0, length + 1e-6, 0.5)
        pts = o[None] + ts[:, None] * d[None]
        ok = (pts >= 0).all(1) & (pts < samp.shape - 1).all(1)
        if ok.mean() < 0.9:
            continue
        raw = samp.sample(np.clip(pts, 0, samp.shape - 2))
        inside = raw > 0  # masked volumes are 0 outside the scroll
        if inside.mean() < 0.5:
            continue
        prof = gaussian_filter1d(raw, a.sigma / 0.5)
        cand = index.candidates(pts[::32])
        th, wl = np.zeros(0), np.zeros(0, int)
        if len(cand):
            th, ii = ray_hits(o, d, length, tri[cand])
            order = np.argsort(th)
            th, wl = th[order], lab[np.array(cand)[ii[order]]]
            if len(th) > 1:
                keep = ~np.r_[False, (np.diff(th) < 1e-6) & (np.diff(wl) == 0)]
                th, wl = th[keep], wl[keep]
        hits_per_ray.append(len(th))
        if len(th) > 1:
            gaps.append(np.diff(th))  # spacing of consecutive fitted crossings along the ray (voxels)
            same_w.append(np.diff(wl) == 0)
        if umb is not None:
            # the CT's own crests along the ray (smoothed profile, prominence >= 0.15 of the ray's p95 - p5, >= 4 voxels
            # apart), as a model-free reference for how far apart the sheets are here
            from scipy.signal import find_peaks
            rng_ = np.percentile(raw[inside], 95) - np.percentile(raw[inside], 5) if inside.any() else 0.0
            if rng_ > 0:
                pk, _ = find_peaks(np.where(inside, prof, prof.min()), prominence=0.15 * rng_, distance=8)
                pk = pk[inside[pk]]
                if len(pk) > 1:
                    tc = ts[pk]
                    mid = o[None] + (0.5 * (tc[1:] + tc[:-1]))[:, None] * d[None]
                    crest_sp.append(np.diff(tc))
                    crest_r.append(np.hypot(mid[:, 1] - np.interp(mid[:, 0], umb[0], umb[1]), mid[:, 2] - np.interp(mid[:, 0], umb[0], umb[2])))
            if len(th) > 1:
                mid = o[None] + (0.5 * (th[1:] + th[:-1]))[:, None] * d[None]
                gap_r.append(np.hypot(mid[:, 1] - np.interp(mid[:, 0], umb[0], umb[1]), mid[:, 2] - np.interp(mid[:, 0], umb[0], umb[2])))
        cf, tf = contrast_at(prof, inside, ts, th, return_t=True)
        c_fit.append(cf)
        if a.null and len(th) >= 3:
            g = float(np.median(np.diff(th)))
            for k in range(a.null):
                null_c[k].append(contrast_at(prof, inside, ts, th + null_rng.uniform(-0.5, 0.5) * g))
        if umb is not None and len(tf):
            p = o[None] + tf[:, None] * d[None]
            r_fit.append(np.hypot(p[:, 1] - np.interp(p[:, 0], umb[0], umb[1]), p[:, 2] - np.interp(p[:, 0], umb[0], umb[2])))
        if tm is not None:
            c_model.append(contrast_at(prof, inside, ts, tm))
            if len(th) and len(tm) >= 3:
                gap = 0.5 * (tm[2:] - tm[:-2])
                dist = np.abs(tm[1:-1, None] - th[None]).min(1)
                phase.append(np.clip(dist / np.maximum(gap, 1e-6), 0, 1))
        if n % 50 == 0:
            samp.trim()
    c_fit = np.concatenate(c_fit) if c_fit else np.zeros(0)
    res = {"label": a.label or os.path.basename(os.path.dirname(a.mesh_dir.rstrip("/"))) + "/" + os.path.basename(a.mesh_dir.rstrip("/")),
           "rays": len(hits_per_ray), "ray_kind": "store" if a.store else "radial",
           "fit": summarize(c_fit), "median_hits_per_ray": float(np.median(hits_per_ray)) if hits_per_ray else None}
    if gaps:
        g = np.concatenate(gaps)
        med = float(np.median(g))
        # windings that sit on the same sheet (e.g. two of them moved onto one sheet) show up as gaps near 0
        res["gaps_vox"] = {"n": int(len(g)), "median": round(med, 2), "frac_lt_half_median": round(float((g < 0.5 * med).mean()), 4),
                           "frac_lt_3vox": round(float((g < 3.0).mean()), 4),
                           # the same, split: two different windings on one sheet, or one winding crossing the ray twice
                           "frac_lt_3vox_two_windings": round(float(((g < 3.0) & ~np.concatenate(same_w)).mean()), 4),
                           "frac_lt_3vox_one_winding": round(float(((g < 3.0) & np.concatenate(same_w)).mean()), 4),
                           "frac_gt_1.5_median": round(float((g > 1.5 * med).mean()), 4)}  # a sheet skipped
    if gap_r or crest_r:
        # fitted winding spacing vs the CT's crest spacing, by radius from the umbilicus (voxels, medians): a fit
        # that follows the sheets one to one has about the same spacing as the crests
        gg, gr = (np.concatenate(gaps), np.concatenate(gap_r)) if gap_r else (np.zeros(0), np.zeros(0))
        cs_, cr_ = (np.concatenate(crest_sp), np.concatenate(crest_r)) if crest_r else (np.zeros(0), np.zeros(0))
        band = {}
        for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 99999)):
            kf, kc = (gr >= lo) & (gr < hi), (cr_ >= lo) & (cr_ < hi)
            if kf.sum() >= 10 or kc.sum() >= 10:
                band[f"{lo}-{hi}"] = {"fit_gap": round(float(np.median(gg[kf])), 1) if kf.sum() >= 10 else None, "fit_n": int(kf.sum()),
                                      "ct_crest_spacing": round(float(np.median(cs_[kc])), 1) if kc.sum() >= 10 else None, "ct_n": int(kc.sum())}
        res["spacing_by_radius_vox"] = band
        if len(cs_):
            res["ct_crest_spacing_vox_median"] = round(float(np.median(cs_)), 2)
    if r_fit and len(c_fit):
        rr = np.concatenate(r_fit)
        if len(rr) == len(c_fit):
            res["fit_by_radius_vox"] = {f"{lo}-{hi}": summarize(c_fit[(rr >= lo) & (rr < hi)])
                                        for lo, hi in ((0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 99999))
                                        if ((rr >= lo) & (rr < hi)).sum() >= 20}
    if a.null and len(c_fit):
        fr = np.array([float((np.concatenate(v) > 0).mean()) for v in null_c if v and len(np.concatenate(v))])
        if len(fr):
            obs = float((c_fit > 0).mean())
            res["null_random_phase"] = {"copies": int(len(fr)), "frac_c_pos_mean": round(float(fr.mean()), 4),
                                        "frac_c_pos_p2.5": round(float(np.percentile(fr, 2.5)), 4),
                                        "frac_c_pos_p97.5": round(float(np.percentile(fr, 97.5)), 4),
                                        "frac_at_or_above_fit": round(float((fr >= obs).mean()), 4)}
    if a.store:
        res["model"] = summarize(np.concatenate(c_model) if c_model else [])
        ph = np.concatenate(phase) if phase else np.zeros(0)
        if len(ph):
            res["phase_model_to_fit"] = {"n": int(len(ph)), "median": round(float(np.median(ph)), 3),
                                         "frac_lt_0.15": round(float((ph < 0.15).mean()), 4),
                                         "frac_0.35_0.65": round(float(((ph >= 0.35) & (ph <= 0.65)).mean()), 4),
                                         "hist_0_to_1_by_0.1": np.histogram(ph, bins=10, range=(0, 1))[0].tolist()}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
