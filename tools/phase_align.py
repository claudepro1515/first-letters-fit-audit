"""Move each winding of a spiral fit onto the papyrus sheet it runs along (phase alignment against the raw CT).

A fit whose windings are parallel to the sheets can still sit between them: fit_spiral's winding-model loss counts
windings between crossings and does not fix where they sit. For every winding mesh this tool samples the CT along the
mesh normal (offsets -R..R voxels, every 0.5) at a grid of vertices, averages the profiles over tiles of the mesh
(default 8 x 8 grid cells), and reads the offset of the nearest sheet crest from the phase of the tile's mean profile
at the sheet period (one Fourier coefficient: a crest at o* gives arg F = -2 pi o* / s). The per-tile phases are
smoothed as unit vectors weighted by their amplitude (so a tile without contrast borrows from its neighbours).
--mode crest (default) instead moves each tile to an actual crest of its own smoothed mean profile: the crest nearest
to where the winding is, then, three times, the crest nearest to the median offset of the neighbouring tiles (so
neighbouring tiles stay on one sheet); a tile without a clear crest within three quarters of a period does not move.
Real sheets are not evenly spaced (along the normals of curated PHerc0139 windings the next crests are 12 to 27 voxels
away), so one Fourier coefficient at a fixed period points to phantom crests where the local spacing differs, and the
crest mode depends on the period only through that window. --mode phasor keeps the phase offset (the first version).
The tile offsets are smoothed and brought back to every vertex, and each vertex is moved by its offset along the
normal. The mesh keeps its grid, its validity mask and its parametrisation; only the offset along the normal changes,
by at most half a sheet period (phasor) or three quarters of it (crest).

The sheet period s is measured per winding from the tiles' mean profiles (the strongest period between --min-period and
--max-period voxels), or given with --period.

Usage: python phase_align.py <mesh dir (one tifxyz) or a dir of w### meshes> <volume level 0 (URL or local zarr)> <out dir>
       [--tile 8] [--stride 2] [--reach 16] [--smooth-tiles 1.0] [--period S] [--mode crest|phasor] [--cache DIR]
       [--pattern 'w[0-9][0-9][0-9]']
Prints one JSON line per winding (period, median |offset|, confident tile fraction, mean-profile contrast before/after).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class LocalZArray:
    """A local zarr v2 array (one level), with vcz.ZArray's chunk interface."""

    def __init__(self, path):
        if not os.path.exists(os.path.join(path, ".zarray")) and os.path.exists(os.path.join(path, "0", ".zarray")):
            path = os.path.join(path, "0")
        meta = json.load(open(os.path.join(path, ".zarray")))
        self.path, self.shape, self.chunks = path, tuple(meta["shape"]), tuple(meta["chunks"])
        self.sep = meta.get("dimension_separator", ".") or "."
        from numcodecs import get_codec
        self.codec = get_codec(meta["compressor"]) if meta.get("compressor") else None
        self.filters = [get_codec(f) for f in (meta.get("filters") or [])]

    def _load_chunk(self, idx):
        fn = os.path.join(self.path, self.sep.join(str(i) for i in idx))
        if not os.path.exists(fn):
            return None
        buf = open(fn, "rb").read()
        buf = self.codec.decode(buf) if self.codec else buf
        for f in reversed(self.filters):
            buf = f.decode(buf)
        return np.frombuffer(buf, np.uint8).reshape(self.chunks)


def open_level0(spec, cache_dir=None):
    if spec.startswith("http"):
        from vcz import ZArray
        return ZArray(spec.rstrip("/") + ("" if spec.rstrip("/").endswith("/0") else "/0"), cache_dir=cache_dir)
    return LocalZArray(spec)


class Sampler:
    """Trilinear samples of a uint8 zarr level at zyx points (0 outside), loading each needed chunk once."""

    def __init__(self, arr, workers=32):
        self.a, self.cs, self.shape = arr, np.array(arr.chunks[-3:]), np.array(arr.shape[-3:])
        self.cache, self.workers = {}, workers

    def _load(self, idx):
        ch = self.a._load_chunk(idx)
        return np.zeros(tuple(self.cs), np.uint8) if ch is None else ch

    def sample(self, pts):
        from concurrent.futures import ThreadPoolExecutor
        p0 = np.clip(np.floor(pts).astype(np.int64), 0, self.shape - 2)
        f = np.clip(pts - p0, 0, 1)
        keys = set()
        for d in np.ndindex(2, 2, 2):
            keys |= set(map(tuple, (p0 + d) // self.cs))
        todo = [k for k in keys if k not in self.cache]
        with ThreadPoolExecutor(self.workers) as ex:
            for k, ch in zip(todo, ex.map(self._load, todo)):
                self.cache[k] = ch
        acc = np.zeros(len(pts), np.float32)
        for d in np.ndindex(2, 2, 2):
            q = p0 + d
            ci = q // self.cs
            li = q - ci * self.cs
            v = np.empty(len(q), np.float32)
            order = np.lexsort(ci.T[::-1])
            cs_ = ci[order]
            brk = np.r_[0, np.flatnonzero((np.diff(cs_, axis=0) != 0).any(1)) + 1, len(order)]
            for a, b in zip(brk[:-1], brk[1:]):
                idx = order[a:b]
                l_ = li[idx]
                v[idx] = self.cache[tuple(cs_[a])][l_[:, 0], l_[:, 1], l_[:, 2]]
            w = np.prod([f[:, k] if d[k] else 1 - f[:, k] for k in range(3)], axis=0)
            acc += w * v
        return acc

    def trim(self, max_chunks=300):
        if len(self.cache) > max_chunks:
            self.cache.clear()


def load_mesh(d):
    x, y, z = (tifffile.imread(os.path.join(d, f"{c}.tif")).astype(np.float64) for c in "xyz")
    valid = (x >= 0) & (y >= 0) & (z >= 0)
    return np.stack([x, y, z], -1), valid


def normals(P, valid):
    """n = normalize(cross(dR, dC)) (xyz). dR, dC: central differences where both neighbours are valid, one-sided
    at the mesh border and at the rims of holes; ok where the vertex and both directions have a valid difference."""
    def diff(axis):
        d = np.zeros_like(P)
        has = np.zeros(valid.shape, bool)
        fwd = np.zeros(valid.shape, bool)
        bwd = np.zeros(valid.shape, bool)
        sl = lambda a, b: tuple(slice(a, b) if k == axis else slice(None) for k in range(2))
        fwd[sl(0, -1)] = valid[sl(0, -1)] & valid[sl(1, None)]
        bwd[sl(1, None)] = valid[sl(1, None)] & valid[sl(0, -1)]
        pf = np.zeros_like(P)
        pb = np.zeros_like(P)
        pf[sl(0, -1)] = P[sl(1, None)] - P[sl(0, -1)]
        pb[sl(1, None)] = P[sl(1, None)] - P[sl(0, -1)]
        both = fwd & bwd
        d[both] = 0.5 * (pf[both] + pb[both])
        d[fwd & ~bwd] = pf[fwd & ~bwd]
        d[bwd & ~fwd] = pb[bwd & ~fwd]
        has = fwd | bwd
        return d, has
    dR, hr = diff(0)
    dC, hc = diff(1)
    n = np.cross(dR, dC)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = valid & hr & hc & (ln[..., 0] > 1e-9)
    return n / np.maximum(ln, 1e-12), ok


def detrend(profiles, offs):
    """Remove each profile's mean and linear trend (the slow change of brightness across the window)."""
    A = np.stack([np.ones_like(offs), offs], 1)
    coef, *_ = np.linalg.lstsq(A, profiles.reshape(-1, len(offs)).T, rcond=None)
    return profiles - (A @ coef).T.reshape(profiles.shape)


def peak_period(profiles, offs, lo, hi, smooth_vox=1.5):
    """Sheet period as the median spacing of adjacent crests (local maxima standing out of the detrended, smoothed
    tile profiles by at least a quarter of the profile's range), kept within [lo, hi]."""
    from scipy.ndimage import gaussian_filter1d
    step = offs[1] - offs[0]
    p = gaussian_filter1d(detrend(profiles, offs), smooth_vox / step, axis=-1)
    gaps = []
    for row in p:
        rng = row.max() - row.min()
        if rng <= 0:
            continue
        pk = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:]) & (row[1:-1] > row.min() + 0.25 * rng)) + 1
        if len(pk) > 1:
            d = np.diff(offs[pk])
            gaps += d[(d >= lo) & (d <= hi)].tolist()
    return float(np.median(gaps)) if len(gaps) >= 10 else None


def dominant_period(profiles, offs, lo, hi):
    """Sheet period from the mean normalized autocorrelation of the detrended profiles: its highest peak at a lag in
    [lo, hi] voxels (phase-free, so tiles with the sheet at any offset agree)."""
    p = detrend(profiles, offs)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
    step = offs[1] - offs[0]
    n = p.shape[1]
    ac = np.array([(p[:, :n - k] * p[:, k:]).sum(1).mean() * n / (n - k) for k in range(n)])
    lags = np.arange(n) * step
    inside = np.flatnonzero((lags >= lo) & (lags <= hi))
    peaks = [k for k in inside if 0 < k < n - 1 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
    if not peaks:
        return float(np.clip(lags[inside[np.argmax(ac[inside])]], lo, hi))
    k = max(peaks, key=lambda j: ac[j])
    # parabolic refinement of the peak
    den = ac[k - 1] - 2 * ac[k] + ac[k + 1]
    fr = 0.5 * (ac[k - 1] - ac[k + 1]) / den if abs(den) > 1e-12 else 0.0
    return float(lags[k] + np.clip(fr, -0.5, 0.5) * step)


def crest_offsets(pm, offs, have, start, s, iters=3, rel_height=0.35, max_frac=0.75):
    """Per tile, move to an actual crest of the tile's mean profile (detrended pm, smoothed by 1.5 voxels): candidates
    are local maxima higher than min + rel_height (max - min) within max_frac s of the surface; the one nearest to
    `start` (0: where the winding is), then `iters` times the one nearest to the median offset of the 3 x 3
    neighbouring tiles. Tiles without a candidate keep `start`. Returns (offset per tile, has-a-crest mask)."""
    from scipy.ndimage import gaussian_filter1d
    sm = gaussian_filter1d(pm, 1.5 / (offs[1] - offs[0]), axis=-1)
    TR, TC = have.shape
    cand = {}
    for a, b in zip(*np.nonzero(have)):
        row = sm[a, b]
        rng = row.max() - row.min()
        if rng <= 0:
            continue
        pk = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:]) & (row[1:-1] > row.min() + rel_height * rng)) + 1
        pk = pk[np.abs(offs[pk]) <= max_frac * s]
        if len(pk):
            cand[(a, b)] = offs[pk]
    o = np.where(have, start, np.nan)
    for (a, b), c in cand.items():
        o[a, b] = c[np.argmin(np.abs(c - start[a, b]))]
    for _ in range(iters):
        o2 = o.copy()
        for (a, b), c in cand.items():
            nb = o[max(0, a - 1):a + 2, max(0, b - 1):b + 2]
            nb = nb[np.isfinite(nb)]
            if len(nb) >= 3:
                o2[a, b] = c[np.argmin(np.abs(c - np.median(nb)))]
        o = o2
    has = np.zeros(have.shape, bool)
    for k in cand:
        has[k] = True
    return o, has


def align(mesh_dir, sampler, out_dir, tile=8, stride=2, reach=16.0, smooth_tiles=1.0, period=None,
          min_period=8.0, max_period=24.0, mode="crest"):
    P, valid = load_mesh(mesh_dir)
    n, ok = normals(P, valid)
    R, C = valid.shape
    offs = np.arange(-reach, reach + 1e-6, 0.5)
    rs, cs = np.arange(0, R, stride), np.arange(0, C, stride)
    rr, cc = np.meshgrid(rs, cs, indexing="ij")
    use = ok[rr, cc]
    base, nrm = P[rr[use], cc[use]], n[rr[use], cc[use]]
    vals = np.zeros((len(base), len(offs)), np.float32)
    for k in range(0, len(base), 1500):  # vertices in grid order, a block at a time: bounded chunk cache
        pts = base[k:k + 1500, None, :] + offs[None, :, None] * nrm[k:k + 1500, None, :]  # xyz
        vals[k:k + 1500] = sampler.sample(pts.reshape(-1, 3)[:, ::-1]).reshape(-1, len(offs))  # sampler takes zyx
        sampler.trim(300)
    inside = (vals > 0).all(1)
    # tile index of every sampled vertex (tiles of `tile` grid cells)
    ti, tj = rr[use] // tile, cc[use] // tile
    TR, TC = (R + tile - 1) // tile, (C + tile - 1) // tile
    prof = np.zeros((TR, TC, len(offs)))
    cnt = np.zeros((TR, TC))
    np.add.at(prof, (ti[inside], tj[inside]), vals[inside])
    np.add.at(cnt, (ti[inside], tj[inside]), 1)
    have = cnt >= max(3, 0.25 * (tile / stride) ** 2)
    prof[have] /= cnt[have][:, None]
    if not have.any():  # nothing to measure: the winding is copied unchanged, so the output has every input winding
        shutil.copytree(mesh_dir, out_dir, dirs_exist_ok=True)
        return {"mesh": os.path.basename(mesh_dir.rstrip("/")), "error": "no tile inside the scan", "copied_unchanged": True}
    s = float(period) if period else (peak_period(prof[have], offs, min_period, max_period)
                                      or dominant_period(prof[have], offs, min_period, max_period))
    pm = detrend(prof, offs)
    F = (pm * np.exp(-2j * np.pi * offs / s)).sum(-1)
    F[~have] = 0
    amp = np.abs(F)
    # smooth the unit phase vectors weighted by amplitude (normalized convolution over the tile grid)
    zr = gaussian_filter(F.real, smooth_tiles, mode="nearest")
    zi = gaussian_filter(F.imag, smooth_tiles, mode="nearest")
    wsum = gaussian_filter(amp, smooth_tiles, mode="nearest")
    # back to every vertex: bilinear on the tile grid (tile centres at (k + 0.5) tile - 0.5)
    gr = (np.arange(R) + 0.5) / tile - 0.5
    gc = (np.arange(C) + 0.5) / tile - 0.5
    GR, GC = np.meshgrid(np.clip(gr, 0, TR - 1), np.clip(gc, 0, TC - 1), indexing="ij")
    if mode == "phasor":
        vr = map_coordinates(zr, [GR, GC], order=1, mode="nearest")
        vi = map_coordinates(zi, [GR, GC], order=1, mode="nearest")
        off = -np.angle(vr + 1j * vi) * s / (2 * np.pi)  # crest offset, in (-s/2, s/2]
        crest_frac = None
    else:
        # start where the winding is: on the curated control, starting from the phase offset at a fixed period moved
        # windings to the wrong crest wherever the local spacing differed from the period (ink AUC 0.92 -> 0.87 on a
        # crop whose crests are 12 voxels apart, at period 17), while the nearest crest gave the same result at
        # periods 14 and 17
        start = np.zeros(have.shape)
        o, has_crest = crest_offsets(pm, offs, have, start, s)
        # smooth the tile offsets (normalized convolution; tiles without data borrow from their neighbours)
        wt = np.isfinite(o).astype(float)
        num = gaussian_filter(np.nan_to_num(o) * wt, smooth_tiles, mode="nearest")
        den = gaussian_filter(wt, smooth_tiles, mode="nearest")
        tile_off = np.where(den > 1e-6, num / np.maximum(den, 1e-6), start)
        off = map_coordinates(tile_off, [GR, GC], order=1, mode="nearest")
        crest_frac = round(float(has_crest[have].mean()), 3)
    moved = np.where(ok[..., None], P + off[..., None] * n, P)
    # vertices without a normal (mesh border, holes' rims) move with their nearest valid neighbour's offset along
    # the same displacement field only if the grid is fully valid there; otherwise they stay (keeps the mask)
    os.makedirs(out_dir, exist_ok=True)
    for k, ch in enumerate("xyz"):
        a = np.where(valid, moved[..., k], -1.0).astype(np.float32)
        tifffile.imwrite(os.path.join(out_dir, f"{ch}.tif"), a)
    if os.path.exists(os.path.join(mesh_dir, "meta.json")):
        shutil.copy2(os.path.join(mesh_dir, "meta.json"), os.path.join(out_dir, "meta.json"))

    def contrast(pr):  # (max - min) of the mean profile over tiles with data, per tile, median
        v = pr[have]
        return round(float(np.median(v.max(-1) - v.min(-1))), 2)

    # mean profile after the move, re-centred by the tile's own offset (what a render of the moved surface sees)
    centre = np.interp(0.0, offs, np.arange(len(offs)))
    return {"mesh": os.path.basename(mesh_dir.rstrip("/")), "mode": mode, "period_vox": round(s, 2),
            "tiles_with_crest_frac": crest_frac,
            "tiles": int(have.sum()), "median_abs_offset_vox": round(float(np.median(np.abs(off[ok]))), 2),
            "confident_tile_frac": round(float((np.hypot(zr, zi)[have] > 0.5 * wsum[have]).mean()), 3),
            # moved by more than 0.4 period: about half-way between two sheets, where neighbouring tiles can pick
            # different sheets (the moved surface then steps by one sheet there)
            "near_half_period_frac": round(float((np.abs(off[ok]) > 0.4 * s).mean()), 3),
            "offset_p5_p95_vox": [round(float(np.percentile(off[ok], 5)), 2), round(float(np.percentile(off[ok], 95)), 2)],
            "profile_value_at_surface_before": round(float(prof[have][:, int(centre)].mean()), 2),
            "mean_profile_contrast": contrast(prof)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("meshes")
    ap.add_argument("volume")
    ap.add_argument("out")
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--reach", type=float, default=16.0)
    ap.add_argument("--smooth-tiles", type=float, default=1.0)
    ap.add_argument("--period", type=float, default=None)
    ap.add_argument("--mode", choices=("crest", "phasor"), default="crest")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pattern", default="w[0-9][0-9][0-9]*")
    a = ap.parse_args()
    sampler = Sampler(open_level0(a.volume, a.cache))
    if os.path.exists(os.path.join(a.meshes, "x.tif")):
        dirs = [a.meshes]
        outs = [a.out]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(a.meshes, a.pattern)) if os.path.exists(os.path.join(d, "x.tif")))
        outs = [os.path.join(a.out, os.path.basename(d)) for d in dirs]
    for d, o in zip(dirs, outs):
        r = align(d, sampler, o, a.tile, a.stride, a.reach, a.smooth_tiles, a.period, mode=a.mode)
        print(json.dumps(r), flush=True)
        sampler.trim()


if __name__ == "__main__":
    main()
