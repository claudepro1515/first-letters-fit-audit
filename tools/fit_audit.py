"""Audit fitted spiral windings (tifxyz) against the Lasagna layer-phase field.

For every mesh vertex we sample the Lasagna `cos` prediction (sheet crests = 1) along the mesh
normal, find where the nearest sheet crest sits relative to the vertex (delta, in voxels) and
the local sheet spacing. Along each mesh row (constant z, running around the winding) the
crest offset is unwrapped, giving how many sheets the mesh slides across while it goes round.

A fit whose windings stay on their sheet slides ~0 sheets per turn. A fit that uses the wrong
spiral_outward_sense cannot follow the true spiral with an orientation-preserving warp, so it has
to slip sheets somewhere; this shows up as systematic drift or as bands of off-sheet vertices.

Usage: python fit_audit.py <mesh_dir_or_parent> --lasagna-cos <url of *_cos.ome.zarr> [--level 1]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcz import OmeZarr  # noqa: E402


def load_tifxyz(d):
    x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    valid = (x >= 0) & (y >= 0) & (z >= 0)
    meta = {}
    if os.path.exists(os.path.join(d, "meta.json")):
        meta = json.load(open(os.path.join(d, "meta.json")))
    return np.stack([z, y, x], -1), valid, meta  # [rows, cols, 3] in zyx


def mesh_normals(P, valid):
    dcol = np.zeros_like(P)
    drow = np.zeros_like(P)
    dcol[:, 1:-1] = P[:, 2:] - P[:, :-2]
    drow[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(drow, dcol)  # zyx cross product (orientation only needs to be consistent)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = valid & (nn[..., 0] > 1e-6)
    ok[:, 0] = ok[:, -1] = False
    ok[0, :] = ok[-1, :] = False
    ok[1:-1, 1:-1] &= valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
    return n / np.maximum(nn, 1e-9), ok


class ChunkSampler:
    """Trilinear sampling of one zarr level at arbitrary points, fetching only needed chunks."""

    def __init__(self, arr, workers=48):
        self.a = arr
        self.cs = np.array(arr.chunks)
        self.cache = {}
        self.workers = workers

    def prefetch(self, pts):
        c = np.unique((np.floor(pts).astype(np.int64) // self.cs), axis=0)
        c2 = np.unique(np.concatenate([c, ((np.floor(pts) + 1).astype(np.int64) // self.cs)]), axis=0)
        todo = [tuple(v) for v in c2 if tuple(v) not in self.cache and all(v >= 0)]

        def job(idx):
            return idx, self.a._load_chunk(idx)

        with ThreadPoolExecutor(self.workers) as ex:
            for idx, ch in ex.map(job, todo):
                self.cache[idx] = ch

    def value(self, iz, iy, ix):
        out = np.zeros(len(iz), np.float32)
        ci = np.stack([iz, iy, ix], -1) // self.cs
        li = np.stack([iz, iy, ix], -1) - ci * self.cs
        keys = [tuple(v) for v in ci]
        uk = {}
        for n, k in enumerate(keys):
            uk.setdefault(k, []).append(n)
        for k, idxs in uk.items():
            ch = self.cache.get(k)
            if ch is None:
                continue
            idxs = np.array(idxs)
            l = li[idxs]
            out[idxs] = ch[l[:, 0], l[:, 1], l[:, 2]]
        return out

    def sample(self, pts):
        """pts [N,3] zyx in this level's voxel units -> trilinear values (uint8 scale)."""
        p0 = np.floor(pts).astype(np.int64)
        f = pts - p0
        shp = np.array(self.a.shape)
        p0 = np.clip(p0, 0, shp - 2)
        acc = np.zeros(len(pts), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
                    acc += w * self.value(p0[:, 0] + dz, p0[:, 1] + dy, p0[:, 2] + dx)
        return acc


def crest_offsets(profiles, ts, min_val=0.2):
    """profiles [N, T] in [-1,1]; return nearest crest offset (voxels), local spacing, crest strength."""
    N, T = profiles.shape
    p = profiles
    pk = np.zeros_like(p, bool)
    pk[:, 1:-1] = (p[:, 1:-1] > p[:, :-2]) & (p[:, 1:-1] >= p[:, 2:]) & (p[:, 1:-1] > min_val)
    delta = np.full(N, np.nan)
    spacing = np.full(N, np.nan)
    strength = np.full(N, np.nan)
    mid = T // 2
    for i in range(N):
        idx = np.nonzero(pk[i])[0]
        if len(idx) == 0:
            continue
        # sub-sample refinement
        pos = []
        for j in idx:
            ym, y0, yp = p[i, j - 1], p[i, j], p[i, j + 1]
            den = ym - 2 * y0 + yp
            fr = 0.5 * (ym - yp) / den if abs(den) > 1e-6 else 0.0
            pos.append(ts[j] + np.clip(fr, -0.5, 0.5) * (ts[1] - ts[0]))
        pos = np.array(pos)
        k = np.argmin(np.abs(pos))
        delta[i] = pos[k]
        strength[i] = p[i, idx[k]]
        if len(pos) > 1:
            spacing[i] = np.median(np.diff(np.sort(pos)))
    return delta, spacing, strength


def audit_mesh(d, sampler, level_scale, t_half=24.0, t_step=1.0, min_val=0.2, every=1):
    P, valid, meta = load_tifxyz(d)
    if every > 1:
        P, valid = P[::every, ::every], valid[::every, ::every]
    n, ok = mesh_normals(P, valid)
    R, C = valid.shape
    ts = np.arange(-t_half, t_half + 1e-6, t_step)
    rr, cc = np.nonzero(ok)
    pts = P[rr, cc][:, None, :] + ts[None, :, None] * n[rr, cc][:, None, :]
    flat = pts.reshape(-1, 3) / level_scale
    sampler.prefetch(flat)
    vals = sampler.sample(flat).reshape(len(rr), len(ts))
    prof = (vals - 128.0) / 127.0
    prof[vals == 0] = np.nan
    good_prof = ~np.isnan(prof).any(1)
    prof2 = np.nan_to_num(prof, nan=-1.0)
    delta, spacing, strength = crest_offsets(prof2, ts, min_val)
    delta[~good_prof] = np.nan
    D = np.full((R, C), np.nan)
    S = np.full((R, C), np.nan)
    D[rr, cc] = delta
    S[rr, cc] = spacing
    sp_med = float(np.nanmedian(S)) if np.isfinite(S).any() else np.nan
    # unwrap along each row (around the winding)
    drift_rows, lens = [], []
    for r in range(R):
        dd = D[r]
        idx = np.nonzero(np.isfinite(dd))[0]
        if len(idx) < 10:
            continue
        # segments of consecutive valid samples (allow gaps of <= 2 columns)
        seg_start = 0
        total = 0.0
        cnt = 0
        prev = dd[idx[0]]
        acc = 0.0
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > 3:
                total += acc
                acc = 0.0
                prev = dd[b]
                continue
            step = dd[b] - prev
            # unwrap by one local spacing
            sp = S[r, b] if np.isfinite(S[r, b]) else sp_med
            if np.isfinite(sp) and sp > 0:
                step = (step + sp / 2) % sp - sp / 2
            acc += step
            prev = dd[b]
            cnt += 1
        total += acc
        drift_rows.append(total / sp_med if sp_med > 0 else np.nan)
        lens.append(cnt)
    on_sheet = np.abs(D) < 0.25 * sp_med
    # phase of the vertex relative to the nearest crest, in units of the local spacing (0 = on a crest);
    # R = |mean exp(2 pi i phase)| is 1 when every vertex sits at the same offset from its sheet, ~0 when random
    Sl = np.where(np.isfinite(S), S, sp_med)
    ph = D / Sl
    fin = np.isfinite(ph)
    R_all = float(np.abs(np.exp(2j * np.pi * ph[fin]).mean())) if fin.any() else None
    # the same in 5 x 5-vertex tiles (a surface can follow the sheets with an offset that varies slowly)
    Rt = []
    for r0 in range(0, R - 4, 5):
        for c0 in range(0, C - 4, 5):
            t = ph[r0:r0 + 5, c0:c0 + 5]
            t = t[np.isfinite(t)]
            if len(t) >= 15:
                Rt.append(float(np.abs(np.exp(2j * np.pi * t).mean())))
    res = dict(
        mesh=os.path.basename(d.rstrip("/")),
        rows=R, cols=C, valid_frac=float(valid.mean()),
        measured_frac=float(np.isfinite(D).sum() / max(valid.sum(), 1)),
        spacing_vox=sp_med,
        on_sheet_frac=float(on_sheet[np.isfinite(D)].mean()) if np.isfinite(D).any() else None,
        mean_abs_delta_layers=float(np.nanmean(np.abs(D)) / sp_med) if sp_med > 0 else None,
        drift_layers_median=float(np.median(drift_rows)) if drift_rows else None,
        drift_layers_mean=float(np.mean(drift_rows)) if drift_rows else None,
        drift_rows=len(drift_rows),
        phase_R=R_all,
        phase_R_tile_median=float(np.median(Rt)) if Rt else None,
        tiles=len(Rt),
    )
    return res, D, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meshes", help="a tifxyz dir or a parent dir of tifxyz dirs")
    ap.add_argument("--lasagna-cos", required=True)
    ap.add_argument("--level", default="1")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--out", default="audit.json")
    ap.add_argument("--save-maps", default=None)
    ap.add_argument("--t-half", type=float, default=24.0, help="probe half-length along the normal (mesh units)")
    ap.add_argument("--t-step", type=float, default=1.0)
    ap.add_argument("--every", type=int, default=1, help="use every n-th mesh row and column")
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.meshes, "x.tif")):
        dirs = [a.meshes]
    else:
        dirs = sorted(d for d in glob.glob(os.path.join(a.meshes, a.pattern)) if os.path.exists(os.path.join(d, "x.tif")))
    g = OmeZarr(a.lasagna_cos, cache_dir=a.cache)
    arr = g.level(a.level)
    scale = g.scales.get(a.level, 2.0)
    sampler = ChunkSampler(arr)
    out = []
    for d in dirs:
        r, D, S = audit_mesh(d, sampler, scale, a.t_half, a.t_step, every=a.every)
        out.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        if a.save_maps:
            os.makedirs(a.save_maps, exist_ok=True)
            np.savez_compressed(os.path.join(a.save_maps, r["mesh"] + ".npz"), D=D, S=S)
        if len(sampler.cache) > 60000:
            sampler.cache.clear()
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
