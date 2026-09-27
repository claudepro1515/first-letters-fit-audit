"""Render contrast of a tifxyz mesh region, before and after smoothing the mesh (CPU, straight from S3).

For a block of the mesh grid, renders 28 layers along the surface normal (one per voxel, centred on the
mesh, like vc_render_tifxyz) at about one pixel per voxel, and measures the local profile contrast of
every 128 x 128 tile: (max - min) / (max + min) of the tile's mean intensity per layer. It also fits the
sheet period to each tile profile and reports where the nearest sheet centre lies relative to the mesh
(sheet_offset_vox, median over tiles) and how consistent that is across tiles
(sheet_phase_concentration: 1 = the mesh keeps the same position relative to the sheets, 0 = it drifts
across them; contrast alone cannot see a rigid shift, these can). A mesh that
follows one sheet gives a peaked layer profile (high contrast); one that wanders across sheets or
between them gives a flat profile. With --smooth the vertex grid is first low-passed (Gaussian,
sigma in grid cells) to test whether mesh roughness is what flattens a render.

Usage:
  python render_contrast.py <tifxyz_dir> <volume.zarr url> --rows 10:31 --cols 100:121 \
      [--smooth 0 1.5 3] [--cache cache] [--png prefix]
Prints one JSON line per smoothing level.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_roughness import load_tifxyz, roughness_grid  # noqa: E402
from vcz import OmeZarr  # noqa: E402

LAYERS = np.arange(-14, 14)  # 28 layers, one voxel apart


class ChunkSampler:
    """Trilinear sampling of one zarr level at arbitrary points, fetching only the chunks needed."""

    def __init__(self, arr, workers=32):
        self.a, self.cs, self.cache, self.workers = arr, np.array(arr.chunks), {}, workers

    def prefetch(self, pts):
        p0 = np.floor(pts).astype(np.int64)
        c = np.unique(np.concatenate([p0 // self.cs, (p0 + 1) // self.cs]), axis=0)
        todo = [tuple(int(x) for x in v) for v in c if tuple(v) not in self.cache and (v >= 0).all()]
        with ThreadPoolExecutor(self.workers) as ex:
            for idx, ch in zip(todo, ex.map(self.a._load_chunk, todo)):
                self.cache[idx] = ch

    def _value(self, p):
        out = np.zeros(len(p), np.float32)
        ci = p // self.cs
        li = p - ci * self.cs
        key = (ci[:, 0] * 1_000_003 + ci[:, 1]) * 1_000_003 + ci[:, 2]
        for k in np.unique(key):
            sel = np.nonzero(key == k)[0]
            ch = self.cache.get(tuple(int(x) for x in ci[sel[0]]))
            if ch is not None:
                l = li[sel]
                out[sel] = ch[l[:, 0], l[:, 1], l[:, 2]]
        return out

    def sample(self, pts):
        p0 = np.floor(pts).astype(np.int64)
        f = (pts - p0).astype(np.float32)
        p0 = np.clip(p0, 0, np.array(self.a.shape) - 2)
        acc = np.zeros(len(pts), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
                    acc += w * self._value(p0 + np.array([dz, dy, dx]))
        return acc


def smooth_grid(P, valid, sigma):
    """Gaussian low-pass of the vertex positions (sigma in grid cells); invalid vertices untouched."""
    if sigma <= 0:
        return P.copy()
    w = gaussian_filter(valid.astype(np.float64), sigma)
    Q = P.copy()
    for k in range(3):
        Q[..., k] = gaussian_filter(np.where(valid, P[..., k], 0.0), sigma) / np.maximum(w, 1e-9)
    Q[~valid] = P[~valid]
    return Q


def densify(P, valid, step):
    """Bilinear upsampling of a grid block to about one vertex per voxel. Returns [H, W, 3] zyx and mask."""
    R, C = valid.shape
    H, W = int((R - 1) * step) + 1, int((C - 1) * step) + 1
    RR, CC = np.meshgrid(np.linspace(0, R - 1, H), np.linspace(0, C - 1, W), indexing="ij")
    D = np.stack([map_coordinates(P[..., k], [RR, CC], order=1) for k in range(3)], -1)
    m = map_coordinates(valid.astype(np.float32), [RR, CC], order=0) > 0.5
    return D, m


def dense_normals(D, sigma=3.0):
    Ds = np.stack([gaussian_filter(D[..., k], sigma) for k in range(3)], -1)
    n = np.cross(np.gradient(Ds, axis=0), np.gradient(Ds, axis=1))
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def render(D, m, n, sampler):
    H, W = m.shape
    mm = m.reshape(-1)
    base, nn = D.reshape(-1, 3)[mm], n.reshape(-1, 3)[mm]
    sampler.prefetch(np.concatenate([base + t * nn for t in (LAYERS[0], 0, LAYERS[-1])]))
    out = np.zeros((len(LAYERS), H * W), np.float32)
    for li, t in enumerate(LAYERS):
        pts = base + t * nn
        sampler.prefetch(pts)
        out[li, mm] = sampler.sample(pts)
    return out.reshape(len(LAYERS), H, W)


def tile_profiles(stack, m, tile=128):
    """Mean layer profile of every 128 x 128 tile that lies fully on the mesh and inside the scan."""
    profs = []
    L, H, W = stack.shape
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            mm = m[y:y + tile, x:x + tile]
            if mm.mean() < 0.9:
                continue
            block = stack[:, y:y + tile, x:x + tile][:, mm]
            if (block == 0).mean() > 0.05:  # outside the scanned / masked volume
                continue
            profs.append(block.mean(1))
    return profs


def tile_contrast(stack, m, tile=128):
    return [float((p.max() - p.min()) / (p.max() + p.min() + 1e-6)) for p in tile_profiles(stack, m, tile)]


def tile_sheet_phase(stack, m, tile=128, pmin=8.0, pmax=30.0):
    """Where the sheets sit relative to the mesh, per tile: fit a + b cos(2 pi t / P) + c sin(2 pi t / P) to
    the tile's layer profile (t = layer offset in voxels, P = its dominant period in [pmin, pmax]); the
    nearest bright-layer centre is at t* = P * atan2(c, b) / (2 pi). Returns (t*, P) per tile.
    A mesh that stays on one sheet gives the same t* in every tile (a surface tracer may sit a few voxels
    off the sheet centre, consistently); one that drifts across sheets gives scattered t*."""
    out = []
    t = LAYERS.astype(float)
    for p in tile_profiles(stack, m, tile):
        x = p - p.mean()
        best = None
        for P in np.arange(pmin, pmax + 0.01, 0.25):
            c, s_ = np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)
            A = np.stack([c, s_], 1)
            coef, *_ = np.linalg.lstsq(A, x, rcond=None)
            r = ((x - A @ coef) ** 2).sum()
            if best is None or r < best[0]:
                best = (r, P, coef)
        _, P, (b, c) = best
        out.append((float(P * np.arctan2(c, b) / (2 * np.pi)), float(P)))
    return out


def phase_summary(ph):
    """Median |t*| and the concentration R of the sheet phase over tiles (1 = same offset everywhere)."""
    if not ph:
        return None, None
    ang = np.array([2 * np.pi * ts / P for ts, P in ph])
    R = float(np.abs(np.mean(np.exp(1j * ang))))
    return round(float(np.median([abs(ts) for ts, _ in ph])), 1), round(R, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh")
    ap.add_argument("volume", help="OME-Zarr volume URL (level 0 is sampled)")
    ap.add_argument("--rows", required=True, help="r0:r1 grid rows of the block")
    ap.add_argument("--cols", required=True, help="c0:c1 grid columns of the block")
    ap.add_argument("--smooth", type=float, nargs="*", default=[0.0])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--png", default=None, help="save the middle layer of each render as <png>_s<sigma>.png")
    a = ap.parse_args()
    P, valid, meta = load_tifxyz(a.mesh)
    r0, r1 = (int(v) for v in a.rows.split(":"))
    c0, c1 = (int(v) for v in a.cols.split(":"))
    sc = (meta.get("scale") or [0.05])[0]
    sampler = ChunkSampler(OmeZarr(a.volume, cache_dir=a.cache).level("0"))
    for s in a.smooth:
        Q = smooth_grid(P, valid, s)
        blk, vb = Q[r0:r1, c0:c1], valid[r0:r1, c0:c1]
        rough = roughness_grid(blk, vb, meta)
        D, m = densify(blk, vb, 1 / sc)
        stack = render(D, m, dense_normals(D), sampler)
        vals = tile_contrast(stack, m)
        off, conc = phase_summary(tile_sheet_phase(stack, m))
        print(json.dumps(dict(
            mesh=os.path.basename(os.path.normpath(a.mesh)), rows=a.rows, cols=a.cols, smooth_sigma_cells=s,
            roughness_u_vox=rough["rms_2nd_diff_step_normalised_u_vox"],
            roughness_v_vox=rough["rms_2nd_diff_step_normalised_v_vox"],
            tiles=len(vals), contrast_median=round(float(np.median(vals)), 3) if vals else None,
            tiles_above_0_20=round(float(np.mean(np.array(vals) > 0.2)), 2) if vals else None,
            sheet_offset_vox=off, sheet_phase_concentration=conc)), flush=True)
        if a.png:
            from PIL import Image
            mid = stack[len(LAYERS) // 2]
            lo, hi = np.percentile(mid[m], [1, 99])
            Image.fromarray(((mid - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)).save(
                f"{a.png}_s{s:g}.png")


if __name__ == "__main__":
    main()
