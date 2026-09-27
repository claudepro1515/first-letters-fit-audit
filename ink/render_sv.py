"""Render the 28-layer surface volume of a tifxyz mesh (published Vesuvius convention) with torch, GPU or CPU.

Convention (checked against published surface volumes, see render_check.py): the mesh grid is interpolated
bilinearly to one canvas pixel per voxel (canvas pixel (i, j) = grid coordinate (i, j) * scale), the normal is
n = normalize(cross(dR, dC)) with dR, dC the derivatives of the grid along its rows and columns, and layer L
(0..27) is the CT sampled trilinearly at surface + (L - 13) * n. Canvas pixels whose four grid neighbours are
not all valid (tifxyz marks holes with -1) are left at 0.

The CT comes from a zarr v2 array read box by box: the public S3 bucket through vcz.py (chunk cache on disk)
or a local directory. Blocks whose bounding box is too large (folded meshes) are split recursively.

Usage:
  python render_sv.py <tifxyz_dir> <out.zarr> --volume <zarr level url or dir> [--cache DIR]
         [--rows r0:r1] [--cols c0:c1] [--device cuda] [--png prefix]
Writes an OME-Zarr (28, H, W) uint8 (chunks 28 x 128 x 128, blosc lz4) that koine_machines' infer.py reads,
and prints one JSON line (canvas size, valid fraction, timings, tile profile contrast).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

OFFSETS = np.arange(28) - 13  # layer L = surface + (L - 13) * n
MAX_BOX = 300_000_000  # voxels per CT box before a block is split


class LocalZarr:
    """A zarr v2 array in a local directory (uncompressed or any numcodecs compressor), read box by box."""

    def __init__(self, path):
        from numcodecs import get_codec
        self.path = path.rstrip("/")
        meta = json.load(open(self.path + "/.zarray"))
        self.shape, self.chunks = tuple(meta["shape"]), tuple(meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        self.sep = meta.get("dimension_separator", ".")
        self.codec = get_codec(meta["compressor"]) if meta.get("compressor") else None
        self.fill = meta.get("fill_value") or 0

    def read(self, z0, z1, y0, y1, x0, x1):
        lo = [max(0, v) for v in (z0, y0, x0)]
        hi = [min(s, v) for s, v in zip(self.shape, (z1, y1, x1))]
        out = np.full([max(0, h - l) for l, h in zip(lo, hi)], self.fill, self.dtype)
        if out.size == 0:
            return out
        rng = [range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, self.chunks)]
        for a in rng[0]:
            for b in rng[1]:
                for c in rng[2]:
                    fn = f"{self.path}/{self.sep.join(map(str, (a, b, c)))}"
                    if not os.path.exists(fn):
                        continue
                    raw = open(fn, "rb").read()
                    buf = self.codec.decode(raw) if self.codec else raw
                    ch = np.frombuffer(buf, self.dtype).reshape(self.chunks)
                    so, sc = [], []
                    for d, i in enumerate((a, b, c)):
                        c0 = i * self.chunks[d]
                        s, e = max(lo[d], c0), min(hi[d], c0 + self.chunks[d])
                        so.append(slice(s - lo[d], e - lo[d]))
                        sc.append(slice(s - c0, e - c0))
                    out[tuple(so)] = ch[tuple(sc)]
        return out


def open_volume(spec, cache=None):
    if spec.startswith("http"):
        from vcz import ZArray
        return ZArray(spec, cache_dir=cache, workers=48)
    return LocalZarr(spec)


def load_tifxyz(d):
    X, Y, Z = (tifffile.imread(os.path.join(d, f"{c}.tif")).astype(np.float64) for c in "xyz")
    meta = json.load(open(os.path.join(d, "meta.json")))
    step = 1.0 / float(meta["scale"][0])
    valid = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z) & (X > -0.5) & (Y > -0.5) & (Z > -0.5)
    return np.stack([X, Y, Z]), valid, step


def interp(G, rr, cc):
    """Bilinear interpolation of G (C, R, Cols) at grid coordinates rr, cc (same shape) -> (..., C)."""
    R, C = G.shape[-2:]
    gx = cc / max(C - 1, 1) * 2 - 1
    gy = rr / max(R - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
    out = F.grid_sample(G[None], grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out[0, :, 0].T.reshape(*rr.shape, G.shape[0])


class Renderer:
    def __init__(self, mesh_dir, volume, device="cpu"):
        P, valid, self.step = load_tifxyz(mesh_dir)
        self.dev = torch.device(device)
        # grid coordinates are small; keep xyz in float64 on CPU for the bbox and float32 on the device
        self.G = torch.from_numpy(P).to(self.dev, torch.float64)
        self.V = torch.from_numpy(valid.astype(np.float64))[None].to(self.dev)
        self.R, self.C = valid.shape
        self.H = int(round((self.R - 1) * self.step)) + 1
        self.W = int(round((self.C - 1) * self.step)) + 1
        self.vol = volume
        self.offs = torch.from_numpy(OFFSETS.astype(np.float64)).to(self.dev)
        self.t_read = self.t_gpu = 0.0
        self.boxes = 0

    def surface(self, i0, i1, j0, j1):
        ii = torch.arange(i0, i1, device=self.dev, dtype=torch.float64) / self.step
        jj = torch.arange(j0, j1, device=self.dev, dtype=torch.float64) / self.step
        rr, cc = torch.meshgrid(ii, jj, indexing="ij")
        P = interp(self.G, rr, cc)
        dR = interp(self.G, rr + 0.5, cc) - interp(self.G, rr - 0.5, cc)
        dC = interp(self.G, rr, cc + 0.5) - interp(self.G, rr, cc - 0.5)
        n = torch.linalg.cross(dR, dC)
        ln = torch.linalg.norm(n, dim=-1, keepdim=True)
        ok = (interp(self.V, rr, cc)[..., 0] > 0.999) & (ln[..., 0] > 1e-9)
        return P, n / ln.clamp_min(1e-12), ok

    def block(self, i0, i1, j0, j1, out):
        P, n, ok = self.surface(i0, i1, j0, j1)
        if not bool(ok.any()):
            return
        pts = P[None] + self.offs[:, None, None, None] * n[None]  # (28, h, w, 3) xyz
        sel = pts[:, ok]
        lo = torch.floor(sel.amin((0, 1))).long() - 1
        hi = torch.ceil(sel.amax((0, 1))).long() + 2
        size = (hi - lo).prod().item()
        if size > MAX_BOX and (i1 - i0 > 16 or j1 - j0 > 16):
            if i1 - i0 >= j1 - j0:
                m = (i0 + i1) // 2
                self.block(i0, m, j0, j1, out); self.block(m, i1, j0, j1, out)
            else:
                m = (j0 + j1) // 2
                self.block(i0, i1, j0, m, out); self.block(i0, i1, m, j1, out)
            return
        x0, y0, z0 = (int(v) for v in lo.tolist())
        x1, y1, z1 = (int(v) for v in hi.tolist())
        t = time.time()
        box = self.vol.read(z0, z1, y0, y1, x0, x1)
        self.t_read += time.time() - t
        self.boxes += 1
        t = time.time()
        # read() clips to the volume; place the clipped box inside the requested one (zeros outside)
        full = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        cz, cy, cx = max(0, -z0), max(0, -y0), max(0, -x0)
        full[cz:cz + box.shape[0], cy:cy + box.shape[1], cx:cx + box.shape[2]] = box
        B = torch.from_numpy(full).to(self.dev).float()[None, None]
        D_, H_, W_ = full.shape
        g = torch.stack([(pts[..., 0] - x0) / max(W_ - 1, 1), (pts[..., 1] - y0) / max(H_ - 1, 1),
                         (pts[..., 2] - z0) / max(D_ - 1, 1)], -1) * 2 - 1
        v = F.grid_sample(B, g[None].float(), mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        v = torch.where(ok[None], v, torch.zeros_like(v)).round().clamp(0, 255).to(torch.uint8)
        out[:, i0:i1, j0:j1] = v.cpu().numpy()
        self.t_gpu += time.time() - t

    def render(self, rows=None, cols=None, block=256):
        i0, i1 = rows or (0, self.H)
        j0, j1 = cols or (0, self.W)
        out = np.zeros((28, i1 - i0, j1 - j0), np.uint8)
        view = _Offset(out, i0, j0)
        for a in range(i0, i1, 1024):
            for b in range(j0, j1, block):
                self.block(a, min(i1, a + 1024), b, min(j1, b + block), view)
        return out


class _Offset:
    """out[:, i0:i1, j0:j1] = v on a canvas crop that starts at (i0, j0)."""

    def __init__(self, arr, i0, j0):
        self.arr, self.i0, self.j0 = arr, i0, j0

    def __setitem__(self, key, value):
        _, si, sj = key
        self.arr[:, si.start - self.i0:si.stop - self.i0, sj.start - self.j0:sj.stop - self.j0] = value


def write_ome_zarr(path, arr, chunk=128):
    """OME-Zarr v2 group with one level '0' of shape (28, H, W); written by hand (works with zarr 2 or 3 readers)."""
    from numcodecs import Blosc
    os.makedirs(os.path.join(path, "0"), exist_ok=True)
    json.dump({"zarr_format": 2}, open(os.path.join(path, ".zgroup"), "w"))
    json.dump({"multiscales": [{"version": "0.4", "axes": [{"name": a, "type": "space"} for a in "zyx"],
                                "datasets": [{"path": "0", "coordinateTransformations": [
                                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]},
              open(os.path.join(path, ".zattrs"), "w"))
    D, H, W = arr.shape
    json.dump({"shape": [D, H, W], "chunks": [D, chunk, chunk], "dtype": "|u1", "fill_value": 0, "order": "C",
               "filters": None, "dimension_separator": ".", "zarr_format": 2,
               "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0}},
              open(os.path.join(path, "0", ".zarray"), "w"))
    codec = Blosc(cname="lz4", clevel=5, shuffle=1)
    for y in range(0, H, chunk):
        for x in range(0, W, chunk):
            blk = arr[:, y:y + chunk, x:x + chunk]
            if not blk.any():
                continue
            c = np.zeros((D, chunk, chunk), np.uint8)
            c[:, :blk.shape[1], :blk.shape[2]] = blk
            open(os.path.join(path, "0", f"0.{y // chunk}.{x // chunk}"), "wb").write(codec.encode(c))


def tile_contrast(stack, tile=128, min_valid=0.9):
    """(max - min) / (max + min) of each tile's mean layer profile (as release/tools/render_contrast.py)."""
    vals = []
    m = stack[13] > 0
    for y in range(0, stack.shape[1] - tile + 1, tile):
        for x in range(0, stack.shape[2] - tile + 1, tile):
            mm = m[y:y + tile, x:x + tile]
            if mm.mean() < min_valid:
                continue
            prof = stack[:, y:y + tile, x:x + tile][:, mm].mean(1)
            vals.append(float((prof.max() - prof.min()) / (prof.max() + prof.min() + 1e-9)))
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh")
    ap.add_argument("out")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--rows", default=None)
    ap.add_argument("--cols", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--png", default=None, help="write <prefix>_layer13.png (and a 4x smaller preview)")
    a = ap.parse_args()
    t0 = time.time()
    r = Renderer(a.mesh, open_volume(a.volume, a.cache), a.device)
    rows = tuple(int(v) for v in a.rows.split(":")) if a.rows else None
    cols = tuple(int(v) for v in a.cols.split(":")) if a.cols else None
    sv = r.render(rows, cols)
    write_ome_zarr(a.out, sv)
    tc = tile_contrast(sv)
    res = {"mesh": a.mesh, "canvas": [r.H, r.W], "rendered": list(sv.shape), "step": r.step,
           "valid_frac": round(float((sv[13] > 0).mean()), 4), "boxes": r.boxes,
           "t_read_s": round(r.t_read, 1), "t_sample_s": round(r.t_gpu, 1), "t_total_s": round(time.time() - t0, 1),
           "tiles": len(tc), "contrast_median": round(float(np.median(tc)), 4) if tc else None}
    if a.png:
        from PIL import Image
        L = sv[13]
        nz = L[L > 0]
        lo, hi = (np.percentile(nz, [1, 99]) if nz.size else (0, 255))
        img = ((L.astype(np.float32) - lo) / max(hi - lo, 1) * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(a.png + "_layer13.png")
        res["png"] = a.png + "_layer13.png"
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
