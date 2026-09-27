"""Minimal stand-in for volume-cartographer's `vc` Python bindings, enough for the winding-model
slab extractor (vesuvius/neural_tracing/winding_models/volume_slab_extractor.py):

  vc.Volume.open(path)                     local zarr v2 array (or a multiscale group -> its "0")
  Volume.sample_planes(origins_xyz, x_steps_xyz, y_steps_xyz, (H, W), level=0, sampling, tile_size)
  Volume.sample_coords(coords_xyz, valid_mask, level=0, sampling, tile_size)
  vc.set_chunk_cache_budget(bytes), vc.set_chunk_cache_io_threads(n)

Chunks may be uncompressed or numcodecs-compressed, with "/" or "." separators; a missing chunk
reads as the fill value, like vc's "known missing" chunks. Coordinates are level-0 XYZ voxels.
A sample is valid when it lies inside the array (vc's blocking sampler reports every in-bounds
coordinate as covered); invalid samples are 0. Trilinear values are rounded to the nearest uint8.
Sampling runs with torch on the device named by $VC_SHIM_DEVICE (e.g. "cuda:0"), else on the CPU.
$VC_SHIM_ZRANGE="z0,z1" marks samples outside the downloaded slices [z0, z1) invalid as well.
Written for running the public winding model on Kaggle without building the C++ bindings.
"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict

import numpy as np

_BUDGET = [1 << 30]
_LOCK = threading.Lock()


def set_chunk_cache_budget(nbytes):
    _BUDGET[0] = max(int(nbytes), 256 << 20)


def set_chunk_cache_io_threads(n):  # reads are synchronous here
    return None


class Volume:
    @staticmethod
    def open(path):
        return Volume(str(path))

    def __init__(self, path):
        if not os.path.isfile(os.path.join(path, ".zarray")) and os.path.isfile(os.path.join(path, "0", ".zarray")):
            path = os.path.join(path, "0")
        meta = json.load(open(os.path.join(path, ".zarray")))
        self.path = path
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        self.dtype = np.dtype(meta["dtype"])
        if self.dtype != np.uint8:
            raise ValueError("only uint8 volumes are supported")
        self.sep = meta.get("dimension_separator", ".") or "."
        self.fill = int(meta.get("fill_value") or 0)
        self.codec = None
        if meta.get("compressor"):
            import numcodecs
            self.codec = numcodecs.get_codec(meta["compressor"])
        self.filters = [__import__("numcodecs").get_codec(f) for f in (meta.get("filters") or [])]
        self._cache = OrderedDict()
        self._cache_bytes = 0

    # --- chunk access -------------------------------------------------------------------------
    def _chunk(self, key):
        with _LOCK:
            a = self._cache.get(key)
            if a is not None:
                self._cache.move_to_end(key)
                return a
        p = os.path.join(self.path, self.sep.join(str(k) for k in key))
        if os.path.exists(p):
            raw = open(p, "rb").read()
            if self.codec is not None:
                raw = self.codec.decode(raw)
            for f in reversed(self.filters):
                raw = f.decode(raw)
            a = np.frombuffer(raw, np.uint8)
            if a.size != int(np.prod(self.chunks)):
                raise RuntimeError("decoded chunk byte size does not match full chunk shape: " + p)
            a = a.reshape(self.chunks)
        else:
            a = None  # known missing: fill value
        with _LOCK:
            nbytes = 0 if a is None else a.nbytes
            self._cache[key] = a
            self._cache_bytes += nbytes
            while self._cache_bytes > _BUDGET[0] and len(self._cache) > 1:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= 0 if old is None else old.nbytes
        return a

    def read_box(self, lo, hi):
        """uint8 array of the half-open box [lo, hi) (zyx), clamped to the array (outside = fill)."""
        lo = [int(v) for v in lo]
        hi = [int(v) for v in hi]
        out = np.full([h - l for l, h in zip(lo, hi)], self.fill, np.uint8)
        clo = [max(l, 0) for l in lo]
        chi = [min(h, s) for h, s in zip(hi, self.shape)]
        if any(a >= b for a, b in zip(clo, chi)):
            return out
        cs = self.chunks
        for cz in range(clo[0] // cs[0], (chi[0] - 1) // cs[0] + 1):
            for cy in range(clo[1] // cs[1], (chi[1] - 1) // cs[1] + 1):
                for cx in range(clo[2] // cs[2], (chi[2] - 1) // cs[2] + 1):
                    a = self._chunk((cz, cy, cx))
                    c0 = (cz * cs[0], cy * cs[1], cx * cs[2])
                    s = [max(clo[d], c0[d]) for d in range(3)]
                    e = [min(chi[d], c0[d] + cs[d]) for d in range(3)]
                    dst = tuple(slice(s[d] - lo[d], e[d] - lo[d]) for d in range(3))
                    if a is None:
                        out[dst] = self.fill
                    else:
                        out[dst] = a[tuple(slice(s[d] - c0[d], e[d] - c0[d]) for d in range(3))]
        return out

    # --- sampling -------------------------------------------------------------------------------
    def _sample_xyz(self, pts_xyz, sampling):
        """pts_xyz: float array [..., 3] of level-0 XYZ coordinates -> (uint8 values, bool valid)."""
        import torch
        import torch.nn.functional as F

        shape = pts_xyz.shape[:-1]
        p = np.asarray(pts_xyz, np.float64).reshape(-1, 3)[:, ::-1]  # zyx
        dims = np.array(self.shape, np.float64)
        valid = np.all((p >= 0) & (p <= dims - 1), axis=1)
        values = np.zeros(len(p), np.uint8)
        if valid.any():
            q = p[valid]
            lo = np.floor(q.min(0)).astype(np.int64)
            hi = np.floor(q.max(0)).astype(np.int64) + 2
            box = self.read_box(lo, hi)
            dev = os.environ.get("VC_SHIM_DEVICE") or "cpu"
            t = torch.from_numpy(box).to(dev).float()[None, None]
            size = torch.tensor([max(n - 1, 1) for n in box.shape[::-1]], dtype=torch.float32, device=dev)  # x, y, z
            g = torch.from_numpy((q - lo)[:, ::-1].astype(np.float32)).to(dev)  # xyz inside the box
            g = g / size * 2 - 1
            mode = "nearest" if sampling == "nearest" else "bilinear"
            out = F.grid_sample(t, g.view(1, -1, 1, 1, 3), mode=mode, padding_mode="border", align_corners=True)
            values[valid] = out.view(-1).add_(0.5).floor_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
        return values.reshape(shape), valid.reshape(shape)

    def sample_planes(self, origins_xyz, x_steps_xyz, y_steps_xyz, shape, level=0, sampling="nearest", tile_size=32):
        """Plane n, row y, column x samples origin[n] + y * y_step[n] + x * x_step[n] -> [N, H, W]."""
        import torch
        import torch.nn.functional as F

        if level != 0:
            raise ValueError("the shim samples level 0 only")
        o = np.asarray(origins_xyz, np.float64)
        xs = np.asarray(x_steps_xyz, np.float64)
        ys = np.asarray(y_steps_xyz, np.float64)
        n, h, w = len(o), int(shape[0]), int(shape[1])
        # each plane is affine in (y, x): its extreme coordinates are at its four corners
        corners = np.concatenate([o + a * (h - 1) * ys + b * (w - 1) * xs for a in (0, 1) for b in (0, 1)])[:, ::-1]
        dims = np.array(self.shape, np.float64)
        lo = np.clip(np.floor(corners.min(0)), 0, dims - 1).astype(np.int64)
        hi = np.clip(np.floor(corners.max(0)) + 2, 1, dims).astype(np.int64)
        dev = os.environ.get("VC_SHIM_DEVICE") or "cpu"
        f32 = dict(dtype=torch.float32, device=dev)
        lo_xyz = lo[::-1].astype(np.float64)
        ot = torch.as_tensor(o - lo_xyz, **f32)
        yt = torch.arange(h, **f32)[None, :, None, None] * torch.as_tensor(ys, **f32)[:, None, None, :]
        xt = torch.arange(w, **f32)[None, None, :, None] * torch.as_tensor(xs, **f32)[:, None, None, :]
        pts = ot[:, None, None, :] + yt + xt  # [N, H, W, 3] xyz relative to the box corner
        del yt, xt
        absolute = pts + torch.as_tensor(lo_xyz, **f32)
        upper = torch.as_tensor((dims - 1)[::-1].copy(), **f32)
        valid = ((absolute >= 0) & (absolute <= upper)).all(-1)
        zr = os.environ.get("VC_SHIM_ZRANGE")  # "z0,z1": only these slices were downloaded
        if zr:
            z0, z1 = (float(v) for v in zr.split(","))
            valid &= (absolute[..., 2] >= z0) & (absolute[..., 2] <= z1 - 1)
        del absolute
        values = torch.zeros((n, h, w), dtype=torch.uint8, device=dev)
        if bool(valid.any()) and np.all(hi > lo):
            box = self.read_box(lo, hi)
            t = torch.from_numpy(box).to(dev).float()[None, None]
            size = torch.as_tensor([max(v - 1, 1) for v in box.shape[::-1]], **f32)  # x, y, z extents
            g = (pts / size * 2 - 1).view(1, -1, 1, 1, 3)
            mode = "nearest" if sampling == "nearest" else "bilinear"
            out = F.grid_sample(t, g, mode=mode, padding_mode="border", align_corners=True).view(n, h, w)
            values = torch.where(valid, out.add_(0.5).floor_().clamp_(0, 255).to(torch.uint8), values)
        return values.cpu().numpy(), valid.to(torch.uint8).cpu().numpy(), {"shim": True}

    def sample_coords(self, coords_xyz, valid_mask, level=0, sampling="nearest", tile_size=32):
        if level != 0:
            raise ValueError("the shim samples level 0 only")
        c = np.asarray(coords_xyz, np.float64)
        values, valid = self._sample_xyz(c, sampling)
        valid &= np.asarray(valid_mask, bool)
        values[~valid] = 0
        return values, valid.astype(np.uint8), {"shim": True}
