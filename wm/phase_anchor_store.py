"""Anchor the crossings of a winding-model crossing store on the CT sheets (writes a new store).

villa's exporter (export_spiral_supervision.decode_center_ray) decodes a ray's crossings as the integer passages of
the model's phase registered at the ray's anchor, the seed point. The winding model predicts a relative phase with
one free offset per slab ("the whole phase field shares one free offset, absorbed by the shift-invariant loss and the
consumer's per-ray registration", winding_model.py), so the crossings sit at the seed's phase: villa seeds on verified
sheets, and its crossings are on sheets; seeded on a fit's windings, they sit wherever those windings do.

This tool moves each ray's crossings, all by the same fraction f of their local gap, to where the CT is brightest:
along the ray the level-0 CT is sampled every 0.5 voxel and smoothed (sigma 1 voxel); for f in [-0.5, 0.5) every
crossing k is tried at t_k + f g_k (g_k = the mean of its two gaps) and the f with the highest mean intensity wins, so
the ~15 crossings of a ray vote together. Rays with fewer than 3 crossings inside the scan, or whose best offset is not
clearly brighter than the worst (score range below --min-contrast of the ray's p95 - p5), are left as they are.
Nothing else changes: levels, rays, origins. Arrays, checksums and the manifest fingerprint are rewritten as villa's
exporter writes them, and manifest["phase_anchor"] records the parameters and statistics, including a check on
crossings not used to choose the offset (f chosen on the even crossings of a ray, c of sheet_hits.py on the odd ones,
before and after).

Usage: python phase_anchor_store.py <store dir> <level-0 CT: local zarr v2 dir> <out dir> [--min-contrast 0.05]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import vc  # noqa: E402  (local zarr v2 reader, chunk cache)

FS = np.arange(-0.5, 0.5, 0.05)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Sampler:
    """Trilinear samples of a local uint8 zarr (vc.Volume chunk cache) at zyx points; 0 outside the array."""

    def __init__(self, vol):
        self.v = vol
        self.cs = np.array(vol.chunks)
        self.shape = np.array(vol.shape)

    def sample(self, pts):
        p0 = np.floor(pts).astype(np.int64)
        inside = np.all((p0 >= 0) & (p0 < self.shape - 1), axis=1)
        p0 = np.clip(p0, 0, self.shape - 2)
        f = np.clip(pts - p0, 0, 1)
        acc = np.zeros(len(pts), np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    q = p0 + (dz, dy, dx)
                    ci = q // self.cs
                    li = q - ci * self.cs
                    v = np.zeros(len(q), np.float32)
                    keys, inv = np.unique(ci, axis=0, return_inverse=True)
                    for n, k in enumerate(keys):
                        ch = self.v._chunk(tuple(int(x) for x in k))
                        if ch is None:
                            continue
                        idx = np.flatnonzero(inv.ravel() == n)
                        l_ = li[idx]
                        v[idx] = ch[l_[:, 0], l_[:, 1], l_[:, 2]]
                    w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
                    acc += w * v
        return np.where(inside, acc, 0.0)


def contrast(prof, ts, inside, t):
    """sheet_hits.py's c at the interior positions of the sorted t (NaN where a point or mid-point is outside)."""
    if len(t) < 3:
        return np.zeros(0)
    rng = np.percentile(prof[inside], 95) - np.percentile(prof[inside], 5)
    if rng <= 0:
        return np.zeros(0)
    mids = 0.5 * (t[1:] + t[:-1])
    ins = np.interp(t, ts, inside.astype(float)) > 0.99
    insm = np.interp(mids, ts, inside.astype(float)) > 0.99
    keep = ins[1:-1] & insm[:-1] & insm[1:]
    c = (np.interp(t[1:-1], ts, prof) - 0.5 * (np.interp(mids[:-1], ts, prof) + np.interp(mids[1:], ts, prof))) / rng
    return c[keep]


def best_offset(prof, ts, inside, t, g):
    """(f, score range / (p95 - p5)) maximising the mean intensity at t + f g over the points inside the scan."""
    ok = np.interp(t, ts, inside.astype(float)) > 0.99
    if ok.sum() < 3:
        return None, 0.0
    rng = np.percentile(prof[inside], 95) - np.percentile(prof[inside], 5)
    if rng <= 0:
        return None, 0.0
    scores = np.array([np.interp(t[ok] + f * g[ok], ts, prof).mean() for f in FS])
    return float(FS[int(np.argmax(scores))]), float((scores.max() - scores.min()) / rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store")
    ap.add_argument("volume", help="local level-0 zarr v2 directory (or a multiscale group with 0/)")
    ap.add_argument("out")
    ap.add_argument("--min-contrast", type=float, default=0.05)
    ap.add_argument("--sigma", type=float, default=1.0, help="smoothing of the CT profile (voxels)")
    a = ap.parse_args()
    t_start = time.time()
    samp = Sampler(vc.Volume.open(a.volume))
    man = json.load(open(os.path.join(a.store, "manifest.json")))
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.store, a.out)
    stats = {"rays": 0, "shifted": 0, "too_few_crossings": 0, "low_contrast": 0, "offsets": [],
             "cv_c_before": [], "cv_c_after": [], "all_c_before": [], "all_c_after": []}
    for shard in man["shards"]:
        d = os.path.join(a.out, shard["name"])
        arr = {k: np.load(os.path.join(d, v["file"])) for k, v in shard["arrays"].items()}
        O = arr["ray_origin_zyx"].astype(np.float64)
        S = arr["ray_step_zyx"].astype(np.float64)
        T = arr["crossing_t"].astype(np.float64).copy()
        off = arr["crossing_offsets"]
        order = np.lexsort((O[:, 2], O[:, 1], O[:, 0]))  # spatial order keeps the chunk cache warm
        for i in order:
            stats["rays"] += 1
            L = float(np.linalg.norm(S[i]))
            if L <= 0 or off[i + 1] - off[i] < 3:
                stats["too_few_crossings"] += 1
                continue
            u = S[i] / L
            t = T[off[i]:off[i + 1]] * L  # voxels from the origin, ascending
            ts = np.arange(t.min() - 8.0, t.max() + 8.0, 0.5)
            raw = samp.sample(O[i][None] + ts[:, None] * u[None])
            inside = raw > 0
            if inside.sum() < 20:
                stats["too_few_crossings"] += 1
                continue
            prof = gaussian_filter1d(raw, a.sigma / 0.5)
            gaps = np.diff(t)
            g = np.r_[gaps[0], 0.5 * (gaps[:-1] + gaps[1:]), gaps[-1]]
            f, con = best_offset(prof, ts, inside, t, g)
            if f is None:
                stats["too_few_crossings"] += 1
                continue
            # held-out check: offset chosen on the even crossings, scored on the odd ones
            if len(t) >= 6:
                fe, _ = best_offset(prof, ts, inside, t[0::2], g[0::2])
                if fe is not None:
                    stats["cv_c_before"] += contrast(prof, ts, inside, t[1::2]).tolist()
                    stats["cv_c_after"] += contrast(prof, ts, inside, t[1::2] + fe * g[1::2]).tolist()
            stats["all_c_before"] += contrast(prof, ts, inside, t).tolist()
            if con < a.min_contrast:
                stats["low_contrast"] += 1
                stats["all_c_after"] += contrast(prof, ts, inside, t).tolist()
                continue
            t_new = t + f * g
            stats["all_c_after"] += contrast(prof, ts, inside, t_new).tolist()
            T[off[i]:off[i + 1]] = t_new / L
            stats["shifted"] += 1
            stats["offsets"].append(f)
            if stats["rays"] % 2000 == 0:
                print(f"{stats['rays']} rays, {stats['shifted']} shifted, {time.time() - t_start:.0f} s", flush=True)
        path = os.path.join(d, shard["arrays"]["crossing_t"]["file"])
        np.save(path, np.ascontiguousarray(T.astype(arr["crossing_t"].dtype)), allow_pickle=False)
        shard["arrays"]["crossing_t"]["bytes"] = os.path.getsize(path)
        shard["arrays"]["crossing_t"]["sha256"] = file_digest(path)

    def frac_pos(c):
        c = np.asarray(c)
        return round(float((c > 0).mean()), 4) if len(c) else None

    offs = np.asarray(stats["offsets"])
    summary = {"tool": "phase_anchor_store.py", "min_contrast": a.min_contrast, "sigma_vox": a.sigma,
               "offsets_tried": [round(float(f), 2) for f in FS], "rays": stats["rays"], "shifted": stats["shifted"],
               "too_few_crossings": stats["too_few_crossings"], "low_contrast": stats["low_contrast"],
               "offset_hist_-0.5_to_0.5_by_0.1": np.histogram(offs, bins=10, range=(-0.5, 0.5))[0].tolist() if len(offs) else [],
               "frac_c_pos_before": frac_pos(stats["all_c_before"]), "frac_c_pos_after": frac_pos(stats["all_c_after"]),
               "heldout_crossings_frac_c_pos_before": frac_pos(stats["cv_c_before"]),
               "heldout_crossings_frac_c_pos_after": frac_pos(stats["cv_c_after"]),
               "heldout_crossings_n": len(stats["cv_c_after"]), "seconds": round(time.time() - t_start, 1)}
    man["phase_anchor"] = summary
    identity = copy.deepcopy(man)
    for k in ("fingerprint", "elapsed_seconds", "export_workers", "rays_per_task"):
        identity.pop(k, None)
    for s in identity["shards"]:
        s.pop("elapsed_seconds", None)
    man["fingerprint"] = canonical_digest(identity)
    open(os.path.join(a.out, "manifest.json"), "w").write(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print("PHASE_ANCHOR " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
