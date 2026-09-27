"""Roughness and parameterisation check for tifxyz meshes (no data download needed).

For each mesh: RMS second difference of the vertex grid projected on the surface normal
(along the grid's u and v directions, in voxels) and the spread of the grid step length.
A smooth, isometric mesh at scale 0.05 has ~20-voxel steps with a tight spread and a second
difference of about 1-2 voxels; large values mean a bumpy surface or a stretched grid.

Usage: python mesh_roughness.py <tifxyz_dir> [<tifxyz_dir> ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import tifffile


def load_tifxyz(d):
    x = tifffile.imread(os.path.join(d, "x.tif")).astype(np.float64)
    y = tifffile.imread(os.path.join(d, "y.tif")).astype(np.float64)
    z = tifffile.imread(os.path.join(d, "z.tif")).astype(np.float64)
    valid = (x >= 0) & (y >= 0) & (z >= 0)
    meta = {}
    if os.path.exists(os.path.join(d, "meta.json")):
        meta = json.load(open(os.path.join(d, "meta.json")))
    return np.stack([z, y, x], -1), valid, meta


def roughness(d):
    P, valid, meta = load_tifxyz(d)
    return roughness_grid(P, valid, meta, name=os.path.basename(os.path.normpath(d)))


def roughness_grid(P, valid, meta, name=""):
    """Roughness of a vertex grid P [rows, cols, 3] (zyx) with validity mask; see module docstring."""
    dcol = np.zeros_like(P)
    drow = np.zeros_like(P)
    dcol[:, 1:-1] = P[:, 2:] - P[:, :-2]
    drow[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(drow, dcol)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    ok = valid.copy()
    ok[[0, -1], :] = False
    ok[:, [0, -1]] = False
    ok[1:-1, 1:-1] &= valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
    sd_u = ((P[:, 2:] - 2 * P[:, 1:-1] + P[:, :-2]) * n[:, 1:-1]).sum(-1)
    sd_v = ((P[2:] - 2 * P[1:-1] + P[:-2]) * n[1:-1]).sum(-1)
    # second differences grow with the square of the local step: rescale to the nominal step so that
    # a stretched grid is not counted as extra roughness
    sc = (meta.get("scale") or [None])[0]
    nominal = 1 / sc if sc else 20.0
    hu = 0.5 * (np.linalg.norm(P[:, 2:] - P[:, 1:-1], axis=-1) + np.linalg.norm(P[:, 1:-1] - P[:, :-2], axis=-1))
    hv = 0.5 * (np.linalg.norm(P[2:] - P[1:-1], axis=-1) + np.linalg.norm(P[1:-1] - P[:-2], axis=-1))
    sd_u_n = (sd_u * (nominal / np.maximum(hu, 1e-6)) ** 2)[ok[:, 1:-1]]
    sd_v_n = (sd_v * (nominal / np.maximum(hv, 1e-6)) ** 2)[ok[1:-1]]
    sd_u = sd_u[ok[:, 1:-1]]
    sd_v = sd_v[ok[1:-1]]
    step = np.linalg.norm(np.diff(P, axis=1), axis=-1)[valid[:, 1:] & valid[:, :-1]]
    return {
        "mesh": name,
        "grid": list(valid.shape),
        "nominal_step_vox": round(1 / sc, 2) if sc else None,
        "step_median_vox": round(float(np.median(step)), 2),
        "step_p5_p95_vox": [round(float(np.percentile(step, 5)), 1), round(float(np.percentile(step, 95)), 1)],
        "rms_2nd_diff_normal_u_vox": round(float(np.sqrt(np.mean(sd_u ** 2))), 2),
        "rms_2nd_diff_normal_v_vox": round(float(np.sqrt(np.mean(sd_v ** 2))), 2),
        "rms_2nd_diff_step_normalised_u_vox": round(float(np.sqrt(np.mean(sd_u_n ** 2))), 2),
        "rms_2nd_diff_step_normalised_v_vox": round(float(np.sqrt(np.mean(sd_v_n ** 2))), 2),
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(roughness(d)))
