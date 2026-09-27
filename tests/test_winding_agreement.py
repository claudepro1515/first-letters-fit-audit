"""Synthetic check of tools/winding_agreement.py: a crossing store whose rays cross cylindrical sheets 16 voxels
apart, and three 'fits': one winding per sheet, two per sheet, and one per sheet shifted half a spacing.
Expected: exactly one winding between consecutive crossings for the first and the third (the count is blind to
phase, which is why tools/sheet_hits.py exists), two for the second.

  python -m pytest tests/test_winding_agreement.py -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
tifffile = pytest.importorskip("tifffile")
CY = CX = 400.0


def write_store(root, n_rays=40, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "s0"), exist_ok=True)
    O, S, T, L, off = [], [], [], [], [0]
    for _ in range(n_rays):
        z = rng.uniform(20, 80)
        th = rng.uniform(0, 2 * np.pi)
        d = np.array([0.0, np.sin(th), np.cos(th)])
        r0 = 40.0
        O.append(np.array([z, CY, CX]) + r0 * d)
        S.append(d * 0.5)  # step of half a voxel: crossing_t is in steps
        radii = 16.0 * np.arange(3, 12)
        T.append((radii - r0) / 0.5)
        L.append(np.arange(len(radii)))
        off.append(off[-1] + len(radii))
    arrays = {"ray_origin_zyx": np.array(O, np.float32), "ray_step_zyx": np.array(S, np.float32),
              "crossing_t": np.concatenate(T).astype(np.float32), "crossing_level": np.concatenate(L).astype(np.int32),
              "crossing_offsets": np.array(off, np.int64)}
    meta = {}
    for k, v in arrays.items():
        np.save(os.path.join(root, "s0", k + ".npy"), v)
        meta[k] = {"file": k + ".npy"}
    json.dump({"shards": [{"name": "s0", "arrays": meta}]}, open(os.path.join(root, "manifest.json"), "w"))


def write_cylinders(d, radii):
    for k, rad in enumerate(radii):
        m = os.path.join(d, f"w{k + 10:03d}")
        os.makedirs(m, exist_ok=True)
        th = np.linspace(0, 2 * np.pi, max(24, int(2 * np.pi * rad / 6)), endpoint=True)
        T, Z = np.meshgrid(th, np.arange(-8.0, 110.0, 6.0))
        for c, a in zip("xyz", (CX + rad * np.cos(T), CY + rad * np.sin(T), Z)):
            tifffile.imwrite(os.path.join(m, f"{c}.tif"), np.where(Z > 0, a, -1).astype(np.float32))


def run(tmp_path, radii, name):
    d = tmp_path / name
    write_cylinders(str(d), radii)
    out = subprocess.run([sys.executable, os.path.join(TOOLS, "winding_agreement.py"), str(d), str(tmp_path / "store"),
                          "--z", "10", "90"], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_agreement_counts(tmp_path):
    write_store(str(tmp_path / "store"))
    one = run(tmp_path, [16.0 * k for k in range(2, 13)], "one")
    two = run(tmp_path, [8.0 * k for k in range(4, 26)], "two")
    shifted = run(tmp_path, [16.0 * k + 8 for k in range(2, 13)], "shifted")
    assert one["frac_exact_d1"] > 0.95
    assert two["count_d1_distribution"]["2"] > 0.95
    assert shifted["frac_exact_d1"] > 0.95  # phase-blind by construction
    assert one["median_crossing_to_nearest_winding_vox"] < 1 and shifted["median_crossing_to_nearest_winding_vox"] > 6
