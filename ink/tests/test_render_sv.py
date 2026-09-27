"""Checks for render_sv.py and inkjob.py.

  python -m pytest ink/tests -q                  (unit tests, offline)
  RENDER_SV_NETWORK=1 python -m pytest ink/tests  (also the comparison with a published surface volume, ~30 MB download)
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
for p in (os.path.join(os.path.dirname(HERE), "..", "tools"), os.path.join(os.path.dirname(HERE), "..", "release", "tools")):
    if os.path.isdir(p):
        sys.path.insert(0, os.path.abspath(p))

import render_sv  # noqa: E402

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


class ArrayVolume:
    """An in-memory stand-in for a zarr level (read box by box, zeros outside)."""

    def __init__(self, a):
        self.a = a
        self.shape = a.shape

    def read(self, z0, z1, y0, y1, x0, x1):
        lo = [max(0, v) for v in (z0, y0, x0)]
        hi = [min(s, v) for s, v in zip(self.shape, (z1, y1, x1))]
        return self.a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()


def write_mesh(d, X, Y, Z, step=20):
    import json

    import tifffile
    os.makedirs(d, exist_ok=True)
    for c, a in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(os.path.join(d, f"{c}.tif"), a.astype(np.float32))
    json.dump({"format": "tifxyz", "scale": [1.0 / step, 1.0 / step], "type": "seg", "uuid": "t"},
              open(os.path.join(d, "meta.json"), "w"))


def test_plane_render_layers(tmp_path):
    """A flat mesh at z = 40 over a volume whose value is the z index: layer L must read z = 40 +/- (L - 13)."""
    vol = np.broadcast_to(np.arange(96, dtype=np.uint8)[:, None, None], (96, 128, 128)).copy()
    rows, cols = np.meshgrid(np.arange(4) * 20.0 + 10, np.arange(5) * 20.0 + 10, indexing="ij")
    write_mesh(tmp_path / "m", cols, rows, np.full_like(rows, 40.0))  # x = cols, y = rows, z = 40
    r = render_sv.Renderer(str(tmp_path / "m"), ArrayVolume(vol), "cpu")
    sv = r.render()
    assert sv.shape == (28, 61, 81)
    centre = sv[:, 30, 40].astype(int)
    # the normal of this grid is +z or -z; either way the layers step by one voxel through z = 40 at layer 13
    assert centre[13] == 40
    assert set(np.diff(centre)) <= {1} or set(np.diff(centre)) <= {-1}


def test_holes_stay_empty(tmp_path):
    vol = np.full((64, 96, 96), 100, np.uint8)
    rows, cols = np.meshgrid(np.arange(4) * 20.0 + 5, np.arange(4) * 20.0 + 5, indexing="ij")
    X, Y, Z = cols.copy(), rows.copy(), np.full_like(rows, 30.0)
    X[1, 1] = Y[1, 1] = Z[1, 1] = -1  # a hole: every canvas pixel touching grid node (1, 1) is invalid
    write_mesh(tmp_path / "h", X, Y, Z)
    sv = render_sv.Renderer(str(tmp_path / "h"), ArrayVolume(vol), "cpu").render()
    assert sv[13, 20, 20] == 0 and sv[13, 50, 50] == 100


def test_ome_zarr_roundtrip(tmp_path):
    zarr = pytest.importorskip("zarr")
    a = (np.random.default_rng(0).random((28, 150, 260)) * 255).astype(np.uint8)
    render_sv.write_ome_zarr(str(tmp_path / "sv.zarr"), a)
    b = zarr.open(str(tmp_path / "sv.zarr"), mode="r")["0"][:]
    assert (a == b).all()


def test_auc_matches_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    import inkjob
    rng = np.random.default_rng(1)
    y = rng.random(5000) < 0.3
    s = np.round(rng.random(5000) * 10 + y * 3)  # many ties
    assert abs(inkjob.auc(s, y) - metrics.roc_auc_score(y, s)) < 1e-12


@pytest.mark.skipif(not os.environ.get("RENDER_SV_NETWORK"), reason="set RENDER_SV_NETWORK=1 to download")
def test_matches_published_surface_volume(tmp_path):
    """Our render of a curated PHerc0139 segment vs its published surface volume: r > 0.85 on every layer."""
    import requests
    from vcz import ZArray
    seg = f"{S3}/PHerc0139/segments/20260317000000-w035_2026031718"
    d = tmp_path / "w035"
    os.makedirs(d)
    for f in ["meta.json", "x.tif", "y.tif", "z.tif"]:
        open(d / f, "wb").write(requests.get(f"{seg}/mesh/20260317000000-on-20250728140407-9.362um.tifxyz/{f}", timeout=300).content)
    r0, c0, n = 700, 1100, 256
    pub = ZArray(f"{seg}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr/0").read(0, 28, r0, r0 + n, c0, c0 + n)
    vol = render_sv.open_volume(f"{S3}/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0")
    ours = render_sv.Renderer(str(d), vol, "cpu").render((r0, r0 + n), (c0, c0 + n))
    for L in range(28):
        a = pub[L].astype(float) - pub[L].mean()
        b = ours[L].astype(float) - ours[L].mean()
        assert (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()) > 0.85, L
