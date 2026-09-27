"""phase_align.py on synthetic volumes: planar sheets every 14 voxels (crests at x = 14 k), in both modes: a winding
between two sheets moves onto one of them, a winding on a sheet stays, and the mesh keeps its grid and validity mask.
Sheets with an irregular spacing: the crest mode moves a winding onto a real sheet, not a phantom crest one period away."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "phase_align.py")
PERIOD = 14.0


def write_volume(d, shape=(192, 192, 192), chunk=64):
    """zarr v2 (uncompressed, '/' separator): intensity 100 + 80 cos(2 pi x / 14), x = last axis."""
    os.makedirs(d, exist_ok=True)
    json.dump({"zarr_format": 2, "shape": list(shape), "chunks": [chunk] * 3, "dtype": "|u1", "compressor": None,
               "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"}, open(f"{d}/.zarray", "w"))
    x = np.arange(shape[2])
    row = (100 + 80 * np.cos(2 * np.pi * x / PERIOD)).astype(np.uint8)
    for cz in range(shape[0] // chunk):
        for cy in range(shape[1] // chunk):
            for cx in range(shape[2] // chunk):
                block = np.broadcast_to(row[cx * chunk:(cx + 1) * chunk], (chunk, chunk, chunk)).copy()
                os.makedirs(f"{d}/{cz}/{cy}", exist_ok=True)
                block.tofile(f"{d}/{cz}/{cy}/{cx}")


def write_plane(d, x0, step=4.0, n=24, y0=40.0, ny=None):
    """A tifxyz mesh: the surface x = x0(y) (x0 a number or a function of y) over y in [y0, y0 + ny step) and z in
    [40, 40 + n step), rows along z, columns along y."""
    os.makedirs(d, exist_ok=True)
    zz, yy = np.meshgrid(40 + step * np.arange(n), y0 + step * np.arange(ny or n), indexing="ij")
    xx = (x0(yy) if callable(x0) else np.full_like(zz, x0)).astype(np.float64)
    xx[0, 0] = yy[0, 0] = zz[0, 0] = -1  # one invalid vertex: must stay invalid
    for c, a in zip("xyz", (xx, yy, zz)):
        tifffile.imwrite(f"{d}/{c}.tif", a.astype(np.float32))
    json.dump({"format": "tifxyz", "scale": [1 / step, 1 / step], "type": "seg", "uuid": os.path.basename(d)}, open(f"{d}/meta.json", "w"))


def run(tmp_path, x0, extra=(), mode="crest"):
    vol, mesh, out = str(tmp_path / "vol"), str(tmp_path / "w001"), str(tmp_path / "out")
    write_volume(vol)
    write_plane(mesh, x0)
    p = subprocess.run([sys.executable, TOOL, mesh, vol, out, "--period", str(PERIOD), "--tile", "6", "--mode", mode, *extra], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout.strip().splitlines()[-1])
    x = tifffile.imread(out + "/x.tif")
    return res, x


@pytest.mark.parametrize("mode", ["crest", "phasor"])
def test_between_sheets_moves_onto_one(tmp_path, mode):
    res, x = run(tmp_path, 77.0, mode=mode)  # half-way between the crests at 70 and 84
    v = x >= 0  # every valid vertex has a normal (one-sided differences at the border and around the hole)
    assert np.all(np.minimum(np.abs(x[v] - 70), np.abs(x[v] - 84)) < 1.0), (x[v].min(), x[v].max())
    assert x[0, 0] == -1


@pytest.mark.parametrize("mode", ["crest", "phasor"])
def test_near_a_sheet_moves_onto_it(tmp_path, mode):
    res, x = run(tmp_path, 87.0, mode=mode)  # 3 voxels off the crest at 84
    assert np.all(np.abs(x[x >= 0] - 84) < 1.0)


@pytest.mark.parametrize("mode", ["crest", "phasor"])
def test_on_a_sheet_stays(tmp_path, mode):
    res, x = run(tmp_path, 84.0, mode=mode)
    assert np.all(np.abs(x[x >= 0] - 84) < 0.5)
    assert res["median_abs_offset_vox"] < 0.5



def write_volume_crests(d, crests, shape=(192, 192, 192), chunk=64):
    """zarr v2 like write_volume, with Gaussian sheets (sigma 1.5 voxels) at the given x positions: an irregular spacing."""
    os.makedirs(d, exist_ok=True)
    json.dump({"zarr_format": 2, "shape": list(shape), "chunks": [chunk] * 3, "dtype": "|u1", "compressor": None,
               "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"}, open(f"{d}/.zarray", "w"))
    x = np.arange(shape[2])
    row = (60 + sum(150 * np.exp(-(x - c) ** 2 / (2 * 1.5 ** 2)) for c in crests)).clip(0, 255).astype(np.uint8)
    for cz in range(shape[0] // chunk):
        for cy in range(shape[1] // chunk):
            for cx in range(shape[2] // chunk):
                block = np.broadcast_to(row[cx * chunk:(cx + 1) * chunk], (chunk, chunk, chunk)).copy()
                os.makedirs(f"{d}/{cz}/{cy}", exist_ok=True)
                block.tofile(f"{d}/{cz}/{cy}/{cx}")


def test_irregular_sheets_crest_mode_finds_the_real_sheet(tmp_path):
    # sheets 14 apart, then a 26-voxel gap (70 -> 96). One Fourier coefficient at the 14-voxel period puts a phantom
    # crest in the gap (at ~83: --mode phasor moves windings at 80 and 87 there); the crest mode moves them onto the
    # real sheets at 70 and 96. (Half-way in the gap, with no crest within 3/4 period, a tile keeps offset 0 and moves only
    # with its neighbours through the smoothing.)
    vol = str(tmp_path / "vol")
    write_volume_crests(vol, [42, 56, 70, 96, 110, 124])
    for x0, sheet in ((80.0, 70.0), (87.0, 96.0)):
        mesh, out = str(tmp_path / f"w{int(x0):03d}"), str(tmp_path / f"out{int(x0)}")
        write_plane(mesh, x0)
        p = subprocess.run([sys.executable, TOOL, mesh, vol, out, "--period", str(PERIOD), "--tile", "6"], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        x = tifffile.imread(out + "/x.tif")
        assert np.all(np.abs(x[x >= 0] - sheet) < 1.0), (x0, x[x >= 0].min(), x[x >= 0].max())
