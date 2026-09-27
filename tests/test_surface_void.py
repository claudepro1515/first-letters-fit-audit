"""tools/surface_void.py on a synthetic volume: sheets (bright planes every 14 voxels) for x < 96, empty papyrus-free
space (dark but inside the scan) for x >= 96. A plane on a sheet is not in a void; a plane in the empty half is."""
import json
import os
import subprocess
import sys

import numpy as np
import tifffile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "surface_void.py")


def write_volume(d, shape=(128, 128, 192), chunk=64):
    os.makedirs(d, exist_ok=True)
    json.dump({"zarr_format": 2, "shape": list(shape), "chunks": [chunk] * 3, "dtype": "|u1", "compressor": None,
               "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "/"}, open(f"{d}/.zarray", "w"))
    x = np.arange(shape[2])
    row = np.where(x < 96, 40 + 180 * (np.cos(2 * np.pi * x / 14) > 0.6), 40).astype(np.uint8)
    for cz in range(shape[0] // chunk):
        for cy in range(shape[1] // chunk):
            for cx in range(shape[2] // chunk):
                block = np.broadcast_to(row[cx * chunk:(cx + 1) * chunk], (chunk, chunk, chunk)).copy()
                os.makedirs(f"{d}/{cz}/{cy}", exist_ok=True)
                block.tofile(f"{d}/{cz}/{cy}/{cx}")


def write_plane(d, x0, step=4.0, n=20):
    os.makedirs(d, exist_ok=True)
    zz, yy = np.meshgrid(24 + step * np.arange(n), 24 + step * np.arange(n), indexing="ij")
    xx = np.full_like(zz, x0, dtype=np.float64)
    for c, a in zip("xyz", (xx, yy, zz)):
        tifffile.imwrite(f"{d}/{c}.tif", a.astype(np.float32))


def test_void_fraction(tmp_path):
    vol = str(tmp_path / "vol")
    write_volume(vol)
    write_plane(str(tmp_path / "fit" / "w001"), 56.0)   # on a sheet (crest at x = 56)
    write_plane(str(tmp_path / "fit" / "w002"), 150.0)  # in the empty half
    p = subprocess.run([sys.executable, TOOL, str(tmp_path / "fit"), vol, "--z", "0", "128", "--stride", "1"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    r = json.loads(p.stdout.strip().splitlines()[-1])
    assert r["void_frac_by_winding"]["w001"] < 0.05, r
    assert r["void_frac_by_winding"]["w002"] > 0.95, r
