"""Synthetic check of tools/sheet_hits.py: bright cylindrical sheets 16 voxels apart around a vertical axis, and
three 'fits' made of cylinders: one on the sheets, one with two windings per sheet, one half a spacing off.
Expected: c > 0 for nearly every crossing, ~half, and nearly none.

  python -m pytest tests/test_sheet_hits.py -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
zarr = pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")

N, CZ, CY, CX = 96, 160, 160, 160  # volume z, and the axis position in y and x (volume 320 x 320 in y/x)


def write_volume(path):
    zz, yy, xx = np.meshgrid(np.arange(N), np.arange(320), np.arange(320), indexing="ij")
    r = np.hypot(yy - CY, xx - CX)
    phase = (r % 16.0) / 16.0
    vol = (60 + 150 * np.exp(-((np.minimum(phase, 1 - phase) * 16.0) ** 2) / (2 * 1.5 ** 2)))  # sheets at r = 16 k
    vol[r > 150] = 0  # outside the 'scan mask'
    root = zarr.open_group(path, mode="w", zarr_format=2)
    root.create_array("0", data=vol.astype(np.uint8), chunks=(32, 64, 64))


def write_cylinders(d, radii):
    for k, rad in enumerate(radii):
        m = os.path.join(d, f"w{k + 10:03d}")
        os.makedirs(m, exist_ok=True)
        th = np.linspace(0, 2 * np.pi, max(24, int(2 * np.pi * rad / 6)), endpoint=True)
        zs = np.arange(-8.0, N + 8.0, 6.0)
        T, Z = np.meshgrid(th, zs)
        for c, a in zip("xyz", (CX + rad * np.cos(T), CY + rad * np.sin(T), Z)):
            tifffile.imwrite(os.path.join(m, f"{c}.tif"), a.astype(np.float32))
        json.dump({"format": "tifxyz", "scale": [1 / 6, 1 / 6], "type": "seg", "uuid": m}, open(os.path.join(m, "meta.json"), "w"))


def run(tmp_path, radii, name):
    d = tmp_path / name
    write_cylinders(str(d), radii)
    out = subprocess.run([sys.executable, os.path.join(TOOLS, "sheet_hits.py"), str(d), str(tmp_path / "vol.zarr"),
                          "--z", "10", str(N - 10), "--umbilicus", str(tmp_path / "umb.json"), "--radial", "40",
                          "--r0", "20", "--r1", "140"], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])["fit"]


def test_sheet_hits_separates_on_sheet_double_and_shifted(tmp_path):
    write_volume(str(tmp_path / "vol.zarr"))
    json.dump({"control_points": [{"z": 0, "y": CY, "x": CX}, {"z": N, "y": CY, "x": CX}]}, open(tmp_path / "umb.json", "w"))
    on = run(tmp_path, [16.0 * k for k in range(2, 9)], "on")
    double = run(tmp_path, [8.0 * k for k in range(4, 18)], "double")
    shifted = run(tmp_path, [16.0 * k + 8 for k in range(2, 9)], "shifted")
    assert on["frac_c_pos"] > 0.95 and on["median_c"] > 0.3
    assert 0.3 < double["frac_c_pos"] < 0.7 and abs(double["median_c"]) < abs(on["median_c"])
    assert shifted["frac_c_pos"] < 0.05 and shifted["median_c"] < -0.3


def test_gaps_flag_windings_on_the_same_sheet(tmp_path):
    """Two windings moved onto one sheet show up as near-zero gaps between consecutive crossings."""
    write_volume(str(tmp_path / "vol.zarr"))
    json.dump({"control_points": [{"z": 0, "y": CY, "x": CX}, {"z": N, "y": CY, "x": CX}]}, open(tmp_path / "umb.json", "w"))

    def gaps(radii, name):
        d = tmp_path / name
        write_cylinders(str(d), radii)
        out = subprocess.run([sys.executable, os.path.join(TOOLS, "sheet_hits.py"), str(d), str(tmp_path / "vol.zarr"),
                              "--z", "10", str(N - 10), "--umbilicus", str(tmp_path / "umb.json"), "--radial", "40",
                              "--r0", "20", "--r1", "140"], capture_output=True, text=True, check=True).stdout
        return json.loads(out.strip().splitlines()[-1])["gaps_vox"]

    on = gaps([16.0 * k for k in range(2, 9)], "on")
    dup = gaps([32.0, 48.0, 48.5, 64.0, 80.0, 96.0, 112.0], "dup")  # one winding pair on the sheet at r = 48
    assert abs(on["median"] - 16.0) < 0.5 and on["frac_lt_3vox"] == 0
    assert 0.1 < dup["frac_lt_3vox"] < 0.25 and dup["frac_lt_half_median"] == dup["frac_lt_3vox"]
    # the close pair belongs to two different windings (not one winding crossing the ray twice)
    assert dup["frac_lt_3vox_two_windings"] == dup["frac_lt_3vox"] and dup["frac_lt_3vox_one_winding"] == 0


def test_windings_at_two_thirds_of_the_sheet_spacing_score_about_a_third(tmp_path):
    """Three windings per two sheets: one of every three sits on a sheet with gaps on both sides (c > 0); the other two
    sit 5.3 voxels off a sheet with a sheet at one of their mid-points (c < 0)."""
    write_volume(str(tmp_path / "vol.zarr"))
    json.dump({"control_points": [{"z": 0, "y": CY, "x": CX}, {"z": N, "y": CY, "x": CX}]}, open(tmp_path / "umb.json", "w"))
    # 11 windings from r = 21.3 to 128: the 9 interior crossings (the outer two have one neighbour) hold 3 on sheets
    fit = run(tmp_path, [64.0 / 3.0 + 32.0 / 3.0 * k for k in range(11)], "two_thirds")
    assert 0.28 < fit["frac_c_pos"] < 0.4


def test_random_phase_null_sits_near_half_and_below_a_fit_on_the_sheets(tmp_path):
    """--null K: the fit's crossings shifted along each ray by a random offset (within half their gap) score about a half
    on evenly spaced sheets; a fit on the sheets is above every copy."""
    write_volume(str(tmp_path / "vol.zarr"))
    json.dump({"control_points": [{"z": 0, "y": CY, "x": CX}, {"z": N, "y": CY, "x": CX}]}, open(tmp_path / "umb.json", "w"))
    d = tmp_path / "on"
    write_cylinders(str(d), [16.0 * k for k in range(2, 9)])
    out = subprocess.run([sys.executable, os.path.join(TOOLS, "sheet_hits.py"), str(d), str(tmp_path / "vol.zarr"),
                          "--z", "10", str(N - 10), "--umbilicus", str(tmp_path / "umb.json"), "--radial", "40",
                          "--r0", "20", "--r1", "140", "--null", "40"], capture_output=True, text=True, check=True).stdout
    res = json.loads(out.strip().splitlines()[-1])
    null = res["null_random_phase"]
    assert res["fit"]["frac_c_pos"] > 0.95 and null["copies"] == 40
    assert 0.3 < null["frac_c_pos_mean"] < 0.7 and null["frac_c_pos_p97.5"] < 0.9 and null["frac_at_or_above_fit"] == 0.0
    # two windings per sheet: the null depends on the fit's spacing (about a third), and the fit (half of its windings
    # on the sheets, 0.5) is above it
    d = tmp_path / "double"
    write_cylinders(str(d), [8.0 * k for k in range(4, 18)])
    out = subprocess.run([sys.executable, os.path.join(TOOLS, "sheet_hits.py"), str(d), str(tmp_path / "vol.zarr"),
                          "--z", "10", str(N - 10), "--umbilicus", str(tmp_path / "umb.json"), "--radial", "40",
                          "--r0", "20", "--r1", "140", "--null", "40"], capture_output=True, text=True, check=True).stdout
    res = json.loads(out.strip().splitlines()[-1])
    assert 0.25 < res["null_random_phase"]["frac_c_pos_mean"] < 0.42 and res["fit"]["frac_c_pos"] == 0.5
