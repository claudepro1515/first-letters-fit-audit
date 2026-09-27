"""tools/block_audit.py end to end on synthetic sheets (bright cylinders every 16 voxels around a vertical axis): windings
on the sheets score high as fitted and aligned, above their random-shift null; the same windings moved 6 voxels off
score lower as fitted and come back onto the sheets after alignment.

  python -m pytest tests/test_block_audit.py -q
"""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
pytest.importorskip("zarr")
pytest.importorskip("tifffile")
from test_sheet_hits import CX, CY, N, write_cylinders, write_volume  # noqa: E402


def audit(tmp_path, radii, name):
    write_cylinders(str(tmp_path / name), radii)
    out = subprocess.run([sys.executable, os.path.join(TOOLS, "block_audit.py"), str(tmp_path / name), str(tmp_path / "vol.zarr"),
                          str(tmp_path / f"work_{name}"), "--z", "10", str(N - 10), "--umbilicus", str(tmp_path / "umb.json"),
                          "--windings", "5", "--period", "16", "--radial", "40", "--r0", "20", "--r1", "140",
                          "--null", "40", "--procs", "2"], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_block_audit_on_and_off_the_sheets(tmp_path):
    write_volume(str(tmp_path / "vol.zarr"))
    json.dump({"control_points": [{"z": 0, "y": CY, "x": CX}, {"z": N, "y": CY, "x": CX}]}, open(tmp_path / "umb.json", "w"))
    on = audit(tmp_path, [16.0 * k for k in range(2, 9)], "on")
    assert on["windings"] == 5 and on["aligned_missing"] == [] and on["phase_align_failed"] == []
    assert on["raw"]["fit"]["frac_c_pos"] > 0.95 and on["aligned"]["fit"]["frac_c_pos"] > 0.95
    assert on["raw"]["null_random_phase"]["frac_at_or_above_fit"] == 0.0
    off = audit(tmp_path, [16.0 * k + 6 for k in range(2, 9)], "off")
    assert off["raw"]["fit"]["frac_c_pos"] < 0.5 < 0.9 < off["aligned"]["fit"]["frac_c_pos"]
