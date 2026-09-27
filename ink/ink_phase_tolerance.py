"""How far off the sheet can a surface be before ink_9um stops reading the ink? (dev tool)

Renders the control crop of a curated segment once with a wide slab of layers (surface + k n, k = -13 - M .. 14 + M,
the published convention), then for every shift s in [-M, M] takes the 28 layers of a surface moved by s voxels
along n (layer L = surface + (L - 13 + s) n), runs ink_9um (forward) on them and scores the prediction against the
published labels inside the labelled area (AUC). Also reports the mean layer profile of the crop, so the shift that
puts the sheet at layer 13 can be read off.

Usage: python ink_phase_tolerance.py MESH_DIR VOLUME_URL LABELS_PNG SUPERVISION_PNG CKPT WORK --box r0 r1 c0 c1
       [--max-shift 8] [--shifts -8 -6 ...] [--cache DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import tifffile
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INK = HERE if os.path.exists(os.path.join(HERE, "render_sv.py")) else os.path.join(os.path.dirname(HERE), "ink")
sys.path[:0] = [INK, os.path.join(INK, "tools")]
import render_sv  # noqa: E402
from inkjob import auc  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("mesh", "volume", "labels", "supervision", "ckpt", "work"):
    ap.add_argument(k)
ap.add_argument("--box", type=int, nargs=4, required=True)
ap.add_argument("--max-shift", type=int, default=8)
ap.add_argument("--shifts", type=int, nargs="*", default=None)
ap.add_argument("--cache", default=None)
ap.add_argument("--env-pythonpath", default=None, help="PYTHONPATH for the ink model's process")
a = ap.parse_args()
os.makedirs(a.work, exist_ok=True)
M = a.max_shift
r0, r1, c0, c1 = a.box
lab = np.array(Image.open(a.labels).convert("L")) > 0
sup = np.array(Image.open(a.supervision).convert("L")) > 0

t = time.time()
rd = render_sv.Renderer(a.mesh, render_sv.open_volume(a.volume, a.cache), "cpu")
wide = np.arange(28 + 2 * M) - 13 - M
rd.offs = torch.from_numpy(wide.astype(np.float64))
stack = np.zeros((len(wide), r1 - r0, c1 - c0), np.uint8)  # (28 + 2M, H, W); Renderer.render allocates 28 layers
view = render_sv._Offset(stack, r0, c0)
for i in range(r0, r1, 1024):
    for j in range(c0, c1, 256):
        rd.block(i, min(r1, i + 1024), j, min(c1, j + 256), view)
t_render = time.time() - t
valid = stack[13 + M] > 0
prof = [round(float(stack[i][valid].mean()), 2) for i in range(len(wide))]
print(json.dumps({"render_s": round(t_render, 1), "layers": [int(w) for w in wide], "mean_profile": prof}), flush=True)

env = dict(os.environ)
if a.env_pythonpath:
    env["PYTHONPATH"] = a.env_pythonpath
for s in (a.shifts if a.shifts is not None else range(-M, M + 1, 2)):
    sub = stack[M + s: M + s + 28]  # layer L of the moved surface = original offset L - 13 + s
    zp = f"{a.work}/shift{s:+d}.zarr"
    render_sv.write_ome_zarr(zp, sub)
    out = f"{a.work}/shift{s:+d}_pred.tif"
    t = time.time()
    p = subprocess.run([sys.executable, "-m", "koine_machines.inference.infer", zp, a.ckpt, out, "--direction", "forward",
                        "--no-compile", "--batch-size", "8", "--num-workers", "2"], env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        raise SystemExit(1)
    pred = tifffile.imread(out)
    v = auc(pred[sup], lab[sup])
    print(json.dumps({"shift_vox": s, "auc_labelled_area": round(v, 3) if v is not None else None,
                      "ink_s": round(time.time() - t, 1), "pred_mean": round(float(pred[sup].mean()), 1)}), flush=True)
