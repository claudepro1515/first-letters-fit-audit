"""Are the winding model's crossings on bright CT? (dev check, CPU)

Radial slabs at one z (radii x angles around the umbilicus), the public winding_model_9um on each, crossings decoded
from the centre ray with villa's decode_center_ray, and the model-free contrast of sheet_hits.py at each crossing:
c = (I(crossing) - mean(I at the mid-points to the neighbouring crossings)) / (p95 - p5 of the ray), I = the slab's own
centre ray (the CT the model saw). frac_c_pos ~0.65 for curated windings on PHerc0139, ~0.33 half a sheet off, 0.5
unrelated. Needed chunks are fetched from the public S3 volume into a local zarr (compressed files as they are).

Usage: python dev_model_brightness.py <volume S3 key, e.g. PHerc0826/volumes/..zarr> <ckpt> <umbilicus.json> <z> <local dir>
       [--radii ...] [--angles ...]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ((HERE + "/stub_cpu") if os.path.isdir(HERE + "/stub_cpu") else (HERE + "/stub"), HERE, os.path.join(os.path.dirname(HERE), "release", "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray  # noqa: E402
from sheet_hits import contrast_at, summarize  # noqa: E402

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
ap = argparse.ArgumentParser()
ap.add_argument("volume_key")
ap.add_argument("ckpt")
ap.add_argument("umbilicus")
ap.add_argument("z", type=float)
ap.add_argument("local")
ap.add_argument("--radii", type=float, nargs="*", default=[500, 800, 1100, 1400, 1700])
ap.add_argument("--angles", type=float, nargs="*", default=[0, 60, 120, 180, 240, 300])
ap.add_argument("--save", default=None, help="npz with every ray's profile, mask, phase and crossings (for offset scans)")
a = ap.parse_args()

vol = Path(a.local)
(vol / "0").mkdir(parents=True, exist_ok=True)
meta = requests.get(f"{S3}/{a.volume_key}/0/.zarray", timeout=60).json()
json.dump(meta, open(vol / "0" / ".zarray", "w"))
open(vol / ".zgroup", "w").write('{"zarr_format": 2}')
CS = np.array(meta["chunks"])
sep = meta.get("dimension_separator", ".")

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
model = WindingModel(cfg.get("model"))
model.load_state_dict(ck["model"])
model.eval()
L, TS, sp = int(cfg["ray_length"]), int(cfg["transverse_size"]), float(cfg["spacing"])
cp = sorted(json.load(open(a.umbilicus))["control_points"], key=lambda q: q["z"])
uy = float(np.interp(a.z, [q["z"] for q in cp], [q["y"] for q in cp]))
ux = float(np.interp(a.z, [q["z"] for q in cp], [q["x"] for q in cp]))


def fetch(key):
    dst = vol / "0" / sep.join(map(str, key))
    if dst.exists():
        return 0
    r = requests.get(f"{S3}/{a.volume_key}/0/" + "/".join(map(str, key)), timeout=120)
    if r.status_code != 200:
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dst) + f".{os.getpid()}.tmp"
    open(tmp, "wb").write(r.content)
    os.replace(tmp, dst)
    return len(r.content)


seeds = []
for r in a.radii:
    for ang in a.angles:
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])  # xyz
        seeds.append((r, ang, d, np.array([ux, uy, a.z]) + r * d))
# chunks each slab touches: ray +-(L/2) sp along d, +-TS/2 sp across (z and the tangent), plus a margin
keys = set()
for r, ang, d, s in seeds:
    t = np.array([-d[1], d[0], 0.0])
    zz = np.array([0.0, 0.0, 1.0])
    for u in np.linspace(-(L - 1) / 2 * sp - 4, (L - 1) / 2 * sp + 4, 16):
        for v in np.linspace(-TS / 2 * sp - 4, TS / 2 * sp + 4, 5):
            for w in np.linspace(-TS / 2 * sp - 4, TS / 2 * sp + 4, 5):
                p = s + u * d + v * t + w * zz  # xyz
                keys.add(tuple((np.array([p[2], p[1], p[0]]) // CS).astype(int)))
t0 = time.time()
with ThreadPoolExecutor(32) as ex:
    nbytes = sum(ex.map(fetch, sorted(keys)))
print(json.dumps({"chunks": len(keys), "mb": round(nbytes / 2**20, 1), "fetch_s": round(time.time() - t0, 1)}), flush=True)

ex_ = VolumeSlabExtractor([VolumeSlabExtractor.scaled_volume_path(vol, 0)], transverse_size=TS, ray_length=L, spacing=sp,
                          sampling=str(cfg.get("sampling", "trilinear")), tile_size=int(cfg.get("tile_size", 64)),
                          cache_bytes=int(cfg.get("volume_cache_bytes", 0)))
c = int(round((TS - 1) / 2))
anchor = int(round((L - 1) / 2))
cs_all, per_ray, saved = [], [], {}
for r, ang, d, s in seeds:
    img, valid, frame = ex_.extract(0, d, s - (L - 1) / 2.0 * sp * d)
    with torch.inference_mode():
        ph = model(torch.from_numpy(img[None]), torch.from_numpy(valid[None]).bool())["phase"].float()[0, c, c].numpy()
    ok = valid[c, c].astype(bool)
    pos, lev = decode_center_ray(ph, ok, anchor=anchor)
    col = img[c, c].astype(np.float32) if img.ndim == 3 else img[0, c, c].astype(np.float32)
    ts = np.arange(len(col)) * sp
    prof = gaussian_filter1d(col, 1.0 / sp)
    inside = (col > 0) & ok
    cc = contrast_at(prof, inside, ts, np.sort(np.asarray(pos, float) * sp))
    k = len(per_ray)
    saved.update({f"prof{k}": prof, f"inside{k}": inside, f"phase{k}": ph, f"pos{k}": np.asarray(pos, float) * sp,
                  f"raw{k}": col})
    cs_all.append(cc)
    per_ray.append({"r": r, "angle": ang, "crossings": int(len(pos)), "frac_c_pos": round(float((cc > 0).mean()), 3) if len(cc) else None})
    print(json.dumps(per_ray[-1]), flush=True)
allc = np.concatenate(cs_all) if cs_all else np.zeros(0)
if a.save:
    np.savez_compressed(a.save, sp=sp, **saved)
    # offset scan: move every model crossing by delta voxels along its ray and recompute frac_c_pos
    scan = {}
    for delta in np.arange(-8, 8.5, 1.0):
        cc_ = [contrast_at(saved[f"prof{k}"], saved[f"inside{k}"], np.arange(len(saved[f"prof{k}"])) * sp,
                           np.sort(saved[f"pos{k}"]) + delta) for k in range(len(per_ray))]
        cc_ = np.concatenate(cc_) if cc_ else np.zeros(0)
        scan[float(delta)] = round(float((cc_ > 0).mean()), 3) if len(cc_) else None
    print("OFFSET_SCAN " + json.dumps(scan), flush=True)
print("MODEL " + json.dumps({"z": a.z, "rays": len(seeds), "model_crossings": summarize(allc),
                             "by_radius": {int(r): summarize(np.concatenate([cs_all[i] for i, x in enumerate(seeds) if x[0] == r]))
                                           for r in a.radii}}), flush=True)
