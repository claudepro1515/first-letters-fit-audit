"""Speed and float16-vs-float32 check of the winding model on one GPU, on radial slabs of one slice.

Seeds sit on circles around the umbilicus at z (radii x angles); each slab's ray runs radially outward.
For both precisions: seconds per slab (batch 4, after a warm-up batch), the centre ray's phase registered
at the anchor, and its decoded crossings. Prints one JSON line starting with BENCH.

Usage: python bench_wm.py <volume zarr> <model ckpt> <umbilicus.json> <z> [--radii ...] [--angles ...]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE + "/stub", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch  # noqa: E402

from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("volume")
ap.add_argument("ckpt")
ap.add_argument("umbilicus")
ap.add_argument("z", type=float)
ap.add_argument("--radii", type=float, nargs="*", default=[400, 700, 1000, 1300, 1600, 1900])
ap.add_argument("--angles", type=float, nargs="*", default=[0, 90, 180, 270])
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
dev = torch.device(a.device)
if dev.type == "cuda":
    torch.cuda.set_device(dev)
    os.environ["VC_SHIM_DEVICE"] = str(dev)
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
model = WindingModel(cfg.get("model"))
model.load_state_dict(ck["model"])
model.to(dev).eval()
L, TS, sp = int(cfg["ray_length"]), int(cfg["transverse_size"]), float(cfg["spacing"])
ex = VolumeSlabExtractor([VolumeSlabExtractor.scaled_volume_path(Path(a.volume), 0)], transverse_size=TS, ray_length=L,
                         spacing=sp, sampling=str(cfg.get("sampling", "trilinear")), tile_size=int(cfg.get("tile_size", 64)),
                         cache_bytes=int(cfg.get("volume_cache_bytes", 0)))
cp = sorted(json.load(open(a.umbilicus))["control_points"], key=lambda q: q["z"])
uy = float(np.interp(a.z, [q["z"] for q in cp], [q["y"] for q in cp]))
ux = float(np.interp(a.z, [q["z"] for q in cp], [q["x"] for q in cp]))
slabs, meta = [], []
t0 = time.time()
for r in a.radii:
    for ang in a.angles:
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
        seed = np.array([ux, uy, a.z]) + r * d
        img, valid, frame = ex.extract(0, d, seed - (L - 1) / 2.0 * sp * d)
        slabs.append((img, valid))
        meta.append((r, ang))
t_extract = (time.time() - t0) / len(slabs)
c = int(round((TS - 1) / 2))  # 64 for 128 columns at stride 1 (exporter's centre)
anchor = int(round((L - 1) / 2))


def run(precision):
    out, times = [], []
    with torch.inference_mode():
        for k in range(0, len(slabs), a.batch):
            b = slabs[k:k + a.batch]
            x = torch.from_numpy(np.stack([s[0] for s in b])).to(dev)
            v = torch.from_numpy(np.stack([s[1] for s in b])).to(dev).bool()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            if precision == "fp16" and dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    ph = model(x, v)["phase"].float()
            else:
                ph = model(x, v)["phase"].float()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t) / len(b))
            out.append(ph[:, c, c, :].cpu().numpy())
            del ph
    per = float(np.median(times[1:])) if len(times) > 1 else float(times[0])
    return np.concatenate(out), per


res = {"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
       "slabs": len(slabs), "extract_s_per_slab": round(t_extract, 3)}
phases = {}
for prec in ("fp32", "fp16"):
    try:
        ph, per = run(prec)
        phases[prec] = ph
        res[f"{prec}_s_per_slab"] = round(per, 4)
        if dev.type == "cuda":
            res[f"{prec}_peak_mem_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 2)
            torch.cuda.reset_peak_memory_stats(dev)
    except Exception as e:  # noqa: BLE001
        res[f"{prec}_error"] = str(e)[:300]
if "fp32" in phases and "fp16" in phases:
    r32 = phases["fp32"] - phases["fp32"][:, anchor:anchor + 1]
    r16 = phases["fp16"] - phases["fp16"][:, anchor:anchor + 1]
    diff = np.abs(r32 - r16)
    res["fp16_vs_fp32_registered_phase_abs_diff"] = {"median": round(float(np.median(diff)), 4),
                                                      "p99": round(float(np.percentile(diff, 99)), 4),
                                                      "max": round(float(diff.max()), 4)}
    same = 0
    for i, (img, valid) in enumerate(slabs):
        p32, _ = decode_center_ray(phases["fp32"][i], valid[c, c].astype(bool), anchor=anchor)
        p16, _ = decode_center_ray(phases["fp16"][i], valid[c, c].astype(bool), anchor=anchor)
        same += int(len(p32) == len(p16))
    res["fp16_same_crossing_count_frac"] = round(same / len(slabs), 3)
ref = phases.get("fp32", phases.get("fp16"))
by_r = {}
for i, (r, ang) in enumerate(meta):
    pos, lev = decode_center_ray(ref[i], slabs[i][1][c, c].astype(bool), anchor=anchor)
    g = np.diff(pos) * sp
    by_r.setdefault(int(r), []).extend(g.tolist())
res["model_crossing_gap_vox_median_by_radius"] = {k: (round(float(np.median(v)), 1) if v else None) for k, v in by_r.items()}
res["windings_per_ray_median"] = round(float(np.median(ref[:, -1] - ref[:, 0])), 2)
print("BENCH " + json.dumps(res), flush=True)
