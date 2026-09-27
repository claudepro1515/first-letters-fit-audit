"""CPU integration test of run_infer.py's pieces + villa's exporter + the fitter's store loader.

Synthetic seeding meshes: two concentric cylinder patches (windings 40 and 41) around the PHerc0826
umbilicus at r = 1000 / 1016, z 9300-9360, inside the 16 locally downloaded level-0 chunks.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE + "/stub")
os.environ["WM_AUTOCAST"] = "fp32"
import torch  # noqa: E402

import run_infer  # noqa: E402  (applies the patches)
import vesuvius.neural_tracing.winding_models.infer_winding_volume as iwv  # noqa: E402
from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray  # noqa: E402

T = HERE + "/t_pipe"
shutil.rmtree(T, ignore_errors=True)
os.makedirs(T + "/meshes/fit")
umb = json.load(open("/tmp/claude-0/-home-claude/f453f88b-9bf6-506d-8057-7d6307920b87/scratchpad/umb0826.json"))
json.dump(umb, open(T + "/umbilicus.json", "w"))
cp = sorted(umb["control_points"], key=lambda q: q["z"])
uz = np.array([q["z"] for q in cp], float)
uy = np.array([q["y"] for q in cp], float)
ux = np.array([q["x"] for q in cp], float)
for w, r in ((40, 1000.0), (41, 1016.0)):
    zs = np.arange(9300, 9361, 4, dtype=float)
    th = np.arange(-12, 13) * 4.0 / r
    Z, TH = np.meshgrid(zs, th, indexing="ij")
    X = np.interp(Z, uz, ux) + r * np.cos(TH)
    Y = np.interp(Z, uz, uy) + r * np.sin(TH)
    d = f"{T}/meshes/fit/w{w:03d}_spliced"
    os.makedirs(d)
    for c, a in zip("xyz", (X, Y, Z)):
        Image.fromarray(a.astype(np.float32)).save(f"{d}/{c}.tif")
    json.dump({"scale": [0.25, 0.25], "format": "tifxyz", "uuid": f"w{w:03d}_spliced"}, open(d + "/meta.json", "w"))
torch.save({"cfg": {"initial_dr_per_winding": 16.0, "output_first_winding": 10, "shell_outer_winding_idx": 130},
            "z_begin": 9290, "z_end": 9370}, T + "/checkpoint_fitted.ckpt")

args = argparse.Namespace(
    fit_checkpoint=T + "/checkpoint_fitted.ckpt", output=T + "/phase_cache.zarr", reference_zarr=HERE + "/vol0826",
    volume_scale=0, model_ckpt=HERE + "/ckpt_final.pth", umbilicus=T + "/umbilicus.json", seed_source="meshes",
    meshes_dir=T + "/meshes/fit", z_range=None, winding_range=[40, 41], winding_step=1, seed_spacing=48.0)
rays = iwv.build_seed_rays_from_meshes(args)
print("seeds", len(rays["seed_xyz"]), "first", rays["seed_xyz"][0], "dir", rays["direction_xyz"][0])
# keep 3 seeds (the model takes ~20 s per slab on this CPU)
keep = np.array([0, len(rays["seed_xyz"]) // 2, len(rays["seed_xyz"]) - 1])
for k in ("seed_xyz", "direction_xyz", "seed_winding"):
    rays[k] = rays[k][keep]
ck = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
cfg = ck["config"]
n = len(rays["seed_xyz"])
run_infer.init_centre_cache(args, rays, np.array([0, n]), [0], cfg)
ex = VolumeSlabExtractor([VolumeSlabExtractor.scaled_volume_path(__import__("pathlib").Path(args.reference_zarr), 0)],
                         transverse_size=int(cfg["transverse_size"]), ray_length=int(cfg["ray_length"]),
                         spacing=float(cfg["spacing"]), sampling=str(cfg["sampling"]), tile_size=int(cfg["tile_size"]),
                         cache_bytes=int(cfg["volume_cache_bytes"]),
                         segment_to_volume_xyz=[VolumeSlabExtractor.load_segment_to_volume_transform(
                             __import__("pathlib").Path(args.reference_zarr), 0, segment_downscale=1, use_registration=False)])
model = WindingModel(cfg.get("model"))
model.load_state_dict(ck["model"])
model.eval()
L = int(cfg["ray_length"])
mid = (L - 1) / 2.0
batch = []
for i in range(n):
    origin = rays["seed_xyz"][i].astype(np.float64) - mid * float(cfg["spacing"]) * rays["direction_xyz"][i].astype(np.float64)
    img, valid, frame = ex.extract(0, rays["direction_xyz"][i], origin)
    batch.append((i, img, valid, frame))
phases, full = [], []
with torch.inference_mode():
    for b in batch:
        _, ph, _ = run_infer.forward_batch_centre([b], model, torch.device("cpu"), 1, args=None, column_stride=1)
        phases.append(ph[0])
        # reference: the full field's centre column straight from the model (same input)
        o = model(torch.from_numpy(b[1][None]), torch.from_numpy(b[2][None]).bool())["phase"].float()
        full.append(o[0, 64, 64].numpy().copy())
        del o
phases = np.stack(phases)
print("centre phases", phases.shape)
w = run_infer.CentreCacheWriter(args.output, 0)
w.add([b[0] for b in batch], phases, [b[2] for b in batch], [b[3] for b in batch])
import zarr  # noqa: E402
g = zarr.open_group(args.output, mode="r+")
g.attrs.update({"complete": True, "num_slabs": n, "num_available_slabs": n})
exp = "/home/claude/vc/villa/vesuvius/src/vesuvius/neural_tracing/winding_models/export_spiral_supervision.py"
r = subprocess.run([sys.executable, exp, args.output, T + "/winding_inference", "--workers", "1", "--no-progress"],
                   capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
# reference decode from the full phase field, exporter conventions (centre 64, anchor 192)
for b, ph in zip(batch, full):
    pos, lev = decode_center_ray(ph, b[2][64, 64].astype(bool), anchor=192)
    print("slab", b[0], "reference crossings", len(pos), np.round(pos[:6], 2).tolist(), lev[:6].tolist())
sys.path.insert(0, "/home/claude/vc/villa/spiral-fitting")
from winding_supervision import load_winding_inference_store  # noqa: E402
store = load_winding_inference_store(T + "/winding_inference", "cpu", z_range=(9290, 9370))
print("store rays", len(store.origin), "crossings", len(store.crossing_t), "z-eligible", store.num_z_eligible_rays)
for i in range(len(store.origin)):
    a, b_ = int(store.offset[i]), int(store.offset[i + 1])
    print("ray", i, "origin zyx", store.origin[i].numpy().round(1).tolist(), "step", store.step[i].numpy().round(3).tolist(),
          "t", store.crossing_t[a:b_][:6].numpy().round(2).tolist(), "levels", store.crossing_level[a:b_][:6].tolist())
s = store.sample_adjacent(5)
print("adjacent sample points", s["points"].shape, "targets", s["target"].tolist())
# check the stored ray origin equals the full-cache convention: slab origin + 64 * (a + b)
f = batch[0][3]
print("expected origin zyx", (f.origin + 64.0 * (f.axis_a + f.axis_b))[::-1].round(1).tolist())
