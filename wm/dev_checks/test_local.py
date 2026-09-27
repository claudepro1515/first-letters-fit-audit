"""Local CPU check of the winding-model pipeline pieces on PHerc0826 (one radial ray at z 9328)."""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)              # vc shim
sys.path.insert(0, HERE + "/stub")    # light vesuvius package
import torch  # noqa: E402

from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray  # noqa: E402
import vc  # noqa: E402

ref = HERE + "/vol0826"
ck = torch.load(HERE + "/ckpt_final.pth", map_location="cpu", weights_only=False)
cfg = ck["config"]
print("config keys:", sorted(k for k in cfg if k in ("ray_length", "transverse_size", "spacing", "column_stride", "sampling", "tile_size", "volume_cache_bytes")))
print({k: cfg.get(k) for k in ("ray_length", "transverse_size", "spacing", "column_stride", "sampling", "tile_size")})
ex = VolumeSlabExtractor([VolumeSlabExtractor.scaled_volume_path(__import__("pathlib").Path(ref), 0)],
                         transverse_size=int(cfg.get("transverse_size", 128)), ray_length=int(cfg.get("ray_length", 384)),
                         spacing=float(cfg.get("spacing", 1.0)), sampling=str(cfg.get("sampling", "trilinear")),
                         tile_size=int(cfg.get("tile_size", 64)), cache_bytes=int(cfg.get("volume_cache_bytes", 0)))
uy, ux, z = 5383.04, 3516.65, 9328.0
direction = np.array([1.0, 0.0, 0.0])
seed = np.array([ux + 1000, uy, z])
L = int(cfg.get("ray_length", 384))
origin = seed - (L - 1) / 2.0 * direction
t = time.time()
img, valid, frame = ex.extract(0, direction, origin)
print("slab", img.shape, img.dtype, valid.shape, valid.mean(), "extract s", round(time.time() - t, 2))

# independent check of the shim: direct nearest/trilinear from the raw box at a few slab voxels
vol = vc.Volume.open(ref + "/0")
rng = np.random.default_rng(0)
errs = []
for _ in range(200):
    i, j, k = rng.integers(0, 128), rng.integers(0, 128), rng.integers(0, L)
    p = frame.origin + i * frame.axis_a + j * frame.axis_b + k * frame.direction  # xyz
    zz, yy, xx = p[2], p[1], p[0]
    z0, y0, x0 = int(np.floor(zz)), int(np.floor(yy)), int(np.floor(xx))
    b = vol.read_box((z0, y0, x0), (z0 + 2, y0 + 2, x0 + 2)).astype(float)
    fz, fy, fx = zz - z0, yy - y0, xx - x0
    c = b[0] * (1 - fz) + b[1] * fz
    c = c[0] * (1 - fy) + c[1] * fy
    v = c[0] * (1 - fx) + c[1] * fx
    errs.append(abs(np.floor(v + 0.5) - img[i, j, k]))
print("shim vs direct trilinear: max abs diff", max(errs), "mean", np.mean(errs))

model = WindingModel(cfg.get("model"))
model.load_state_dict(ck["model"])
model.eval()
nparams = sum(p.numel() for p in model.parameters())
print("model params", nparams)
x = torch.from_numpy(img[None])
v = torch.from_numpy(valid[None].astype(bool))
t = time.time()
with torch.inference_mode():
    out = model(x, v)
print("forward fp32 CPU s", round(time.time() - t, 1), {k: tuple(o.shape) for k, o in out.items()})
phase = out["phase"][0].float().numpy()
np.save(HERE + "/test_phase.npy", phase)
np.save(HERE + "/test_img.npy", img)
cs = int(cfg.get("column_stride", 4))
cols = phase.shape[0]
center = int(round((cols * cs - 1) / cs / 2))
anchor = int(round((L - 1) / 2))
pos, lev = decode_center_ray(phase[center, center], valid[center * cs, center * cs].astype(bool), anchor=anchor)
print("centre-ray crossings:", len(pos), "levels", lev.min() if len(lev) else None, lev.max() if len(lev) else None)
print("positions", np.round(pos, 1).tolist())
gaps = np.diff(pos)
print("gap median", float(np.median(gaps)) if len(gaps) else None, "gaps", np.round(gaps, 1).tolist())
# CT profile along the centre ray (mean over a 5x5 column bundle) for comparison
prof = img[center * cs - 2:center * cs + 3, center * cs - 2:center * cs + 3].mean((0, 1))
np.save(HERE + "/test_prof.npy", prof)
from scipy.signal import find_peaks  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402
ps = gaussian_filter1d(prof.astype(float), 1.5)
pk, _ = find_peaks(ps, distance=6, prominence=3)
print("CT peaks", len(pk), pk.tolist())
print("CT peak gaps median", float(np.median(np.diff(pk))) if len(pk) > 1 else None)
