"""Winding model vs Lasagna grad_mag vs CT peaks on radial rays of PHerc0826 at z 9328 (r 808-1192).

At 180 and 270 degrees the September audit found grad_mag counting ~2x the CT sheets (folded regions).
For each angle: one 128x128x384 slab (seed at r = 1000, ray radially outward), the model's phase along the
centre ray (windings = phase change over the valid samples 8..376), the grad_mag integral along the same
segment (u8 * 0.001 windings per voxel, level 2), and CT peaks (median over 9 nearby profiles).
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, HERE + "/stub", "/home/claude/vc/release/tools"]
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray  # noqa: E402
from vcz import OmeZarr  # noqa: E402

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
GM = OmeZarr(f"{S3}/PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421/PHerc0826_grad_mag.ome.zarr").level("2")
ck = torch.load(HERE + "/ckpt_final.pth", map_location="cpu", weights_only=False)
cfg = ck["config"]
model = WindingModel(cfg["model"])
model.load_state_dict(ck["model"])
model.eval()
ex = VolumeSlabExtractor([Path(HERE + "/vol0826/0")], transverse_size=128, ray_length=384, spacing=1.0,
                         sampling="trilinear", tile_size=64, cache_bytes=int(cfg["volume_cache_bytes"]))
uy, ux, z = 5388.0, 3517.0, 9328.0
angles = [int(a) for a in sys.argv[1:]] or [0, 180, 270]
res = []
figs = []
for ang in angles:
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
    seed = np.array([ux, uy, z]) + 1000.0 * d
    origin = seed - 191.5 * d
    img, valid, frame = ex.extract(0, d, origin)
    t = time.time()
    with torch.inference_mode():
        ph = model(torch.from_numpy(img[None]), torch.from_numpy(valid[None]).bool())["phase"][0].float().numpy()
    tf = time.time() - t
    c = 64
    pc = ph[c, c]
    pos, lev = decode_center_ray(pc, valid[c, c].astype(bool), anchor=192, edge_trim=0)
    model_w = float(pc[376] - pc[8])
    # grad_mag along the same centre ray (samples 8..376): points frame.origin + 64 (a + b) + k d
    k = np.arange(8, 377)
    p = frame.origin + 64.0 * (frame.axis_a + frame.axis_b) + k[:, None] * frame.direction  # xyz
    q = np.round(p[:, ::-1] / 4.0).astype(int)  # zyx level 2
    lo, hi = q.min(0), q.max(0) + 1
    blk = GM.read(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    g = blk[q[:, 0] - lo[0], q[:, 1] - lo[1], q[:, 2] - lo[2]].astype(float)
    gm_w = float((g * 0.001).sum())
    counts = []
    for da in (-2, 0, 2):
        for db in (-2, 0, 2):
            prof = img[c + da, c + db, 8:377].astype(float)
            s = gaussian_filter1d(prof, 1.5)
            pk, _ = find_peaks(s, distance=5, prominence=max(3.0, 0.05 * (np.percentile(s, 95) - np.percentile(s, 5))))
            counts.append(len(pk))
    r = {"angle": ang, "model_windings_8_376": round(model_w, 2), "model_crossings": len(pos),
         "model_gap_median_vox": round(float(np.median(np.diff(pos))), 1) if len(pos) > 1 else None,
         "gradmag_windings_8_376": round(gm_w, 2), "gradmag_zero_frac": round(float((g == 0).mean()), 3),
         "ct_peaks_median": float(np.median(counts)), "ct_peaks_range": [min(counts), max(counts)],
         "valid_frac": round(float(valid.mean()), 3), "forward_s": round(tf, 1)}
    print(json.dumps(r), flush=True)
    res.append(r)
    figs.append((ang, img, ph, pos))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, axs = plt.subplots(len(figs), 2, figsize=(16, 3.2 * len(figs)))
axs = np.atleast_2d(axs)
for i, (ang, img, ph, pos) in enumerate(figs):
    axs[i, 0].imshow(img[:, 64], cmap="gray", aspect="auto")
    for x in pos:
        axs[i, 0].axvline(x, color="r", lw=0.6, alpha=0.7)
    axs[i, 0].set_title(f"PHerc0826 z 9328, ray at {ang} deg, r 808-1192: CT (axis_a x ray), model crossings on the centre ray (red)")
    axs[i, 1].imshow(np.mod(ph[:, 64], 1.0), cmap="twilight", aspect="auto")
    axs[i, 1].set_title(f"model phase mod 1 ({res[i]['model_windings_8_376']} windings; grad_mag {res[i]['gradmag_windings_8_376']}; CT peaks {res[i]['ct_peaks_median']:.0f})")
plt.tight_layout()
plt.savefig(HERE + "/test_angles.png", dpi=70)
print("saved", HERE + "/test_angles.png")
