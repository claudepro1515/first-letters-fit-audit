import sys, os, json, numpy as np, torch, zarr
sys.path[:0] = ["/home/claude/vc/wm", "/home/claude/vc/wm/stub"]
from pathlib import Path
from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel
from vesuvius.neural_tracing.winding_models.export_spiral_supervision import decode_center_ray
g = zarr.open_group("t_grid/cache.zarr", mode="r")
fr = g["frame"]["shard_0"][:]
cols = g.attrs["column_grid"]["transverse_samples"]
k0 = cols.index([64, 64])
f = fr[k0]
slab_origin = f[0] - 64.0 * (f[1] + f[2])
ck = torch.load("ckpt_final.pth", map_location="cpu", weights_only=False); cfg = ck["config"]
ex = VolumeSlabExtractor([Path("vol0826/0")], transverse_size=128, ray_length=384, spacing=1.0, sampling="trilinear", tile_size=64)
direction = f[3]
img, valid, frame = ex.extract(0, direction, slab_origin + 63.5 * (f[1] + f[2]))
print("frame origin matches", np.allclose(frame.origin, slab_origin), np.allclose(frame.axis_a, f[1]), np.allclose(frame.axis_b, f[2]))
m = WindingModel(cfg["model"]); m.load_state_dict(ck["model"]); m.eval()
with torch.inference_mode():
    ph = m(torch.from_numpy(img[None]), torch.from_numpy(valid[None]).bool())["phase"][0].float().numpy()
sys.path.insert(0, "/home/claude/vc/release/tools")
from winding_agreement import load_store
st = load_store("t_grid/wi")
print("store rays", len(st["offsets"]) - 1, "crossings", len(st["crossing_t"]))
seed_w = []
ok = 0
for k, (a, b) in enumerate(cols):
    pos, lev = decode_center_ray(ph[a, b], valid[a, b].astype(bool), anchor=192)
    exp_origin = (slab_origin + a * f[1] + b * f[2])[::-1]
    # find the store ray with this origin
    idx = [i for i in range(len(st["offsets"]) - 1) if np.allclose(st["ray_origin_zyx"][i], exp_origin, atol=1e-3)]
    if not idx:
        print(k, (a, b), "no ray in store (crossings", len(pos), ")"); continue
    i = idx[0]
    t = st["crossing_t"][st["offsets"][i]:st["offsets"][i + 1]]
    L = st["crossing_level"][st["offsets"][i]:st["offsets"][i + 1]]
    same = len(t) == len(pos) and np.allclose(t, pos, atol=0.02) and np.array_equal(L, lev)
    ok += same
    print(k, (a, b), "crossings", len(pos), "store", len(t), "match" if same else "DIFF", np.round(pos[:4], 1).tolist())
print("matched", ok, "of", len(cols))
