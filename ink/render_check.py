"""Render a crop of a tifxyz segment ourselves and compare it with the published surface volume.

Control: PHerc0139 w043 (20260112000000), 9.362 um mesh (grid step 20) and its published 28-layer surface
volume on the same canvas (6120 x 8120). For each pixel the mesh is interpolated bilinearly, the normal is
the cross product of the canvas-axis derivatives, and the CT is sampled trilinearly at offsets k * n.
Prints the Pearson correlation of every (published layer, rendered offset) pair for both normal signs,
so the published convention (sign, which offset is layer 0) can be read off.
"""
import json
import os
import sys

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

sys.path.insert(0, "/home/claude/vc/release/tools")
from vcz import OmeZarr  # noqa: E402

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SEG = "PHerc0139/segments/20260112000000-w043_2026011217"
SV = OmeZarr(f"{S3}/{SEG}/surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr", cache_dir="/home/claude/vc/ink/cache_sv").level("0")
CT = OmeZarr(f"{S3}/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr", cache_dir="/home/claude/vc/ink/cache_ct").level("0")
D = "/home/claude/vc/ink/ctrl"
X, Y, Z = (tifffile.imread(f"{D}/{c}.tif").astype(np.float64) for c in "xyz")
step = 1.0 / json.load(open(f"{D}/meta.json"))["scale"][0]  # 20
r0, c0, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 256
K = 20  # render offsets -K..K


def interp(grid, rr, cc):
    return map_coordinates(grid, [rr, cc], order=1, mode="nearest")


rr, cc = np.meshgrid(np.arange(r0, r0 + n) / step, np.arange(c0, c0 + n) / step, indexing="ij")
P = np.stack([interp(G, rr, cc) for G in (X, Y, Z)], -1)  # xyz
h = 0.5
dR = np.stack([interp(G, rr + h, cc) - interp(G, rr - h, cc) for G in (X, Y, Z)], -1)
dC = np.stack([interp(G, rr, cc + h) - interp(G, rr, cc - h) for G in (X, Y, Z)], -1)
N = np.cross(dC, dR)
N /= np.linalg.norm(N, axis=-1, keepdims=True)
pub = SV.read(0, 28, r0, r0 + n, c0, c0 + n).astype(np.float64)
print("published crop", pub.shape, "mean", pub.mean().round(1))
ks = np.arange(-K, K + 1)
pts = P[None] + ks[:, None, None, None] * N[None]  # [2K+1, n, n, 3] xyz
zyx = pts[..., ::-1]
lo = np.floor(zyx.reshape(-1, 3).min(0)).astype(int) - 1
hi = np.ceil(zyx.reshape(-1, 3).max(0)).astype(int) + 2
box = CT.read(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]).astype(np.float64)
rend = map_coordinates(box, [(zyx[..., i] - lo[i]).ravel() for i in range(3)], order=1).reshape(zyx.shape[:-1])
print("rendered", rend.shape, "box", box.shape)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


for sign in (1, -1):
    R = rend if sign == 1 else rend[::-1]
    best = []
    for L in range(28):
        cs = [corr(pub[L], R[j]) for j in range(len(ks))]
        j = int(np.argmax(cs))
        best.append((L, int(ks[j]) * sign, round(cs[j], 3)))
    print("normal sign", sign, "published layer -> best offset (along +N if sign 1), corr:", best)
