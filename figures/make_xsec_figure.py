"""Cross-section figure (dev): fitted windings drawn over one CT slice, same scale for every panel.

Usage: python xsec_figure.py out.png  (panels are defined below: mesh dir, volume, cache, umbilicus, z, centre)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/home/claude/vc/release/tools")
from phase_align import Sampler, load_mesh, open_level0  # noqa: E402

S = "/tmp/claude-0/-home-claude/f453f88b-9bf6-506d-8057-7d6307920b87/scratchpad"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def plane_polyline(P, v, zc):
    xx, yy, zz = P[..., 0], P[..., 1], P[..., 2]
    d = zz - zc
    s = (d[:-1] * d[1:] <= 0) & v[:-1] & v[1:] & (d[:-1] != d[1:])
    has = s.any(0)
    r = np.argmax(s, 0)
    c = np.arange(zz.shape[1])
    rx = np.minimum(r + 1, zz.shape[0] - 1)
    t = d[r, c] / np.where(has, d[r, c] - d[rx, c], 1)
    return (np.where(has, xx[r, c] + t * (xx[rx, c] - xx[r, c]), np.nan),
            np.where(has, yy[r, c] + t * (yy[rx, c] - yy[r, c]), np.nan))


def panel(ax, mesh_dir, windings, vol, cache, zc, cy, cx, half, title):
    smp = Sampler(open_level0(vol, cache))
    y0, x0 = int(cy) - half, int(cx) - half
    yy, xx = np.meshgrid(np.arange(y0, y0 + 2 * half), np.arange(x0, x0 + 2 * half), indexing="ij")
    img = smp.sample(np.stack([np.full(yy.size, float(zc)), yy.ravel().astype(float), xx.ravel().astype(float)], 1))
    ax.imshow(img.reshape(2 * half, 2 * half), cmap="gray", extent=(x0, x0 + 2 * half, y0 + 2 * half, y0))
    cols = plt.cm.tab10(np.arange(10))
    for w in windings:
        d = f"{mesh_dir}/w{w:03d}"
        if not os.path.exists(d + "/x.tif"):
            continue
        P, v = load_mesh(d)
        px, py = plane_polyline(P, v, zc)
        ax.plot(px, py, lw=1.4, color=cols[w % 10])
    ax.set_xlim(x0, x0 + 2 * half)
    ax.set_ylim(y0 + 2 * half, y0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def umb_at(path, z):
    cp = sorted(json.load(open(path))["control_points"], key=lambda q: q["z"])
    return (float(np.interp(z, [q["z"] for q in cp], [q["y"] for q in cp])),
            float(np.interp(z, [q["z"] for q in cp], [q["x"] for q in cp])))


if __name__ == "__main__":
    out = sys.argv[1]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), dpi=110)
    # 1. curated PHerc0139 windings (ground truth)
    P, v = load_mesh(f"{S}/crop/ctrl0139/w037")
    z0 = 5500
    dz = np.where(v, np.abs(P[..., 2] - z0), np.inf)
    rr, cc = np.nonzero(dz < 2.0)
    k = len(rr) // 2  # a vertex of w037 on the slice, half-way along it
    cx, cy = P[rr[k], cc[k], 0], P[rr[k], cc[k], 1]
    panel(axs[0], f"{S}/crop/ctrl0139", range(34, 42), f"{S3}/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr",
          f"{S}/ct0139", z0, cy, cx, 160, "PHerc0139, curated windings w034-w041 (z 5500)")
    # 2. armando-gaona's published recipe fit, PHerc0211
    cy, cx = umb_at("/home/claude/vc/data/PHerc0211_umbilicus.json", 8500)
    panel(axs[1], "/home/claude/vc/ext/armando/meshes", range(0, 140), f"{S3}/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr",
          f"{S}/ct0cache0211", 8500, cy + 1000 * np.sin(np.radians(240)), cx + 1000 * np.cos(np.radians(240)), 160,
          "PHerc0211, published recipe fit (armando-gaona), z 8500, r 1000")
    # 3. rodriguescarson's published recipe fit, PHerc0826
    cy, cx = umb_at("/home/claude/vc/data/PHerc0826_umbilicus.json", 9328)
    panel(axs[2], f"{S}/rc0826_8928/meshes", range(0, 90), f"{S3}/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr",
          f"{S}/ct0cache0826", 9328, cy + 1000 * np.sin(np.radians(0)), cx + 1000 * np.cos(np.radians(0)), 160,
          "PHerc0826, published recipe fit (rodriguescarson), z 9328, r 1000")
    fig.suptitle("Fitted windings over the CT, full resolution, 320 x 320 voxels (3 mm): curated windings follow the sheets; the First Letters recipe's do not", fontsize=11)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
