"""One render + ink job: tifxyz mesh -> 28-layer surface volume (render_sv.py) -> ink_9um (koine_machines infer.py,
both layer orders) -> PNGs and one JSON line of statistics. Also the control: a crop of a segment with published
ink labels, rendered by us and taken from the published surface volume, scored against the labels (AUC).

Runs in its own environment (python of a venv that has koine_machines' dependencies); the notebook kernel only
launches it, so the kernel's packages are never changed.

Usage:
  python inkjob.py job --mesh DIR --volume SRC --prefix OUT/tag --work DIR --ckpt CKPT [--png-scale 0.5] [--cache DIR]
         [--umbilicus U.json]   (reports normal_outward_frac and the pre-registered primary_direction)
         [--flatten-lasagna VILLA/lasagna]   (renders the Lasagna-flattened mesh; reports the grid distortion before/after)
  python inkjob.py control --seg URL --mesh-name NAME --sv-name NAME --volume URL --box r0 r1 c0 c1
         --labels PNG --supervision PNG --prefix OUT/control --work DIR --ckpt CKPT [--cache DIR]
The ink model runs on the visible CUDA device if there is one (the caller sets CUDA_VISIBLE_DEVICES), else on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import tifffile
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import render_sv  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_png(a, path, pct=(1, 99), scale=1.0):
    a = np.asarray(a).astype(np.float32)
    nz = a[a > 0]
    lo, hi = np.percentile(nz, pct) if nz.size else (0, 255)
    img = Image.fromarray(((a - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8))
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    img.save(path, optimize=True)


def auc(scores, labels):
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    s = np.asarray(scores, np.float64).ravel()
    y = np.asarray(labels, bool).ravel()
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return None
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]  # 1-based average ranks, ties shared
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def infer(sv_zarr, ckpt, out_tif):
    cmd = [sys.executable, "-m", "koine_machines.inference.infer", sv_zarr, ckpt, out_tif, "--direction", "both",
           "--no-compile", "--batch-size", "32", "--num-workers", "4"]
    if DEVICE == "cuda":
        cmd += ["--gpus", "0"]
    t = time.time()
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError("ink inference failed:\n" + p.stdout[-3000:])
    base = out_tif[:-4]
    return round(time.time() - t, 1), {"forward": out_tif, "reverse": base + "_reverse.tif"}


def normal_orientation(mesh_dir, umbilicus_json):
    """Fraction of grid vertices whose render normal n = cross(dR, dC) points away from the scroll axis.

    Used to choose the ink model's layer order before looking at its output (no test-time selection): the control
    segment (PHerc0139 w035; its 2.399 um copy against that scan's umbilicus) has n pointing inward everywhere
    (outward fraction 0.0005) and ink_9um reads it in the 'forward' order (AUC 0.97; 'reverse' 0.51). fit_spiral's
    CW exports also point inward (published PHerc0211 fit: 0.005-0.018)."""
    x, y, z = (tifffile.imread(os.path.join(mesh_dir, f"{c}.tif")).astype(np.float64) for c in "xyz")
    P = np.stack([x, y, z], -1)
    ok = (x > 0) & (y > 0) & (z > 0)
    dR = np.zeros_like(P)
    dC = np.zeros_like(P)
    dR[1:-1] = P[2:] - P[:-2]
    dC[:, 1:-1] = P[:, 2:] - P[:, :-2]
    n = np.cross(dR, dC)
    m = ok.copy()
    m[0] = m[-1] = False
    m[:, 0] = m[:, -1] = False
    m[1:-1, 1:-1] &= ok[2:, 1:-1] & ok[:-2, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2]
    cp = sorted(json.load(open(umbilicus_json))["control_points"], key=lambda q: q["z"])
    uz, uy, ux = (np.array([q[k] for q in cp], float) for k in ("z", "y", "x"))
    rx = P[..., 0] - np.interp(P[..., 2], uz, ux)
    ry = P[..., 1] - np.interp(P[..., 2], uz, uy)
    dot = n[..., 0] * rx + n[..., 1] * ry
    return float((dot[m] > 0).mean()) if m.any() else None


def grid_distortion(mesh_dir):
    """Neighbouring-vertex distances / nominal grid step (p5, p50, p95) along columns and rows, and the p5 of the
    angle between the grid directions: 1, 1 and 90 degrees for an isometric parametrisation."""
    x, y, z = (tifffile.imread(os.path.join(mesh_dir, f"{c}.tif")).astype(np.float64) for c in "xyz")
    P = np.stack([z, y, x], -1)
    ok = (x > 0) & (y > 0) & (z > 0)
    step = 1.0 / json.load(open(os.path.join(mesh_dir, "meta.json")))["scale"][0]
    dc = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[ok[:, 1:] & ok[:, :-1]] / step
    dr = np.linalg.norm(P[1:] - P[:-1], axis=-1)[ok[1:] & ok[:-1]] / step
    u, v = P[1:, :-1] - P[:-1, :-1], P[:-1, 1:] - P[:-1, :-1]
    m = ok[1:, :-1] & ok[:-1, 1:] & ok[:-1, :-1]
    cosang = np.abs((u * v).sum(-1)[m]) / (np.linalg.norm(u, axis=-1)[m] * np.linalg.norm(v, axis=-1)[m] + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    r2 = lambda q: [round(float(t), 2) for t in q]  # noqa: E731
    return {"columns_p5_p50_p95": r2(np.percentile(dc, [5, 50, 95])), "rows_p5_p50_p95": r2(np.percentile(dr, [5, 50, 95])),
            "angle_p5_deg": round(float(np.percentile(ang, 5)), 1)}


def lasagna_flatten(mesh_dir, lasagna_dir, work, device):
    """Flatten a tifxyz mesh with villa's Lasagna forward flattener (configs/flatten_fast_nofilter.json, the one
    spiral-fitting/render_ink.py uses), without torch.compile and the fused Triton Adam (portable; a winding takes
    about a minute). Returns the flattened tifxyz directory."""
    cfg = json.load(open(os.path.join(lasagna_dir, "configs", "flatten_fast_nofilter.json")))
    for st in cfg["stages"]:
        st["args"].update({"compile_flatten": False, "compile_flatten_combined": False, "fused_flatten_adam_clamp": False,
                           "auto_steps_max": 20000})
    os.makedirs(work, exist_ok=True)
    cfg_path, ov_path = os.path.join(work, "flatten.json"), os.path.join(work, "overlay.json")
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    json.dump({"external_surfaces": [{"path": os.path.abspath(mesh_dir)}]}, open(ov_path, "w"))
    out = os.path.join(work, "lasagna")
    p = subprocess.run([sys.executable, "fit.py", cfg_path, ov_path, "--out-dir", out, "--device", device], cwd=lasagna_dir,
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    flat = os.path.join(out, "tifxyz", "flatten.tifxyz")
    if p.returncode != 0 or not os.path.exists(os.path.join(flat, "x.tif")):
        raise RuntimeError("flattening failed:\n" + p.stdout[-3000:])
    return flat


def job(a):
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    orient = normal_orientation(a.mesh, a.umbilicus) if a.umbilicus else None
    flat_info = None
    if a.flatten_lasagna:
        before = grid_distortion(a.mesh)
        flat = lasagna_flatten(a.mesh, a.flatten_lasagna, os.path.join(a.work, "flat"), DEVICE)
        flat_info = {"t_flatten_s": round(time.time() - t0, 1), "before": before, "after": grid_distortion(flat)}
        a.mesh = flat
    r = render_sv.Renderer(a.mesh, render_sv.open_volume(a.volume, a.cache), DEVICE)
    sv = r.render()
    zarr_path = os.path.join(a.work, "sv.zarr")
    render_sv.write_ome_zarr(zarr_path, sv)
    tc = render_sv.tile_contrast(sv)
    valid = sv[13] > 0
    res = {"canvas": [r.H, r.W], "valid_frac": round(float(valid.mean()), 4), "boxes": r.boxes,
           "flatten": flat_info,
           "normal_outward_frac": None if orient is None else round(orient, 4),
           "primary_direction": None if orient is None else ("forward" if orient < 0.5 else "reverse"),
           "t_render_s": round(time.time() - t0, 1), "tiles": len(tc),
           "contrast_median": round(float(np.median(tc)), 4) if tc else None}
    to_png(sv[13], a.prefix + "_layer13.png", scale=a.png_scale)
    # ink sometimes shows directly in the render as bright areas near the surface: max over layers 10-16
    to_png(sv[10:17].max(0), a.prefix + "_max10-16.png", pct=(1, 99.5), scale=a.png_scale)
    del sv
    res["t_ink_s"], preds = infer(zarr_path, a.ckpt, os.path.join(a.work, "pred.tif"))
    for d, f in preds.items():
        p = tifffile.imread(f)
        res[f"ink_{d}_mean"] = round(float(p[valid].mean()), 2) if valid.any() else None
        res[f"ink_{d}_p99"] = round(float(np.percentile(p[valid], 99)), 2) if valid.any() else None
        to_png(p, f"{a.prefix}_ink_{d}.png", pct=(0.5, 99.5), scale=a.png_scale)
    shutil.rmtree(a.work, ignore_errors=True)
    print(json.dumps(res), flush=True)


def control(a):
    import requests
    from vcz import ZArray
    os.makedirs(a.work + "/mesh", exist_ok=True)
    for f in ["meta.json", "x.tif", "y.tif", "z.tif"]:
        open(f"{a.work}/mesh/{f}", "wb").write(requests.get(f"{a.seg}/mesh/{a.mesh_name}/{f}", timeout=300).content)
    r0, r1, c0, c1 = a.box
    lab = np.array(Image.open(a.labels).convert("L")) > 0
    sup = np.array(Image.open(a.supervision).convert("L")) > 0
    res = {"box": a.box, "device": DEVICE}
    pub = ZArray(f"{a.seg}/surface-volumes/{a.sv_name}/0", cache_dir=a.cache).read(0, 28, r0, r1, c0, c1)
    t = time.time()
    rd = render_sv.Renderer(a.work + "/mesh", render_sv.open_volume(a.volume, a.cache), DEVICE)
    ours = rd.render((r0, r1), (c0, c1))
    res["t_render_s"] = round(time.time() - t, 1)

    def corr(x, y):
        x = x.astype(np.float64) - x.mean(); y = y.astype(np.float64) - y.mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-9))
    res["corr_per_layer_ours_vs_published"] = [round(corr(pub[L], ours[L]), 3) for L in range(28)]
    for name, stack in (("published", pub), ("ours", ours)):
        zp = f"{a.work}/{name}.zarr"
        render_sv.write_ome_zarr(zp, stack)
        to_png(stack[13], f"{a.prefix}_{name}_layer13.png")
        res[f"{name}_ink_s"], preds = infer(zp, a.ckpt, f"{a.work}/{name}_pred.tif")
        for d, f in preds.items():
            p = tifffile.imread(f)
            v = auc(p[sup], lab[sup])
            res[f"{name}_{d}_auc_labelled_area"] = round(v, 3) if v is not None else None
            to_png(p, f"{a.prefix}_{name}_ink_{d}.png", pct=(0.5, 99.5))
    Image.fromarray(lab.astype(np.uint8) * 255).save(f"{a.prefix}_labels.png")
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--mesh", required=True)
    j.add_argument("--volume", required=True)
    j.add_argument("--prefix", required=True)
    j.add_argument("--work", required=True)
    j.add_argument("--ckpt", required=True)
    j.add_argument("--png-scale", type=float, default=1.0)
    j.add_argument("--cache", default=None)
    j.add_argument("--umbilicus", default=None, help="umbilicus JSON: sets primary_direction from the normal orientation")
    j.add_argument("--flatten-lasagna", default=None, help="villa's lasagna/ directory: flatten the mesh before rendering")
    c = sub.add_parser("control")
    for k in ("--seg", "--mesh-name", "--sv-name", "--volume", "--labels", "--supervision", "--prefix", "--work", "--ckpt"):
        c.add_argument(k, required=True)
    c.add_argument("--box", type=int, nargs=4, required=True)
    c.add_argument("--cache", default=None)
    a = ap.parse_args()
    job(a) if a.cmd == "job" else control(a)


if __name__ == "__main__":
    main()
