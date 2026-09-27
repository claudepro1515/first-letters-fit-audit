"""Number of winding-model seed rays infer_winding_volume.py would cast for each seed spacing.

Usage: python count_seeds.py <fit checkpoint> <meshes dir> <umbilicus.json> <first> <last> <step> <spacing>...
Prints one JSON line per spacing: {"spacing": s, "slabs": n}.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE + "/stub", HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import vesuvius.neural_tracing.winding_models.infer_winding_volume as iwv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("meshes")
ap.add_argument("umbilicus")
ap.add_argument("first", type=int)
ap.add_argument("last", type=int)
ap.add_argument("step", type=int)
ap.add_argument("spacings", type=float, nargs="+")
a = ap.parse_args()
for s in a.spacings:
    ns = argparse.Namespace(fit_checkpoint=a.ckpt, meshes_dir=a.meshes, z_range=None, winding_range=[a.first, a.last],
                            winding_step=a.step, seed_spacing=s, umbilicus=a.umbilicus)
    rays = iwv.build_seed_rays_from_meshes(ns)
    print(json.dumps({"spacing": s, "slabs": int(len(rays["seed_xyz"]))}), flush=True)
