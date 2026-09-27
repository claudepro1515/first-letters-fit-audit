"""Ground truth for tools/phase_align.py on curated windings (dev script).

ROOT holds the curated PHerc0139 windings w034-w041 as published (ROOT/ctrl0139/wNNN, tifxyz, cut to z 4800-6200 for
speed) and, next to them, test variants moved along their normals and then aligned (ROOT/<variant>/wNNN). For every
vertex of an aligned interior winding (w035-w040): the point-to-plane distance to the nearest vertex of each curated
winding. 'own' = within 3 voxels of its own curated winding, 'other' = within 3 voxels of another curated winding (a
neighbouring sheet), 'none' = neither (between sheets). Also the share of the vertices on some curated sheet that are
not on the sheet most of their winding is on (sheet jumps).

Usage: PA_GT_ROOT=<dir> python phase_align_groundtruth.py <variant> [<variant> ...]
"""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, "/home/claude/vc/release/tools")
from phase_align import load_mesh, normals
S = os.environ.get("PA_GT_ROOT", "/tmp/claude-0/-home-claude/f453f88b-9bf6-506d-8057-7d6307920b87/scratchpad/crop")
W = [f"w{k:03d}" for k in range(34, 42)]
cur = {}
for w in W:
    P, v = load_mesh(f"{S}/ctrl0139/{w}")
    n, ok = normals(P, v)
    cur[w] = (P[ok], n[ok], cKDTree(P[ok]))


def plane_dist(q, w):
    P, n, t = cur[w]
    d, i = t.query(q, k=3)
    return np.abs(((q[:, None, :] - P[i]) * n[i]).sum(-1)).min(1) * (d[:, 0] < 30)  + (d[:, 0] >= 30) * 1e9


for variant in sys.argv[1:]:
    tot = {"own": 0, "other": 0, "none": 0}
    split = []
    for w in W[1:-1]:
        P, v = load_mesh(f"{S}/{variant}/{w}")
        q = P[v][::7]
        dist = np.stack([plane_dist(q, x) for x in W], 1)
        own = dist[:, W.index(w)] < 3
        other = (~own) & (np.delete(dist, W.index(w), 1).min(1) < 3)
        tot["own"] += int(own.sum()); tot["other"] += int(other.sum()); tot["none"] += int((~own & ~other).sum())
        # sheet jumps: of the vertices on some curated sheet, the share not on the sheet most of this winding is on
        on = dist.min(1) < 3
        if on.any():
            which = dist[on].argmin(1)
            split.append(float((which != np.bincount(which).argmax()).mean()))
    n_ = sum(tot.values())
    print(json.dumps({"variant": variant, "vertices": n_, **{k: round(v / n_, 3) for k, v in tot.items()},
                      "off_majority_sheet_frac": round(float(np.mean(split)), 3)}), flush=True)
