"""Snap every crossing of a winding-model crossing store onto the nearest crest of the CT (writes a new store).

villa's exporter registers a ray's crossings at its seed, so they sit at the seed's phase. Each crossing moves to the
nearest crest of the CT along the ray (level 0 every 0.5 voxel, smoothed by --sigma; peaks of prominence >= --prominence
of the ray's p95 - p5, >= 4 voxels apart, as in sheet_hits.py) within --window of its local gap; a crest claimed twice
goes to the nearer crossing; snaps that swap crossings or leave them closer than --min-gap are undone. Levels do not
change. With --drop-unsnapped only snapped crossings that agree with the CT's crest count are kept: each run of >= 2
snapped crossings with the same crest number minus level (as many crests as the model has windings between them)
becomes a ray of its own, so the phase term (phase_patch.py) pulls windings onto crests only. Checksums and the
fingerprint are rewritten as villa's exporter writes them; manifest["crest_snap"] has the statistics.

Usage: python snap_store.py <store> <level-0 CT: local zarr v2 dir> <out> [--window 0.5] [--prominence 0.15]
       [--min-gap 4] [--sigma 1] [--drop-unsnapped]
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vc  # noqa: E402  (local zarr v2 reader, chunk cache)


def sample(vol, pts):
    """Trilinear samples of a local uint8 vc.Volume at zyx points; 0 outside the array."""
    cs, shape = np.array(vol.chunks), np.array(vol.shape)
    p0 = np.floor(pts).astype(np.int64)
    inside = np.all((p0 >= 0) & (p0 < shape - 1), axis=1)
    p0 = np.clip(p0, 0, shape - 2)
    f = np.clip(pts - p0, 0, 1)
    acc = np.zeros(len(pts), np.float32)
    for corner in np.ndindex(2, 2, 2):
        q = p0 + corner
        ci, li = q // cs, q % cs
        v = np.zeros(len(q), np.float32)
        keys, inv = np.unique(ci, axis=0, return_inverse=True)
        for n, k in enumerate(keys):
            ch = vol._chunk(tuple(int(x) for x in k))
            if ch is not None:
                idx = np.flatnonzero(inv.ravel() == n)
                v[idx] = ch[li[idx, 0], li[idx, 1], li[idx, 2]]
        acc += np.prod(np.where(corner, f, 1 - f), axis=1) * v
    return np.where(inside, acc, 0.0)


def crests(prof, inside, ts, prominence):
    """Positions along the ray (voxels, to a tenth by a parabola) of the crests of a smoothed profile sampled at ts."""
    if inside.sum() < 20:
        return np.zeros(0)
    rng = np.percentile(prof[inside], 95) - np.percentile(prof[inside], 5)
    if rng <= 0:
        return np.zeros(0)
    pk, _ = find_peaks(np.where(inside, prof, prof[inside].min()), prominence=prominence * rng, distance=8)
    pk = pk[inside[pk] & (pk > 0) & (pk < len(prof) - 1)]
    a, b, c = prof[pk - 1], prof[pk], prof[pk + 1]
    den = a - 2 * b + c
    frac = np.where(den < 0, 0.5 * (a - c) / np.where(den < 0, den, -1.0), 0.0)
    return ts[pk] + np.clip(frac, -0.5, 0.5) * (ts[1] - ts[0])


def snap_ray(t, tc, window, min_gap, fixed=None):
    """New positions for the ascending crossings t given the ascending crests tc: each to its nearest crest within
    window * its local gap, one crossing per crest, order kept, none closer than min_gap unless the model's own
    crossings are; crossings in `fixed` stay. Returns (t_new, snapped mask, counts of 'no_crest', 'conflict', 'undone')."""
    t_new, moved = t.copy(), np.zeros(len(t), bool)
    why = {"no_crest": 0, "conflict": 0, "undone": 0}
    free = np.ones(len(t), bool) if fixed is None else ~np.asarray(fixed, bool)
    if len(t) < 2 or not len(tc):
        why["no_crest"] += int(free.sum())
        return t_new, moved, why
    gaps = np.diff(t)
    g = np.r_[gaps[0], 0.5 * (gaps[:-1] + gaps[1:]), gaps[-1]]
    j = np.zeros(len(t), int)
    if len(tc) > 1:
        j = np.clip(np.searchsorted(tc, t), 1, len(tc) - 1)
        j = np.where(np.abs(tc[j - 1] - t) <= np.abs(tc[j] - t), j - 1, j)
    d = tc[j] - t
    reach = np.abs(d) <= window * g
    why["no_crest"] += int((free & ~reach).sum())
    ok = free & reach
    for jj in np.unique(j[ok]):  # a crest claimed twice goes to the nearer crossing
        k = np.flatnonzero(ok & (j == jj))
        if len(k) > 1:
            lose = k[np.argsort(np.abs(d[k]), kind="stable")[1:]]
            ok[lose] = False
            why["conflict"] += len(lose)
    t_new[ok] = tc[j[ok]]
    moved = ok.copy()
    while True:  # undo snaps that swap two crossings or leave them closer than min_gap (the one that moved more)
        for k in np.flatnonzero(np.diff(t_new) < min_gap):
            cand = [x for x in (k, k + 1) if moved[x]]
            if cand:
                x = max(cand, key=lambda q: abs(t_new[q] - t[q]))
                t_new[x], moved[x] = t[x], False
                why["undone"] += 1
                break
        else:
            return t_new, moved, why


def consistent_runs(tc, t_new, moved, levels, min_len=2):
    """Runs of snapped crossings, in order along the ray, with the same crest number minus winding level; runs of at
    least min_len crossings, as arrays of crossing indices."""
    k = np.flatnonzero(moved)
    if len(k) < min_len:
        return []
    diff = np.searchsorted(tc, t_new[k]) - np.asarray(levels)[k]
    return [r for r in np.split(k, np.flatnonzero(np.diff(diff) != 0) + 1) if len(r) >= min_len]


def contrast(prof, ts, t):
    """sheet_hits.py's c (CT at a crossing minus the mean at the mid-points to its neighbours, / p95 - p5)."""
    if len(t) < 3:
        return np.zeros(0)
    rng = np.percentile(prof, 95) - np.percentile(prof, 5)
    m = 0.5 * (t[1:] + t[:-1])
    return (np.interp(t[1:-1], ts, prof) - 0.5 * (np.interp(m[:-1], ts, prof) + np.interp(m[1:], ts, prof))) / max(rng, 1e-6)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("store")
    ap.add_argument("volume", help="local level-0 zarr v2 directory (or a multiscale group with 0/)")
    ap.add_argument("out")
    ap.add_argument("--window", type=float, default=0.5, help="reach, as a fraction of the crossing's local gap")
    ap.add_argument("--prominence", type=float, default=0.15, help="crest prominence, fraction of the ray's p95 - p5")
    ap.add_argument("--min-gap", type=float, default=4.0, help="voxels")
    ap.add_argument("--sigma", type=float, default=1.0, help="smoothing of the CT profile (voxels)")
    ap.add_argument("--drop-unsnapped", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    vol = vc.Volume.open(a.volume)
    man = json.load(open(os.path.join(a.store, "manifest.json")))
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.store, a.out)
    st = dict(rays=0, skipped=0, crossings=0, snapped=0, no_crest=0, conflict=0, undone=0, kept=0, rays_out=0, near3=0)
    shift, cgap, cb, ca = [], [], [], []
    for shard in man["shards"]:
        d = os.path.join(a.out, shard["name"])
        arr = {k: np.load(os.path.join(d, v["file"])) for k, v in shard["arrays"].items()}
        O, S = arr["ray_origin_zyx"].astype(np.float64), arr["ray_step_zyx"].astype(np.float64)
        T, off, lev = arr["crossing_t"].astype(np.float64).copy(), arr["crossing_offsets"], arr["crossing_level"]
        runs = {}
        for i in np.lexsort((O[:, 2], O[:, 1], O[:, 0])):  # spatial order keeps the chunk cache warm
            st["rays"] += 1
            L, n = float(np.linalg.norm(S[i])), int(off[i + 1] - off[i])
            if L <= 0 or n < 2:
                st["skipped"] += 1
                continue
            t = T[off[i]:off[i + 1]] * L  # voxels from the origin, ascending
            ts = np.arange(t.min() - 12.0, t.max() + 12.0, 0.5)
            raw = sample(vol, O[i][None] + ts[:, None] * (S[i] / L)[None])
            inside = raw > 0
            if inside.sum() < 20:
                st["skipped"] += 1
                continue
            prof = gaussian_filter1d(raw, a.sigma / 0.5)
            tc = crests(prof, inside, ts, a.prominence)
            t_new, moved, why = snap_ray(t, tc, a.window, a.min_gap, fixed=np.interp(t, ts, inside.astype(float)) < 0.99)
            for k in why:
                st[k] += why[k]
            st["crossings"] += n
            st["snapped"] += int(moved.sum())
            shift += (t_new - t)[moved].tolist()
            if len(tc) > 1:
                cgap += np.diff(tc).tolist()
                st["near3"] += int((np.abs(t[:, None] - tc[None]).min(1) <= 3.0).sum())
            cb += contrast(prof, ts, t).tolist()
            ca += contrast(prof, ts, t_new).tolist()
            T[off[i]:off[i + 1]] = t_new / L
            if a.drop_unsnapped:
                runs[i] = [off[i] + r for r in consistent_runs(tc, t_new, moved, lev[off[i]:off[i + 1]])]
                st["kept"] += int(sum(len(r) for r in runs[i]))
            if st["rays"] % 2000 == 0:
                print(f"{st['rays']} rays, {st['snapped']} of {st['crossings']} crossings snapped, {time.time() - t0:.0f} s",
                      flush=True)
        new = {"crossing_t": T.astype(arr["crossing_t"].dtype)}
        if a.drop_unsnapped:
            rr = [(i, r) for i in range(len(O)) for r in runs.get(i, [])]
            ray = np.array([i for i, _ in rr], np.int64)
            idx = np.concatenate([r for _, r in rr]) if rr else np.zeros(0, np.int64)
            new = {"ray_origin_zyx": arr["ray_origin_zyx"][ray], "ray_step_zyx": arr["ray_step_zyx"][ray],
                   "seed_winding": arr["seed_winding"][ray], "crossing_t": T[idx].astype(arr["crossing_t"].dtype),
                   "crossing_level": lev[idx],
                   "crossing_offsets": np.r_[0, np.cumsum([len(r) for _, r in rr])].astype(off.dtype)}
            st["rays_out"] += len(rr)
            shard["num_crossings"], shard["num_retained_rays"] = int(len(idx)), len(rr)
        for name, value in new.items():
            path = os.path.join(d, shard["arrays"][name]["file"])
            np.save(path, np.ascontiguousarray(value), allow_pickle=False)
            shard["arrays"][name].update(bytes=os.path.getsize(path), sha256=sha(path), shape=list(value.shape))
    nc, sh_, cg = max(st["crossings"], 1), np.abs(shift), np.asarray(cgap)
    fpos = lambda c: round(float((np.asarray(c) > 0).mean()), 4) if len(c) else None  # noqa: E731
    summary = {"tool": "snap_store.py", "window": a.window, "prominence": a.prominence, "min_gap_vox": a.min_gap,
               "sigma_vox": a.sigma, "drop_unsnapped": a.drop_unsnapped, "rays": st["rays"], "rays_skipped": st["skipped"],
               "crossings": st["crossings"], "snapped_frac": round(st["snapped"] / nc, 4),
               "no_crest_frac": round(st["no_crest"] / nc, 4), "conflict_frac": round(st["conflict"] / nc, 4),
               "undone_frac": round(st["undone"] / nc, 4), "within_3vox_of_a_crest_before_frac": round(st["near3"] / nc, 4),
               "abs_shift_vox": {q: round(float(np.percentile(sh_, q)), 2) for q in (25, 50, 75, 90)} if len(sh_) else None,
               "ct_crest_gap_vox_median": round(float(np.median(cg)), 2) if len(cg) else None,
               "frac_c_pos_before": fpos(cb), "frac_c_pos_after": fpos(ca), "seconds": round(time.time() - t0, 1)}
    if a.drop_unsnapped:
        summary.update(kept_frac=round(st["kept"] / nc, 4), count_inconsistent_frac=round((st["snapped"] - st["kept"]) / nc, 4),
                       rays_out=st["rays_out"])
        man["num_crossings"] = int(sum(s["num_crossings"] for s in man["shards"]))
        man["num_rays"] = st["rays_out"]
    man["crest_snap"] = summary
    ident = copy.deepcopy(man)  # villa's fingerprint: the manifest without these keys
    for k in ("fingerprint", "elapsed_seconds", "export_workers", "rays_per_task"):
        ident.pop(k, None)
    for s in ident["shards"]:
        s.pop("elapsed_seconds", None)
    man["fingerprint"] = hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    open(os.path.join(a.out, "manifest.json"), "w").write(json.dumps(man, indent=2, sort_keys=True) + "\n")
    print("CREST_SNAP " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
