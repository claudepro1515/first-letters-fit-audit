"""Tests of wm/snap_store.py: crossings move onto the CT's crests, one per crest, in order; the rewritten store loads
in villa's WindingInferenceStore (checksums and fingerprint) when a villa checkout is available."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
WM = os.path.dirname(HERE)
sys.path.insert(0, WM)
from snap_store import snap_ray  # noqa: E402

SHEETS = np.arange(30.0, 120.0, 16.0)  # sheet positions along x (voxels)


def test_each_crossing_goes_to_its_nearest_crest():
    rng = np.random.default_rng(0)
    t = SHEETS + rng.uniform(-5, 5, len(SHEETS))
    t_new, moved, why = snap_ray(t, SHEETS, 0.5, 4.0)
    assert moved.all() and np.allclose(t_new, SHEETS)
    assert why == {"no_crest": 0, "conflict": 0, "undone": 0}


def test_out_of_reach_and_fixed_crossings_stay():
    t = SHEETS + 1.0
    tc = np.delete(SHEETS, 1)  # no crest near crossing 1: the nearest is 15-17 voxels away, beyond half its gap
    fixed = np.zeros(len(t), bool)
    fixed[3] = True
    t_new, moved, why = snap_ray(t, tc, 0.5, 4.0, fixed=fixed)
    assert not moved[1] and t_new[1] == t[1] and not moved[3] and t_new[3] == t[3]
    assert why["no_crest"] == 1 and moved.sum() == len(t) - 2


def test_a_crest_goes_to_the_nearer_crossing_and_order_and_gap_are_kept():
    tc = np.array([30.0, 46.0, 62.0])
    t = np.array([28.0, 33.0, 47.0])  # the first two both reach the crest at 30; 28 is nearer
    t_new, moved, why = snap_ray(t, tc, 0.5, 4.0)
    assert why["conflict"] == 1
    # 28 -> 30 would leave 3 voxels to the unsnapped 33: undone, both stay; 47 -> 46
    assert np.allclose(t_new, [28.0, 33.0, 46.0]) and moved.tolist() == [False, False, True]
    assert why["undone"] == 1
    assert np.all(np.diff(t_new) > 0)


def test_the_models_own_close_crossings_are_left_alone():
    t = np.array([30.0, 32.0, 46.5])  # the model put two crossings 2 voxels apart; neither can snap to 30 twice
    t_new, moved, _ = snap_ray(t, np.array([30.0, 46.0]), 0.5, 4.0)
    assert t_new[0] == 30.0 and t_new[1] == 32.0 and t_new[2] == 46.0


def _zarr(path, vol, chunks=(16, 16, 64)):
    os.makedirs(path)
    json.dump({"zarr_format": 2, "shape": list(vol.shape), "chunks": list(chunks), "dtype": "|u1", "compressor": None,
               "fill_value": 0, "order": "C", "filters": None, "dimension_separator": "."},
              open(os.path.join(path, ".zarray"), "w"))
    for i in range(0, vol.shape[0], chunks[0]):
        for j in range(0, vol.shape[1], chunks[1]):
            for k in range(0, vol.shape[2], chunks[2]):
                c = np.zeros(chunks, np.uint8)
                b = vol[i:i + chunks[0], j:j + chunks[1], k:k + chunks[2]]
                c[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                open(os.path.join(path, f"{i // chunks[0]}.{j // chunks[1]}.{k // chunks[2]}"), "wb").write(c.tobytes())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _store(path, origins, steps, t_per_ray):
    os.makedirs(os.path.join(path, "shard_0"))
    t = np.concatenate(t_per_ray).astype(np.float32)
    arrays = {"ray_origin_zyx": np.asarray(origins, np.float32), "ray_step_zyx": np.asarray(steps, np.float32),
              "crossing_t": t, "crossing_level": np.concatenate([np.arange(len(x)) for x in t_per_ray]).astype(np.int16),
              "crossing_offsets": np.r_[0, np.cumsum([len(x) for x in t_per_ray])].astype(np.int64),
              "seed_winding": np.zeros(len(origins), np.int16)}
    desc = {}
    for k, v in arrays.items():
        f = os.path.join(path, "shard_0", k + ".npy")
        np.save(f, v, allow_pickle=False)
        desc[k] = {"file": k + ".npy", "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest(),
                   "shape": list(v.shape), "dtype": v.dtype.str, "bytes": os.path.getsize(f)}
    man = {"artifact_type": "winding_inference_crossings", "format_version": 1, "coordinate_order": "zyx",
           "num_rays": len(origins), "num_crossings": int(len(t)), "shards": [{"name": "shard_0", "arrays": desc}]}
    man["fingerprint"] = _digest(man)
    json.dump(man, open(os.path.join(path, "manifest.json"), "w"))


def test_end_to_end_on_a_synthetic_scan(tmp_path):
    x = np.arange(160, dtype=float)
    prof = 60 + 150 * np.exp(-0.5 * ((x[:, None] - SHEETS[None]) / 1.5) ** 2).sum(1)
    vol = np.broadcast_to(np.clip(prof, 0, 255).astype(np.uint8), (32, 32, 160)).copy()
    _zarr(str(tmp_path / "ct"), vol)
    rng = np.random.default_rng(1)
    origins, steps, ts = [], [], []
    for z in (8.0, 20.0):
        for y in (6.0, 18.0, 26.0):
            origins.append([z, y, 10.0])
            steps.append([0.0, 0.0, 2.0])  # 2 voxels per unit of t: the tool works in voxels, the store in t
            ts.append((SHEETS + rng.uniform(-5, 5, len(SHEETS)) - 10.0) / 2.0)
    _store(str(tmp_path / "store"), origins, steps, ts)
    out = subprocess.run([sys.executable, os.path.join(WM, "snap_store.py"), str(tmp_path / "store"), str(tmp_path / "ct"),
                          str(tmp_path / "snapped")], capture_output=True, text=True, check=True).stdout
    summary = json.loads(next(line[11:] for line in out.splitlines() if line.startswith("CREST_SNAP ")))
    assert summary["snapped_frac"] == 1.0 and summary["frac_c_pos_after"] == 1.0
    man = json.load(open(tmp_path / "snapped" / "manifest.json"))
    t_new = np.load(tmp_path / "snapped" / "shard_0" / "crossing_t.npy")
    vox = 10.0 + 2.0 * t_new
    assert np.abs(vox - np.tile(SHEETS, len(origins))).max() < 0.25
    ident = {k: v for k, v in man.items() if k != "fingerprint"}
    assert man["fingerprint"] == _digest(ident) and man["crest_snap"]["tool"] == "snap_store.py"
    villa = os.environ.get("VILLA_SPIRAL_FITTING", "/home/claude/vc/villa/spiral-fitting")
    if not os.path.isdir(villa):
        pytest.skip("no villa checkout: the loader check needs spiral-fitting/winding_supervision.py")
    pytest.importorskip("torch")
    sys.path.insert(0, villa)
    from winding_supervision import load_winding_inference_store
    store = load_winding_inference_store(str(tmp_path / "snapped"), "cpu", verify=True)
    assert store.fingerprint["num_crossings"] == len(t_new)


def test_runs_keep_the_crossings_that_agree_with_the_crest_count():
    from snap_store import consistent_runs
    tc = np.delete(SHEETS, 2)  # the CT shows no crest at 62, the model has a winding there (level 2)
    t_new = np.array([30.0, 46.0, 64.0, 78.0, 94.0, 110.0])
    moved = np.array([True, True, False, True, True, True])
    runs = consistent_runs(tc, t_new, moved, np.arange(6))
    # between 46 (level 1) and 78 (level 3) the model has 2 windings but the CT no crest: two runs, no pair across
    assert [r.tolist() for r in runs] == [[0, 1], [3, 4, 5]]
    moved[0] = False  # a run of one crossing gives no pair: left out
    assert [r.tolist() for r in consistent_runs(tc, t_new, moved, np.arange(6))] == [[3, 4, 5]]


def test_drop_unsnapped_keeps_levels_and_loads_in_villa(tmp_path):
    x = np.arange(160, dtype=float)
    prof = 60 + 150 * np.exp(-0.5 * ((x[:, None] - SHEETS[None]) / 1.5) ** 2).sum(1)
    vol = np.broadcast_to(np.clip(prof, 0, 255).astype(np.uint8), (32, 32, 160)).copy()
    _zarr(str(tmp_path / "ct"), vol)
    origins = [[8.0, 6.0, 10.0], [20.0, 18.0, 10.0], [20.0, 26.0, 10.0]]
    steps = [[0.0, 0.0, 1.0]] * 3
    t1 = SHEETS + 2.0
    t1[2] = SHEETS[2] + 9.0  # reaches the crest at 78, which the next crossing (at 80) is nearer to: not snapped
    ts = [t1 - 10.0, SHEETS - 3.0 - 10.0, np.zeros(0)]  # the third ray has no crossings at all
    _store(str(tmp_path / "store"), origins, steps, ts)
    out = subprocess.run([sys.executable, os.path.join(WM, "snap_store.py"), str(tmp_path / "store"), str(tmp_path / "ct"),
                          str(tmp_path / "snapped"), "--drop-unsnapped"], capture_output=True, text=True, check=True).stdout
    summary = json.loads(next(line[11:] for line in out.splitlines() if line.startswith("CREST_SNAP ")))
    assert summary["kept_frac"] == round(11 / 12, 4) and summary["count_inconsistent_frac"] == 0.0
    assert summary["rays_out"] == 2
    d = tmp_path / "snapped" / "shard_0"
    t, lev, off = (np.load(d / f) for f in ("crossing_t.npy", "crossing_level.npy", "crossing_offsets.npy"))
    assert off.tolist() == [0, 5, 11]  # the ray without crossings is gone
    assert lev[:5].tolist() == [0, 1, 3, 4, 5]  # the pair around the dropped crossing spans 2 windings
    assert np.abs(10.0 + t[:5] - np.delete(SHEETS, 2)).max() < 0.25 and np.abs(10.0 + t[5:] - SHEETS).max() < 0.25
    man = json.load(open(tmp_path / "snapped" / "manifest.json"))
    assert man["num_crossings"] == 11 and man["num_rays"] == 2
    assert man["shards"][0]["arrays"]["crossing_t"]["shape"] == [11]
    assert man["shards"][0]["arrays"]["ray_origin_zyx"]["shape"] == [2, 3]
    villa = os.environ.get("VILLA_SPIRAL_FITTING", "/home/claude/vc/villa/spiral-fitting")
    if not os.path.isdir(villa):
        pytest.skip("no villa checkout")
    pytest.importorskip("torch")
    sys.path.insert(0, villa)
    from winding_supervision import load_winding_inference_store
    store = load_winding_inference_store(str(tmp_path / "snapped"), "cpu", verify=True)
    pairs = store.sample_adjacent(200)
    assert set(pairs["target"].tolist()) == {1.0, 2.0}
