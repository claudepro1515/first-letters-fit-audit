"""Run villa's infer_winding_volume.py --native-phase-only on Kaggle T4 GPUs, with three adaptations.

1. `vc` is vc.py next to this file: a pure-Python/torch stand-in for volume-cartographer's C++ bindings
   (local zarr v2 volume, trilinear slab sampling on the worker's GPU), so nothing has to be compiled.
2. Autocast uses float16 (WM_AUTOCAST=fp16, default) or float32 (WM_AUTOCAST=fp32) instead of bfloat16,
   which T4s do not support in hardware.
3. The phase cache keeps only each slab's centre column, the only column export_spiral_supervision.py
   decodes. This checkpoint (scrollprize/winding_model_9um) predicts all 128 x 128 columns, 25 MB of float32
   per slab, which a full native cache cannot hold on Kaggle's disk. The stored frame origin is the centre
   ray's start, so the exporter's ray origins and directions are exactly those of the full cache.
   WM_COLUMN_GRID="n:offset" (default "1:0") keeps an n x n grid of columns per slab instead, spaced `offset`
   voxels apart around the centre (e.g. "3:40": columns 24, 64 and 104 on each transverse axis). The model
   predicts every column ("the central ray is only the sampling frame"), so this adds n^2 - 1 rays per slab
   at no extra GPU cost; each is stored as its own cache entry with its own frame origin.
Everything else (seeding on the fit's _spliced meshes, slab frames, the model, the exporter) is villa's code.

Usage: python run_infer.py <same arguments as infer_winding_volume.py, including --native-phase-only>
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import vesuvius.neural_tracing.winding_models.infer_winding_volume as iwv  # noqa: E402

AUTOCAST = os.environ.get("WM_AUTOCAST", "fp16")
GRID_N, GRID_OFFSET = (int(v) for v in os.environ.get("WM_COLUMN_GRID", "1:0").split(":"))
if GRID_N < 1 or GRID_N % 2 == 0:
    raise SystemExit("WM_COLUMN_GRID needs an odd n")


def centre_index(columns, column_stride):
    """Centre native column as export_spiral_supervision.py computes it (128 columns, stride 1 -> 64)."""
    return int(round((columns * column_stride - 1) / column_stride / 2))


def grid_columns(native_columns, column_stride):
    """Native column indices (a, b) of the kept columns, centre first when n = 1."""
    c = centre_index(native_columns, column_stride)
    step = GRID_OFFSET // column_stride
    half = GRID_N // 2
    cols = [(c + i * step, c + j * step) for i in range(-half, half + 1) for j in range(-half, half + 1)]
    if any(not 0 <= v < native_columns for ab in cols for v in ab):
        raise SystemExit(f"WM_COLUMN_GRID {GRID_N}:{GRID_OFFSET} leaves the {native_columns}-column slab")
    return cols


def forward_batch_centre(batch, model, device, column_upsample=1, *, args=None, column_stride=None,
                         out_shape=None, phase_offsets=None, phase_seed_windings=None):
    images = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
    valid = torch.from_numpy(np.stack([b[2] for b in batch])).to(device)
    if AUTOCAST == "fp32":
        out = model(images, valid.bool())
    else:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(images, valid.bool())
    phase = out["phase"].float()
    cols = grid_columns(phase.shape[1], int(column_stride or 1))
    kept = torch.stack([phase[:, a, b, :] for a, b in cols], dim=1)  # [B, K, L]
    return None, kept.cpu().numpy(), None


def init_centre_cache(args, rays, bounds, gpus, model_cfg):
    """iwv._initialize_native_phase_cache with the kept columns of each slab (one cache entry per column)."""
    import zarr
    from zarr.codecs import BloscCodec, BloscShuffle, PackBits

    group = zarr.open_group(args.output, mode="w")
    phase_group = group.create_group("phase")
    valid_group = group.create_group("valid")
    frame_group = group.create_group("frame")
    available_group = group.create_group("available")
    rays_group = group.create_group("rays")
    ray_length = int(model_cfg.get("ray_length", 384))
    transverse = int(model_cfg.get("transverse_size", 128))
    column_stride = int(model_cfg.get("column_stride", 4))
    native_columns = transverse // column_stride
    cols = grid_columns(native_columns, column_stride)
    per = len(cols)  # cache entries per slab
    for key in ("seed_xyz", "direction_xyz", "seed_winding"):
        value = np.repeat(np.asarray(rays[key]), per, axis=0)
        first_chunk = max(1, min(len(value), 1 << 16))
        rays_group.create_array(key, data=value, chunks=(first_chunk, *value.shape[1:]))
    compressor = BloscCodec(cname="zstd", clevel=1, shuffle=BloscShuffle.shuffle)
    shards = []
    for slot, gpu in enumerate(gpus):
        lo, hi = int(bounds[slot]) * per, int(bounds[slot + 1]) * per
        count = hi - lo
        name = f"shard_{gpu}"
        block = max(1, min(count, 1024))
        phase_group.create_array(name, shape=(count, 1, 1, ray_length), chunks=(block, 1, 1, ray_length),
                                 dtype="float32", compressors=[compressor])
        valid_group.create_array(name, shape=(count, 1, 1, ray_length), chunks=(block, 1, 1, ray_length),
                                 dtype="bool", filters=[PackBits()], compressors=[compressor])
        compact_chunk = max(1, min(count, 1 << 12))
        frame_group.create_array(name, shape=(count, 4, 3), chunks=(compact_chunk, 4, 3), dtype="float64",
                                 compressors=[compressor])
        available_group.create_array(name, shape=(count,), chunks=(compact_chunk,), dtype="bool",
                                     filters=[PackBits()], compressors=[compressor], fill_value=False)
        shards.append({"name": name, "gpu": int(gpu), "lo": lo, "hi": hi})

    c = centre_index(native_columns, column_stride)
    group.attrs.update({
        "artifact_type": "winding_native_phase_cache",
        "format_version": 1,
        "phase_dtype": "float32",
        "fit_checkpoint": str(os.path.realpath(args.fit_checkpoint)),
        "model_ckpt": str(os.path.realpath(args.model_ckpt)),
        "reference_zarr": str(os.path.realpath(args.reference_zarr)),
        "volume_scale": int(args.volume_scale),
        "ray_length": ray_length,
        "transverse_size": transverse,
        "column_stride": column_stride,
        "native_columns": 1,
        "model_native_columns": native_columns,
        "centre_column_only": per == 1,
        "centre_native_column": c,
        "centre_transverse_sample": c * column_stride,
        "column_grid": {"n": GRID_N, "offset_vox": GRID_OFFSET, "entries_per_slab": per,
                        "transverse_samples": [[a * column_stride, b * column_stride] for a, b in cols]},
        "frame_origin": "the kept column's ray start: slab origin + spacing * (a * axis_a + b * axis_b), (a, b) its transverse samples",
        "autocast": AUTOCAST,
        "slab_sampler": "vc.py shim (torch trilinear, align_corners, rounded to uint8)",
        "spacing": float(model_cfg.get("spacing", 1.0)),
        "sampling": str(model_cfg.get("sampling", "trilinear")),
        "crossing_sigma_wv": float(model_cfg.get("crossing_sigma_wv", 1.0)),
        "phase_shards": shards,
        "z_range": [int(value) for value in rays["z_range"]],
        "seed_windings": [int(value) for value in rays["seed_windings"]],
        "dr_per_winding": float(rays["dr_per_winding"]),
        "umbilicus": str(rays["umbilicus_path"]),
        "complete": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


class CentreCacheWriter:
    """Stores the kept columns' phase and validity, and each column's ray frame, one cache entry per column."""

    def __init__(self, group_path, gpu):
        import zarr

        group = zarr.open_group(str(group_path), mode="r+")
        name = f"shard_{gpu}"
        self.phase = group["phase"][name]
        self.valid = group["valid"][name]
        self.frame = group["frame"][name]
        self.available = group["available"][name]
        self.spacing = float(group.attrs["spacing"])
        self.cols = [tuple(ab) for ab in group.attrs["column_grid"]["transverse_samples"]]

    @staticmethod
    def _runs(indices):  # contiguous index runs, as iwv._NativePhaseCacheWriter._runs
        indices = np.asarray(indices, dtype=np.int64)
        starts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1]
        return zip(starts, np.r_[starts[1:], len(indices)])

    def add(self, indices, phase, valid, frames):
        per = len(self.cols)
        slabs = np.asarray(indices, dtype=np.int64)
        entries = (slabs[:, None] * per + np.arange(per)[None]).ravel()
        phase = np.asarray(phase, dtype=np.float32).reshape(len(slabs) * per, 1, 1, -1)
        valid = np.stack([np.asarray(v, dtype=bool)[a, b, :] for v in valid for a, b in self.cols])[:, None, None, :]
        frame_values = np.stack([
            np.stack([f.origin + self.spacing * (a * f.axis_a + b * f.axis_b), f.axis_a, f.axis_b, f.direction])
            for f in frames for a, b in self.cols]).astype(np.float64)
        for begin, end in self._runs(entries):
            destination = slice(int(entries[begin]), int(entries[end - 1]) + 1)
            source = slice(int(begin), int(end))
            self.phase[destination] = phase[source]
            self.valid[destination] = valid[source]
            self.frame[destination] = frame_values[source]
            self.available[destination] = True


_original_gpu_worker = iwv.gpu_worker


def gpu_worker_shim(gpu, args, shard_path, result_path, progress_queue=None):
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu))
        os.environ["VC_SHIM_DEVICE"] = f"cuda:{int(gpu)}"
    try:
        return _original_gpu_worker(gpu, args, shard_path, result_path, progress_queue)
    except BaseException:
        # keep the worker's traceback where the notebook can find it (the log tail is mostly progress bars)
        import traceback
        text = traceback.format_exc()
        print(f"[gpu{gpu}] WORKER ERROR\n{text}", flush=True)
        try:
            with open(os.path.join(args.output, f"worker_{gpu}_error.txt"), "w") as f:
                f.write(text)
        except OSError:
            pass
        raise


def finish_partial_cache(output):
    """After a worker failure, mark the cache complete with the slabs that were written (the exporter reads only
    the entries flagged available), so one bad slab costs its shard's remainder, not the whole run."""
    import zarr
    group = zarr.open_group(output, mode="r+")
    available = sum(int(np.count_nonzero(group["available"][item["name"]][:])) for item in group.attrs["phase_shards"])
    total = sum(int(item["hi"]) - int(item["lo"]) for item in group.attrs["phase_shards"])
    group.attrs.update({"complete": True, "partial_after_worker_error": True,
                        "num_available_entries": int(available), "num_entries": int(total)})
    print(f"[main] WARNING: a worker failed; kept {available:,}/{total:,} cache entries and marked the cache complete",
          flush=True)


iwv.gpu_worker = gpu_worker_shim
iwv._forward_batch = forward_batch_centre
iwv._initialize_native_phase_cache = init_centre_cache
iwv._NativePhaseCacheWriter = CentreCacheWriter

if __name__ == "__main__":
    if "--native-phase-only" not in sys.argv:
        sys.exit("run_infer.py only supports --native-phase-only (centre-column cache)")
    try:
        iwv.main()
    except RuntimeError as error:
        if "worker exit codes" not in str(error):
            raise
        finish_partial_cache(iwv.parse_args().output)  # the second positional argument
