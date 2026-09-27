"""Build a Kaggle notebook that runs villa's winding-model supervision loop on one band of one scroll.

  1. seed fit S: the First Letters recipe for a few thousand steps (only to place the seed rays);
  2. winding-model inference (scrollprize/winding_model_9um) on rays seeded on S's winding meshes, with
     villa's infer_winding_volume.py --native-phase-only (run_infer.py adaptations: vc shim, float16 on T4,
     centre-column cache), then villa's export_spiral_supervision.py -> compact crossing store;
     a second, smaller seed set on other windings -> held-out store (never used for fitting);
  3. refits W*: the recipe plus fit_spiral's winding_model spacing loss on the training store;
  4. audit: winding pitch, radius per winding, render contrast, CT cross-sections, and winding_agreement.py
     (fitted windings between consecutive model crossings) on the held-out store, for every run and for
     earlier runs attached as inputs.

Usage: python make_wm_notebook.py spec.json out.ipynb   (spec: see spec_0826_wm.json)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(*cands):
    """First existing path: the repo layout (../wm, ../ink, ../tools next to kaggle/) or the dev layout."""
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(cands)


spec = json.load(open(sys.argv[1]))
SCROLL, RUN = spec["scroll"], spec["run"]
TRACKS = f"{SCROLL}_{RUN}_surface_m7_L0_th0.2"
UMB = f"{SCROLL}/representations/umbilicus/{spec['umbilicus']}"
# an umbilicus given as a full URL (e.g. a community file on GitHub) is fetched from there instead of the S3 bucket
UMB_URL = spec["umbilicus"] if spec["umbilicus"].startswith("http") else "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + UMB
LAS = f"{SCROLL}/representations/predictions/lasagna/{spec['lasagna']}"
VOLUME = f"{SCROLL}/volumes/{spec['volume']}"
Z0, Z1 = spec["z"]
VILLA_COMMIT = spec.get("villa", "75c79ac5f506d4b9a89bcfbef8e8c0f2f0c3acb3")
STEPS = spec.get("steps", 30000)
SEED_STEPS = spec.get("seed_steps", 2000)
BASE = dict(spec["base"], z_begin=Z0, z_end=Z1)
SEED_RUN = dict(BASE, **spec["seed_overrides"], optimizer_num_training_steps=SEED_STEPS)
RUNS = {n: dict(BASE, **o, optimizer_num_training_steps=STEPS) for n, o in spec["runs"].items()}
assert 1 <= len(RUNS) <= 2
INFER = spec.get("infer", {})
WM_DIR = _find(os.path.join(HERE, "..", "wm"), "/home/claude/vc/wm")
TOOLS_DIR = spec.get("tools_dir") or _find(os.path.join(HERE, "..", "tools"), "/home/claude/vc/release/tools")

cells = []


def packed(var, files, what):
    """A code cell defining var = {path: source}, stored xz-compressed (Kaggle's importer silently drops notebooks above ~60 KB)."""
    import base64
    import hashlib
    import lzma
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode(), preset=9 | lzma.PRESET_EXTREME)).decode()
    lines = [f"# {what}, embedded as xz-compressed JSON {{path: source}} (sha256 of each file below)"]
    lines += [f"#   {p}  {hashlib.sha256(t.encode()).hexdigest()[:16]}" for p, t in files.items()]
    lines += ["import base64 as _b64, json as _json, lzma as _lzma",
              f"{var} = _json.loads(_lzma.decompress(_b64.b85decode("]
    lines += [f"    {blob[i:i + 120]!r}" for i in range(0, len(blob), 120)]
    lines += ["))).decode())" if False else ")).decode())"]
    return "\n".join(lines)


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def prune(t):
    """Drop the dead branch of every `if True:` / `if False:` the substitutions left (Kaggle's importer limit)."""
    lines = t.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^( *)if (True|False):\s*(#.*)?$", line)
        if not m:
            continue
        ind = len(m.group(1))
        body_end = i + 1
        while body_end < len(lines) and (not lines[body_end].strip() or len(lines[body_end]) - len(lines[body_end].lstrip()) > ind):
            body_end += 1
        else_end = body_end
        if else_end < len(lines) and re.match(r"^ {%d}else:\s*(#.*)?$" % ind, lines[else_end]):
            else_end += 1
            while else_end < len(lines) and (not lines[else_end].strip() or len(lines[else_end]) - len(lines[else_end].lstrip()) > ind):
                else_end += 1
            keep = lines[i + 1:body_end] if m.group(2) == "True" else lines[body_end + 1:else_end]
        else:
            keep = lines[i + 1:body_end] if m.group(2) == "True" else []
        keep = [x[4:] if x.startswith(" " * (ind + 4)) else x for x in keep]
        return prune("\n".join(lines[:i] + keep + lines[else_end:]))
    return t


def code(t, **subs):
    for k, v in subs.items():
        t = t.replace(f"__{k}__", str(v))
    left = re.findall(r'__[A-Z][A-Z0-9_]*__', t)
    assert not left, left
    if spec.get("prune_dead_code"):  # opt-in, so that the notebooks of earlier runs regenerate byte for byte
        t = prune(t)
        compile(t, "<cell>", "exec")  # the pruned cell must still be valid Python
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip()})


md(f"""# {spec['title']}

Settings: **Accelerator GPU T4 x2**, **Internet on**. Runs as a background commit (Save Version -> Save & Run All).

{spec['notes']}

Pipeline (all villa code at `{VILLA_COMMIT[:8]}` except the three adaptations in `run_infer.py`):
1. **Seeds**: {('the windings of the attached fit `' + spec['seed_from_input'] + '`' + (', moved onto the sheets (`tools/phase_align.py`)' if spec.get('align_seeds_period_vox') else '')) if spec.get('seed_from_input') else f'fit `S`, the First Letters recipe, {SEED_STEPS:,} steps, only to place seed rays on winding meshes'}.
2. **Winding-model inference**: `infer_winding_volume.py --native-phase-only` with the public
   [`scrollprize/winding_model_9um`](https://huggingface.co/scrollprize/winding_model_9um), seeded on {'the meshes of item 1' if spec.get('seed_from_input') else "`S`'s meshes"};
   then `export_spiral_supervision.py`. A second seed set on other windings gives a **held-out** crossing store.
3. **Refits**: the recipe plus `fit_spiral`'s `winding_model` spacing loss (its default production mode, which
   the recipe turns off because no crossing store existed for this scroll), {STEPS:,} steps.
4. **Audit**: {'none here (no GPU time): the CPU audit notebook audits these meshes and stores' if spec.get('audit') == 'none' else 'winding pitch, radius, render contrast, CT cross-sections, and `winding_agreement.py`: the number of fitted windings between consecutive model crossings on held-out rays (1 when windings follow the sheets)'}.""")
PHASE_RUNS = {n: o["__phase_weight"] for n, o in spec["runs"].items() if o.get("__phase_weight")}
SNAP = any(o.get("__store") == "snapped" for o in spec["runs"].values())
if PHASE_RUNS:
    md(f"""**Phase term** (runs {', '.join(f'`{n}` weight {w:g}' for n, w in PHASE_RUNS.items())}; `wm/phase_patch.py`):
villa's winding-model loss is relative (how many windings lie between two crossings of a ray), so it fixes the density
of the windings, not where they sit between the sheets. The patch adds mean(1 - cos(2 pi s / dr)) over the sampled
crossings (s = shifted radius): zero when every crossing lies on some winding. """ + (
        "`__store: snapped`: crossings moved onto the CT's crests (`wm/snap_store.py`)."
        if SNAP else "The other run is the same fit without it."))


code("""
import os, subprocess, sys, time, json, shutil, glob, re, threading
T0 = time.time()
def sh(cmd, check=True, cwd=None, env=None, tail=6000):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-tail:], flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')
    return p.stdout
def log(*a):
    print(f'[{(time.time() - T0) / 60:.1f} min]', *a, flush=True)
sh('nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv')
WORK = '/kaggle/temp'
os.makedirs(WORK, exist_ok=True)
sh('df -h /kaggle/working ' + WORK + ' / | cat; free -g; nproc')
""" + (f"print('NOTEBOOK', {spec['version_tag']!r}, flush=True)\n" if spec.get("version_tag") else ""))

code("""
# 1. uv + Python 3.14 + villa at a pinned commit (spiral-fitting and the vesuvius winding-model code)
sh('curl -LsSf https://astral.sh/uv/install.sh | sh')
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ['PATH']
sh('uv python install 3.14')
os.chdir(WORK)
if not os.path.isdir('villa'):
    sh('git clone --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git')
    sh('git sparse-checkout init --cone && git sparse-checkout set spiral-fitting lasagna vesuvius', cwd='villa')
    sh('git checkout __VILLA__', cwd='villa')
SF = WORK + '/villa/spiral-fitting'
PY = SF + '/.venv/bin/python'
WMSRC = WORK + '/villa/vesuvius/src/vesuvius/neural_tracing/winding_models'
t = time.time(); sh('uv sync', cwd=SF); print(f'uv sync {(time.time() - t) / 60:.1f} min')
sh(f'uv pip install --python {PY} tqdm pillow', check=False)
sh(PY + ' -c "import torch, zarr, numcodecs, scipy, PIL, tqdm; print(torch.__version__, zarr.__version__, numcodecs.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"')
""", VILLA=VILLA_COMMIT)

code("""
# 2. Data: umbilicus, tracks, Lasagna nx/ny/grad_mag for the band, the winding model, a dummy outer shell;
#    the level-0 CT slices the slabs need are downloaded in the background while the seed fit runs.
import urllib.request, urllib.parse, concurrent.futures as cf, xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
S3 = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
DL = 'https://dl.ash2txt.org/datasets/spiral_datasets/__SCROLL__/__RUN__/tracks'
ROOT = WORK + '/dataset___SCROLL__'
os.makedirs(ROOT + '/tracks', exist_ok=True); os.makedirs(ROOT + '/lasagna_inputs', exist_ok=True)

def fetch(url, dest, tries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                n = int(r.headers.get('Content-Length') or 0)
                tmp = dest + '.part'
                with open(tmp, 'wb') as f:
                    shutil.copyfileobj(r, f, 1 << 22)
            if n and os.path.getsize(tmp) != n:
                raise IOError(f'size mismatch {os.path.getsize(tmp)} != {n}')
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception as e:
            print('retry', k, url, e, flush=True); time.sleep(5)
    raise RuntimeError('download failed: ' + url)

def list_keys(prefix):
    keys, token = [], None
    while True:
        q = f'{S3}/?list-type=2&prefix={prefix}' + (f'&continuation-token={urllib.parse.quote(token)}' if token else '')
        x = ET.fromstring(urllib.request.urlopen(q, timeout=120).read())
        ns = {'s': x.tag.split('}')[0].strip('{')}
        keys += [c.find('s:Key', ns).text for c in x.findall('s:Contents', ns)]
        if x.find('s:IsTruncated', ns).text != 'true':
            return keys
        token = x.find('s:NextContinuationToken', ns).text

t = time.time()
fetch('__UMB_URL__', ROOT + '/umbilicus.json')
for name in ['__TRACKS__.dbm', '__TRACKS__.dbm.crossings.npz', '__TRACKS__.extract.json']:
    print(name, fetch(DL + '/' + name, ROOT + '/tracks/' + name) / 2**30, 'GB', flush=True)
VT = ROOT + '/tracks/__TRACKS__.dbm.vctracks'
for name in ['metadata.json', 'header.bin', 'offsets.i64', 'source_ids.u64', 'family_codes.i8', 'z_bounds.i32',
             'arclengths.f64', 'tortuosities.f64', 'coordinates.i32']:
    print(name, fetch(DL + '/__TRACKS__.dbm.vctracks/' + name, VT + '/' + name) / 2**30, 'GB', flush=True)
sig = json.load(open(VT + '/metadata.json'))['source_db_signature'][0]
dbm_path = ROOT + '/tracks/' + sig[0]
assert os.path.getsize(dbm_path) == sig[1], 'DBM size differs from the packed store signature'
os.utime(dbm_path, ns=(sig[2], sig[2]))
log('tracks ready')
zc0, zc1 = (__Z0__ // 4 - 32) // 32, (__Z1__ // 4 + 32) // 32
for ch in ['nx', 'ny', 'grad_mag']:
    src = '__LAS__/__SCROLL___' + ch + '.ome.zarr'
    dst = ROOT + '/lasagna_inputs/las_008_' + ch + '.ome.zarr'
    for meta in ['.zgroup', '.zattrs', '2/.zarray', '2/.zattrs']:
        try:
            fetch(S3 + '/' + src + '/' + meta, dst + '/' + meta, tries=1)
        except Exception as e:
            print('no', meta, e)
    keys = []
    for zc in range(zc0, zc1 + 1):
        keys += list_keys(src + f'/2/{zc}/')
    with cf.ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(lambda k: fetch(S3 + '/' + k, dst + '/' + k[len(src) + 1:]), keys))
    print(ch, len(keys), 'chunks', sum(sizes) / 2**20, 'MB', flush=True)
json.dump({"schema_version": 1, "name": "__SCROLL__", "voxel_size_um": 9.362, "spiral_outward_sense": "__SENSE__",
           "normal_zarr_group": "2", "lasagna_scale": 4, "paths": {"tracks_dbm": "tracks/__TRACKS__.dbm"}},
          open(ROOT + '/spiral-scroll.json', 'w'), indent=2)

# Dummy outer shell: fit_spiral's winding_model mode requires one (it keeps only pairs inside it, and tracks with an
# endpoint inside it). This one is the scan's own boundary: for every z and angle, the point where the ray from the
# umbilicus leaves the volume. Everything in the scan is inside it, so it removes nothing.
fetch(S3 + '/__VOLUME__/0/.zarray', WORK + '/volume_l0_zarray.json')
VSHAPE = json.load(open(WORK + '/volume_l0_zarray.json'))['shape']
cp = sorted(json.load(open(ROOT + '/umbilicus.json'))['control_points'], key=lambda q: q['z'])
uz = np.array([q['z'] for q in cp], float); uy_ = np.array([q['y'] for q in cp], float); ux_ = np.array([q['x'] for q in cp], float)
zs = np.arange(__Z0__ - 600, __Z1__ + 601, 2.0); th = np.radians(np.arange(0, 360, 0.5) + 0.25)
Zg, Tg = np.meshgrid(zs, th, indexing='ij')
cx_, cy_ = np.interp(Zg, uz, ux_), np.interp(Zg, uz, uy_)
cs_, sn_ = np.cos(Tg), np.sin(Tg)
with np.errstate(divide='ignore'):
    tx = np.where(cs_ > 0, (VSHAPE[2] - 1 - cx_) / cs_, np.where(cs_ < 0, -cx_ / cs_, np.inf))
    ty = np.where(sn_ > 0, (VSHAPE[1] - 1 - cy_) / sn_, np.where(sn_ < 0, -cy_ / sn_, np.inf))
rr = np.minimum(tx, ty) - 1.0
shell = ROOT + '/outer_shell'
os.makedirs(shell, exist_ok=True)
for c, a in zip('xyz', (cx_ + rr * cs_, cy_ + rr * sn_, Zg)):
    Image.fromarray(a.astype(np.float32)).save(f'{shell}/{c}.tif')
json.dump({'scale': [0.5, 0.5], 'format': 'tifxyz', 'type': 'seg', 'uuid': 'scan_boundary_shell'}, open(shell + '/meta.json', 'w'))
print('outer shell = scan boundary, radius', round(float(rr.min())), '-', round(float(rr.max())), 'voxels', flush=True)

WM = WORK + '/wm'
os.makedirs(WM, exist_ok=True)
CKPT_WM = WM + '/winding_model_9um_ckpt_final.pth'
fetch('https://huggingface.co/scrollprize/winding_model_9um/resolve/main/ckpt_final.pth', CKPT_WM)
fetch('https://huggingface.co/scrollprize/winding_model_9um/resolve/main/config.json', WM + '/winding_model_9um_config.json')
log('dataset ready'); sh(f'du -sh {ROOT}/* {WM}/* | cat')

# Level-0 CT for the band plus a margin (slabs reach 64 voxels across and up to 192 along the ray)
VOL = WORK + '/volume_l0'
free_gb = shutil.disk_usage(WORK).free / 2**30
MARGIN = 256 if free_gb > 80 else 160
VZ0, VZ1 = max(0, (__Z0__ - MARGIN) // 128 * 128), (__Z1__ + MARGIN + 127) // 128 * 128
VOLRES = {}
def download_volume():
    src = '__VOLUME__/0'
    os.makedirs(VOL + '/0', exist_ok=True)
    fetch(S3 + '/' + src + '/.zarray', VOL + '/0/.zarray')
    fetch(S3 + '/__VOLUME__/.zattrs', VOL + '/.zattrs')
    open(VOL + '/.zgroup', 'w').write('{"zarr_format": 2}')
    keys = []
    for cz in range(VZ0 // 128, VZ1 // 128):
        keys += list_keys(src + f'/{cz}/')
    t = time.time()
    with cf.ThreadPoolExecutor(48) as ex:
        sizes = list(ex.map(lambda k: fetch(S3 + '/' + k, VOL + '/0/' + k[len(src) + 1:]), keys))
    VOLRES.update(chunks=len(keys), gb=round(sum(sizes) / 2**30, 2), minutes=round((time.time() - t) / 60, 1))
vol_thread = threading.Thread(target=download_volume, daemon=True)
vol_thread.start()
log(f'volume download started: z [{VZ0}, {VZ1}), {free_gb:.0f} GB free')
""", SCROLL=SCROLL, RUN=RUN, UMB=UMB, UMB_URL=UMB_URL, TRACKS=TRACKS, Z0=Z0, Z1=Z1, LAS=LAS, SENSE=spec["sense"], VOLUME=VOLUME)

code("""
# 3. Run configurations (every key exists in villa __VILLA8__; unknown keys abort the fit) and helpers
SEED = __SEED_RUN__
RUNS = __RUNS__
OUT = WORK + '/spiral_out'

def dataset_for(sense, store=None):
    # the dataset a run reads: ROOT itself, or a sibling of symlinks with its own spiral-scroll.json (the other sense)
    # and/or its own winding_inference (store 'snapped': the training store snapped onto the CT's crests, section 5)
    other = bool(sense) and sense != '__SENSE__'
    if not other and not store:
        return ROOT, WORK + '/spiral_cache'
    ds = ROOT + ('_' + sense if other else '') + ('_' + store if store else '')
    if not os.path.isdir(ds):
        os.makedirs(ds)
        for f in os.listdir(ROOT):
            if not ((other and f == 'spiral-scroll.json') or (store and f == 'winding_inference')):
                os.symlink(f'{ROOT}/{f}', f'{ds}/{f}')
        if other:
            j = json.load(open(ROOT + '/spiral-scroll.json'))
            j['spiral_outward_sense'] = sense
            json.dump(j, open(ds + '/spiral-scroll.json', 'w'), indent=2)
        if store:
            shutil.copytree(f'{TRAIN}_{store}', ds + '/winding_inference')
    return ds, WORK + '/spiral_cache' + ('_' + sense if other else '')

def launch(name, cfg, gpu):
    cfg = dict(cfg)
    ds, cache = dataset_for(cfg.pop('__sense', None), cfg.pop('__store', None))  # '__sense', '__store': ours, not fitter keys
    phase_w = float(cfg.pop('__phase_weight', 0) or 0)  # ours too: the phase term of wm/phase_patch.py
    env = dict(os.environ, FIT_SPIRAL_CONFIG_OVERRIDES=json.dumps(cfg), FIT_SPIRAL_RUN_DIR=f'{OUT}/{name}',
               FIT_SPIRAL_CACHE_DIR=cache, WANDB_MODE='disabled', CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONUNBUFFERED='1', WM_PHASE_WEIGHT=str(phase_w))
    os.makedirs(f'{OUT}/{name}', exist_ok=True)
    json.dump(dict(cfg, dataset=ds, wm_phase_weight=phase_w), open(f'{OUT}/{name}/overrides.json', 'w'), indent=1)
    logf = open(f'{OUT}/{name}.log', 'w')
    return subprocess.Popen([PY, '-u', 'fit_spiral.py', '--dataset', ds], cwd=SF, env=env, stdout=logf, stderr=subprocess.STDOUT)

def mem_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return round(int(line.split()[1]) / 2**20, 1)

def wait(procs, every=180, deadline=None):
    while any(p.poll() is None for p in procs.values()):
        time.sleep(every)
        for n in procs:
            tail = open(f'{OUT}/{n}.log').read()[-300:].replace('\\r', ' | ').replace('\\n', ' | ')
            print(f'[{(time.time() - T0) / 60:.0f} min, {mem_gb()} GB free] {n}: {tail}', flush=True)
        if deadline and time.time() > deadline:
            print('DEADLINE reached: stopping the fits', flush=True)
            for p in procs.values():
                if p.poll() is None:
                    p.terminate()
            time.sleep(60)
            for p in procs.values():
                if p.poll() is None:
                    p.kill()
    return {n: p.returncode for n, p in procs.items()}

def past_first_step(name):
    return bool(re.search(r'Optimizing — [1-9][\\d,]*/', open(f'{OUT}/{name}.log').read()))

def run_logged(cmd, name, env=None, every=180):
    # long command: output to WORK/<name>.log, a tail printed every few minutes, the end printed at exit
    print('$', cmd, flush=True)
    path = f'{WORK}/{name}.log'
    with open(path, 'w') as logf:
        p = subprocess.Popen(cmd, shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT)
        while p.poll() is None:
            time.sleep(every if p.poll() is None else 0)
            if p.poll() is None:
                log(name, open(path).read()[-300:].replace('\\r', ' | ').replace('\\n', ' | '))
    txt = open(path).read()
    for k in [m.start() for m in re.finditer(r'WORKER ERROR|Traceback \\(most recent call last\\)', txt)][:3]:
        print('ERROR BLOCK in ' + name + ':\\n' + txt[k:k + 2500].replace('\\r', '\\n'), flush=True)
    print(txt[-3000:].replace('\\r', '\\n'), flush=True)
    shutil.copy2(path, '/kaggle/working/')
    if p.returncode != 0:
        raise RuntimeError(f'{name} failed ({p.returncode})')
    return txt
""", VILLA8=VILLA_COMMIT[:8], SEED_RUN=repr(SEED_RUN), RUNS=repr(RUNS), SENSE=spec["sense"])

wm_files = {
    "vc.py": open(os.path.join(WM_DIR, "vc.py")).read(),
    "run_infer.py": open(os.path.join(WM_DIR, "run_infer.py")).read(),
    "count_seeds.py": open(os.path.join(WM_DIR, "count_seeds.py")).read(),
    "stub/vesuvius/__init__.py": "",
    "stub/vesuvius/tifxyz.py": open(os.path.join(WM_DIR, "stub/vesuvius/tifxyz.py")).read(),
}
BENCH_ON = spec.get("bench", True)
if BENCH_ON:
    wm_files["bench_wm.py"] = open(os.path.join(WM_DIR, "bench_wm.py")).read()
if any(o.get("__phase_weight") for o in spec["runs"].values()):
    wm_files["phase_patch.py"] = open(os.path.join(WM_DIR, "phase_patch.py")).read()
if INFER.get("phase_anchor"):
    wm_files["phase_anchor_store.py"] = open(os.path.join(WM_DIR, "phase_anchor_store.py")).read()
if SNAP:
    wm_files["snap_store.py"] = open(os.path.join(WM_DIR, "snap_store.py")).read()
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": packed("WM_FILES", wm_files, "Winding-model runner files (see run_infer.py's docstring)")})
TOOL_FILES = ["vcz.py", "mesh_roughness.py", "winding_pitch.py", "render_contrast.py", "winding_agreement.py"]
if spec.get("audit", "full") in ("light", "none"):  # agreement only (or nothing) here; the rest in the CPU audit notebook
    TOOL_FILES = ["vcz.py", "winding_agreement.py"] if spec.get("audit") == "light" else []  # phase_align reads the local CT
if spec.get("align_seeds_period_vox"):
    TOOL_FILES += ["phase_align.py"]
tools_src = {f: open(os.path.join(TOOLS_DIR, f)).read() for f in TOOL_FILES}
TOOLS_CELL = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": packed("TOOLS_SRC", tools_src, "Audit tools (first-letters-scan-atlas)")}
if spec.get("align_seeds_period_vox"):
    cells.append(TOOLS_CELL)

STORE_FROM = spec.get("store_from_input")  # a notebook whose crossing stores to reuse (no seed fit, no inference)
SNAP_ARGS = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in INFER.get("snap", {}).items() if k != "drop_unsnapped") + (
    " --drop-unsnapped" if INFER.get("snap", {}).get("drop_unsnapped") else "")
if not STORE_FROM:
    code("""
# 4. Seed fit S on GPU 0 (its first step also builds the shared fit cache). Meanwhile, once the CT is on disk,
#    benchmark the winding model on GPU 1: seconds per slab and float16 vs float32 on radial slabs.
for rel, src in WM_FILES.items():
    os.makedirs(os.path.dirname(os.path.join(WM, rel)), exist_ok=True)
    open(os.path.join(WM, rel), 'w').write(src)
VSRC = WORK + '/villa/vesuvius/src/vesuvius'
for name in ('neural_tracing', 'tifxyz_canvas.py'):   # the rest of the vesuvius package is not needed
    if not os.path.lexists(WM + '/stub/vesuvius/' + name):
        os.symlink(VSRC + '/' + name, WM + '/stub/vesuvius/' + name)
WM_ENV = dict(os.environ, PYTHONPATH=WM + ':' + WM + '/stub', VC_SHIM_ZRANGE=f'{VZ0},{VZ1}', PYTHONUNBUFFERED='1')
if any(r.get('__phase_weight') for r in RUNS.values()):
    # the phase term (wm/phase_patch.py): off unless a run sets WM_PHASE_WEIGHT, so the seed fit is villa's own
    sh(f'{sys.executable} {WM}/phase_patch.py {SF}')
    sh(f'git -C {SF} diff --stat | cat', check=False)
SEED_FROM = __SEED_FROM__  # a fit attached as input seeds the inference instead of a fresh recipe fit (second-generation store)
seed_proc = None if SEED_FROM else launch('S_seed', SEED, 0)
vol_thread.join()
log('volume', VOLRES); sh(f'du -sh {VOL} | cat')
if __BENCH_ON__:
    out = sh(f'{PY} {WM}/bench_wm.py {VOL} {CKPT_WM} {ROOT}/umbilicus.json {(__Z0__ + __Z1__) // 2} --device cuda:0',
             env=dict(WM_ENV, CUDA_VISIBLE_DEVICES='1'), check=False)
    BENCH = next((json.loads(l[6:]) for l in out.splitlines() if l.startswith('BENCH ')), {})
    d = BENCH.get('fp16_vs_fp32_registered_phase_abs_diff', {})
    AUTOCAST = 'fp16' if ('fp16_s_per_slab' in BENCH and d.get('p99', 9) < 0.1 and BENCH.get('fp16_same_crossing_count_frac', 0) >= 0.9) else 'fp32'
    T_SLAB = BENCH.get(AUTOCAST + '_s_per_slab', 0.5)
else:  # measured on Kaggle's T4s in earlier runs (float16 = float32 crossings on 24 benchmark slabs)
    BENCH, AUTOCAST, T_SLAB = {'skipped': True}, 'fp16', 0.85
log('winding model on T4:', AUTOCAST, T_SLAB, 's/slab')
if SEED_FROM:
    roots = sorted(glob.glob(f'/kaggle/input/**/{SEED_FROM}/meshes', recursive=True))
    assert roots, 'seed fit not attached: ' + SEED_FROM
    S_MESHES = roots[0]
    # the inference reads only cfg (initial spacing, winding range) and the z range from the fit checkpoint
    S_CKPT = WORK + '/seed_from_input.ckpt'
    seed_cfg = {'cfg': {'initial_dr_per_winding': 16.0, 'output_first_winding': 10, 'shell_outer_winding_idx': 130,
                        'gap_expander_num_windings': 130}, 'z_begin': __Z0__, 'z_end': __Z1__}
    subprocess.run([PY, '-c', 'import sys, json, torch; torch.save(json.loads(sys.argv[1]), sys.argv[2])',
                    json.dumps(seed_cfg), S_CKPT], check=True)
    codes = {'S_seed': 0}
    if __ALIGN_PERIOD__:
        # villa registers each ray's crossings at its seed and seeds on verified sheets; our seeds are a fit's windings,
        # parallel to the sheets at an unknown offset: move the seed windings onto the sheets first (tools/phase_align.py)
        TOOLS = WORK + '/atlas/tools'
        os.makedirs(TOOLS, exist_ok=True)
        for fname, src in TOOLS_SRC.items():
            open(os.path.join(TOOLS, fname), 'w').write(src)
        # all windings: villa orients rays toward the mesh of w+1 or w-1
        need = [w for w in range(__FIRST__ - 1, __LAST__ + 2) if os.path.isdir(f'{S_MESHES}/w{w:03d}_spliced')]
        ALIGNED = WORK + '/seed_aligned'
        groups = [need[k::4] for k in range(4)]
        procs_a = []
        for k, grp in enumerate(groups):
            sub = f'{WORK}/seed_subsets/{k}'
            os.makedirs(sub, exist_ok=True)
            for w in grp:
                if not os.path.lexists(f'{sub}/w{w:03d}_spliced'):
                    os.symlink(f'{S_MESHES}/w{w:03d}_spliced', f'{sub}/w{w:03d}_spliced')
            procs_a.append(subprocess.Popen([PY, f'{TOOLS}/phase_align.py', sub, VOL, ALIGNED, '--period', '__ALIGN_PERIOD__',
                                             '--pattern', 'w[0-9][0-9][0-9]_spliced'], text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT))
        ALIGN = []
        for pa in procs_a:
            o, _ = pa.communicate()
            ALIGN += [json.loads(l) for l in o.splitlines() if l.startswith('{')]
            if pa.returncode:
                print(o[-3000:], flush=True)
        print('ALIGN_SEEDS ' + json.dumps(ALIGN), flush=True)
        kept = [w for w in need if not os.path.exists(f'{ALIGNED}/w{w:03d}_spliced/x.tif')]
        for w in kept:  # not aligned: as fitted
            shutil.rmtree(f'{ALIGNED}/w{w:03d}_spliced', ignore_errors=True)
            os.symlink(f'{S_MESHES}/w{w:03d}_spliced', f'{ALIGNED}/w{w:03d}_spliced')
        log('seed windings aligned:', len(ALIGN), 'of', len(need), '; as fitted:', kept)
        S_MESHES = ALIGNED
else:
    codes = wait({'S_seed': seed_proc}, every=120, deadline=T0 + 3 * 3600)
    print(open(f'{OUT}/S_seed.log').read()[-2500:].replace('\\r', '\\n'))
    S_CKPT = f'{OUT}/S_seed/checkpoint_fitted.ckpt'
    S_MESHES = sorted(glob.glob(f'{OUT}/S_seed/meshes/*/'), key=os.path.getmtime)
    assert codes['S_seed'] == 0 and os.path.exists(S_CKPT) and S_MESHES, f'seed fit failed: {codes}'
    S_MESHES = S_MESHES[-1].rstrip('/')
log('seed fit done:', S_MESHES, len(glob.glob(S_MESHES + '/w[0-9][0-9][0-9]_spliced')), 'spliced windings')
""", Z0=Z0, Z1=Z1, SEED_FROM=repr(spec.get("seed_from_input")), BENCH_ON=bool(BENCH_ON),
     ALIGN_PERIOD=spec.get("align_seeds_period_vox") or 0, FIRST=INFER.get("first", 10), LAST=INFER.get("last", 130),
     TRAIN_STEP=INFER.get("winding_step", 4), HELD_FIRST=INFER.get("held_first", 12), HELD_STEP=INFER.get("held_step", 8))

    code("""
# 5. Winding-model inference -> compact crossing stores (training seeds, and held-out seeds on other windings).
#    The seed spacing is the finest of the candidates whose inference fits the time budget at the measured speed.
TRAIN_STEP, BUDGET_S, SPACINGS = __TRAIN_STEP__, __BUDGET_MIN__ * 60, __SPACINGS__
out = sh(f'{PY} {WM}/count_seeds.py {S_CKPT} {S_MESHES} {ROOT}/umbilicus.json __FIRST__ __LAST__ {TRAIN_STEP} ' + ' '.join(map(str, SPACINGS)), env=WM_ENV, tail=1500)
counts = {int(float(j['spacing'])): j['slabs'] for j in (json.loads(l) for l in out.splitlines() if l.startswith('{'))}
SPACING = next((sp for sp in SPACINGS if counts[sp] * T_SLAB / 2 * 1.3 <= BUDGET_S), SPACINGS[-1])
log('training seeds: winding step', TRAIN_STEP, 'spacing', SPACING, 'slabs', counts[SPACING], 'all counts', counts)
BATCH = 4 if AUTOCAST == 'fp16' else 2
def infer(tag, first, last, step, spacing):
    cache = f'{WM}/cache_{tag}.zarr'
    t = time.time()
    run_logged(f'{PY} {WM}/run_infer.py {S_CKPT} {cache} --reference-zarr {VOL} --volume-scale 0 --model-ckpt {CKPT_WM} '
               f'--umbilicus {ROOT}/umbilicus.json --meshes-dir {S_MESHES} --winding-range {first} {last} --winding-step {step} '
               f'--seed-spacing {spacing} --batch-size {BATCH} --extract-threads 6 --gpus 0,1 --native-phase-only',
               'infer_' + tag, env=dict(WM_ENV, WM_AUTOCAST=AUTOCAST, WM_COLUMN_GRID='__GRID__'))
    store = f'{WM}/winding_inference_{tag}'
    shutil.rmtree(store, ignore_errors=True)
    sh(f'{PY} {WMSRC}/export_spiral_supervision.py {cache} {store} --workers 4 --edge-trim 2 --no-progress', tail=2000)
    log(tag, 'inference + export', round((time.time() - t) / 60, 1), 'min')
    return store
TRAIN = infer('train', __FIRST__, __LAST__, TRAIN_STEP, SPACING)
HELD = infer('heldout', __HELD_FIRST__, __LAST__, __HELD_STEP__, SPACING * 2)
RAW = {'train': TRAIN, 'heldout': HELD}
if __PHASE_ANCHOR__:
    # the crossings sit at the seed's phase (villa registers each ray at its seed; the model's phase has one free offset per
    # slab): move each ray's crossings onto the brightest CT (wm/phase_anchor_store.py), both stores the same way
    ANCHOR = {}
    for tag in ('train', 'heldout'):
        out = sh(f'{PY} {WM}/phase_anchor_store.py {RAW[tag]} {VOL} {RAW[tag]}_anchored', env=WM_ENV, tail=3000)
        ANCHOR[tag] = next((json.loads(l[13:]) for l in out.splitlines() if l.startswith('PHASE_ANCHOR ')), None)
    print('ANCHOR ' + json.dumps(ANCHOR), flush=True)
    TRAIN, HELD = TRAIN + '_anchored', HELD + '_anchored'

def store_stats(root):
    man = json.load(open(root + '/manifest.json'))
    gaps, per = [], []
    for s in man['shards']:
        a = {k: np.load(f"{root}/{s['name']}/{v['file']}") for k, v in s['arrays'].items()}
        L = np.linalg.norm(a['ray_step_zyx'], axis=1)
        o = a['crossing_offsets']
        per += np.diff(o).tolist()
        for i in range(len(o) - 1):
            gaps += (np.diff(a['crossing_t'][o[i]:o[i + 1]]) * L[i]).tolist()
    g = np.array(gaps)
    return {'rays': man['num_rays'], 'crossings': man['num_crossings'],
            'crossings_per_ray_median': float(np.median(per)) if per else None,
            'adjacent_gap_vox': {q: round(float(np.percentile(g, p)), 1) for q, p in (('p10', 10), ('p25', 25), ('median', 50), ('p75', 75), ('p90', 90))} if len(g) else None,
            'gap_under_8_vox_frac': round(float((g < 8).mean()), 3) if len(g) else None, 'fingerprint': man['fingerprint']}
STORES = {'train': store_stats(TRAIN), 'heldout': store_stats(HELD)}
print('STORES ' + json.dumps(STORES), flush=True)
shutil.copytree(TRAIN, ROOT + '/winding_inference', dirs_exist_ok=False)
for tag, src in (('train', TRAIN), ('heldout', HELD)):
    shutil.copytree(src, f'/kaggle/working/winding_inference_{tag}', dirs_exist_ok=True)
    if src != RAW[tag]:  # the stores as villa's exporter wrote them, before the phase anchoring
        shutil.copytree(RAW[tag], f'/kaggle/working/unanchored/winding_inference_{tag}', dirs_exist_ok=True)
if __SNAP__:
    # runs with '__store': 'snapped' read the training store with each crossing moved onto the nearest crest of the CT,
    # keeping only crossings that agree with the CT's crest count (wm/snap_store.py), from a sibling dataset (launch)
    out = sh(f'{PY} {WM}/snap_store.py {TRAIN} {VOL} {TRAIN}_snapped __SNAP_ARGS__', env=WM_ENV, tail=3000)
    SNAPPED = next((json.loads(l[11:]) for l in out.splitlines() if l.startswith('CREST_SNAP ')), None)
    print('CREST_SNAP ' + json.dumps(SNAPPED), flush=True)
    shutil.copytree(TRAIN + '_snapped', '/kaggle/working/winding_inference_train_snapped', dirs_exist_ok=True)
    STORES['train_snapped'] = dict(store_stats(TRAIN + '_snapped'), crest_snap=SNAPPED)
""", SNAP=SNAP, SNAP_ARGS=SNAP_ARGS, TRAIN_STEP=INFER.get("winding_step", 4), BUDGET_MIN=INFER.get("budget_min", 75),
     SPACINGS=INFER.get("spacings", [64, 80, 96, 128]), FIRST=INFER.get("first", 10), LAST=INFER.get("last", 130),
     HELD_FIRST=INFER.get("held_first", 12), HELD_STEP=INFER.get("held_step", 8), GRID=INFER.get("column_grid", "1:0"),
     PHASE_ANCHOR=bool(INFER.get("phase_anchor", False)))
else:
    code("""
# 4-5. The crossing stores of an earlier run, attached as input (__STORE_FROM__): no seed fit and no inference here
for rel, src in WM_FILES.items():
    os.makedirs(os.path.dirname(os.path.join(WM, rel)), exist_ok=True)
    open(os.path.join(WM, rel), 'w').write(src)
VSRC = WORK + '/villa/vesuvius/src/vesuvius'
for name in ('neural_tracing', 'tifxyz_canvas.py'):   # the rest of the vesuvius package is not needed
    if not os.path.lexists(WM + '/stub/vesuvius/' + name):
        os.symlink(VSRC + '/' + name, WM + '/stub/vesuvius/' + name)
WM_ENV = dict(os.environ, PYTHONPATH=WM + ':' + WM + '/stub', VC_SHIM_ZRANGE=f'{VZ0},{VZ1}', PYTHONUNBUFFERED='1')
if any(r.get('__phase_weight') for r in RUNS.values()):
    # the phase term (wm/phase_patch.py): off unless a run sets WM_PHASE_WEIGHT
    sh(f'{sys.executable} {WM}/phase_patch.py {SF}')
    sh(f'git -C {SF} diff --stat | cat', check=False)
found = {os.path.basename(p): p for p in sorted(glob.glob('/kaggle/input/**/__STORE_FROM__/winding_inference_*', recursive=True))}
assert 'winding_inference_train' in found, 'store run not attached: __STORE_FROM__'
TRAIN, HELD = WM + '/winding_inference_train', WM + '/winding_inference_heldout'
shutil.copytree(found['winding_inference_train'], TRAIN, dirs_exist_ok=True)
if 'winding_inference_heldout' in found:
    shutil.copytree(found['winding_inference_heldout'], HELD, dirs_exist_ok=True)
BENCH, AUTOCAST, SPACING, codes = {'skipped': True, 'stores_from': '__STORE_FROM__'}, None, None, {}
vol_thread.join()
log('volume', VOLRES); log('stores from', found)
def store_stats(root):
    man = json.load(open(root + '/manifest.json'))
    gaps, per = [], []
    for s in man['shards']:
        a = {k: np.load(f"{root}/{s['name']}/{v['file']}") for k, v in s['arrays'].items()}
        L = np.linalg.norm(a['ray_step_zyx'], axis=1)
        o = a['crossing_offsets']
        per += np.diff(o).tolist()
        for i in range(len(o) - 1):
            gaps += (np.diff(a['crossing_t'][o[i]:o[i + 1]]) * L[i]).tolist()
    g = np.array(gaps)
    return {'rays': man['num_rays'], 'crossings': man['num_crossings'],
            'crossings_per_ray_median': float(np.median(per)) if per else None,
            'adjacent_gap_vox': {q: round(float(np.percentile(g, p)), 1) for q, p in (('p10', 10), ('p25', 25), ('median', 50), ('p75', 75), ('p90', 90))} if len(g) else None,
            'gap_under_8_vox_frac': round(float((g < 8).mean()), 3) if len(g) else None, 'fingerprint': man['fingerprint']}
STORES = {'train': store_stats(TRAIN), 'heldout': store_stats(HELD) if os.path.isdir(HELD) else None}
print('STORES ' + json.dumps(STORES), flush=True)
shutil.copytree(TRAIN, ROOT + '/winding_inference', dirs_exist_ok=False)
if __SNAP__:
    # runs with '__store': 'snapped' read the training store with each crossing moved onto the nearest crest of the CT,
    # keeping only crossings that agree with the CT's crest count (wm/snap_store.py), from a sibling dataset (launch)
    out = sh(f'{PY} {WM}/snap_store.py {TRAIN} {VOL} {TRAIN}_snapped __SNAP_ARGS__', env=WM_ENV, tail=3000)
    SNAPPED = next((json.loads(l[11:]) for l in out.splitlines() if l.startswith('CREST_SNAP ')), None)
    print('CREST_SNAP ' + json.dumps(SNAPPED), flush=True)
    shutil.copytree(TRAIN + '_snapped', '/kaggle/working/winding_inference_train_snapped', dirs_exist_ok=True)
    STORES['train_snapped'] = dict(store_stats(TRAIN + '_snapped'), crest_snap=SNAPPED)
""", STORE_FROM=STORE_FROM, SNAP=SNAP, SNAP_ARGS=SNAP_ARGS)

code("""
# 6. Refits with the winding-model spacing loss, one per GPU (the second starts after the first's first step)
LIMIT_S, AUDIT_S = 11.75 * 3600, 1.0 * 3600
names = list(RUNS)
procs = {names[0]: launch(names[0], RUNS[names[0]], len(names) - 1)}
t = time.time()
while procs[names[0]].poll() is None and time.time() - t < 3 * 3600 and not past_first_step(names[0]):
    time.sleep(30)
log(names[0], 'past its first step; exit code so far:', procs[names[0]].poll())
if procs[names[0]].poll() not in (None, 0):
    # keep going: the crossing stores are already in /kaggle/working, and the audit still runs on S_seed
    print('REFIT FAILED EARLY', names[0], open(f'{OUT}/{names[0]}.log').read()[-6000:].replace('\\r', '\\n'), flush=True)
    codes = {names[0]: procs[names[0]].returncode}
else:
    for gpu, n in enumerate(names[1:]):
        procs[n] = launch(n, RUNS[n], gpu)
    codes = wait(procs, deadline=T0 + LIMIT_S - AUDIT_S)
codes['S_seed'] = 0
print('runs', codes)
for n in names:
    if os.path.exists(f'{OUT}/{n}.log'):
        print('=====', n); print(open(f'{OUT}/{n}.log').read()[-3000:].replace('\\r', '\\n'))
""")

if not spec.get("align_seeds_period_vox"):
    cells.append(TOOLS_CELL)

AUDIT_MODE = spec.get("audit", "full")
if AUDIT_MODE in ("light", "none"):  # agreement only, or none: the rest runs in the CPU audit notebook
    code("""
# 7. Outputs; with audit 'light' also the agreement with the winding model (winding_agreement.py) on both stores for
#    every run and earlier runs attached as inputs. Everything else runs in the CPU audit notebook (no GPU time).
TOOLS = WORK + '/atlas/tools'
os.makedirs(TOOLS, exist_ok=True)
for fname, src in TOOLS_SRC.items():
    open(os.path.join(TOOLS, fname), 'w').write(src)
sh('pip install -q tifffile numcodecs')

def audit(name, g):
    ws = glob.glob(g + '/w[0-9][0-9][0-9]/x.tif')
    res = {'mesh_dir': g, 'windings': len(ws)}
    for tag, store in (('heldout', HELD), ('train', TRAIN)):
        if ws and '__AMODE__' == 'light':
            out = sh(f'python {TOOLS}/winding_agreement.py {g} {store} --z __Z0__ __Z1__ --max-rays 4000 --umbilicus {ROOT}/umbilicus.json', check=False, tail=1500)
            res['agreement_' + tag] = next((json.loads(l) for l in out.splitlines() if l.startswith('{')), None)
    return res

summary = {'scroll': '__SCROLL__', 'z': [__Z0__, __Z1__], 'villa': '__VILLA8__', 'bench': BENCH, 'autocast': AUTOCAST,
           'seed_spacing': SPACING, 'stores': STORES, 'anchor': globals().get('ANCHOR'), 'runs': {}, 'earlier_runs': {}}
for n in ['S_seed'] + names:
    groups = sorted(set(os.path.dirname(os.path.dirname(p)) for p in glob.glob(f'{OUT}/{n}/meshes/*/w[0-9][0-9][0-9]/x.tif')))
    res = {'exit': codes.get(n), 'overrides': SEED if n == 'S_seed' else RUNS[n]}
    if os.path.exists(f'{OUT}/{n}.log'):
        if groups:
            res.update(audit(n, groups[0]))
            shutil.copytree(groups[0], f'/kaggle/working/{n}/meshes', dirs_exist_ok=True)
        for f in glob.glob(f'{OUT}/{n}/*.json'):
            os.makedirs(f'/kaggle/working/{n}', exist_ok=True)
            shutil.copy2(f, f'/kaggle/working/{n}/')
        shutil.copy2(f'{OUT}/{n}.log', '/kaggle/working/')
    summary['runs'][n] = res
for m in sorted(glob.glob('/kaggle/input/**/meshes', recursive=True)) if '__AMODE__' == 'light' else []:
    if glob.glob(m + '/w[0-9][0-9][0-9]/x.tif'):
        label = os.path.relpath(os.path.dirname(m), '/kaggle/input').replace('/', '__')
        summary['earlier_runs'][label] = audit('earlier__' + label, m)
json.dump(summary, open('/kaggle/working/summary.json', 'w'), indent=1)
for n, r in list(summary['runs'].items()) + list(summary['earlier_runs'].items()):
    ah, at = r.get('agreement_heldout') or {}, r.get('agreement_train') or {}
    print(f"RESULT {n}: {r.get('windings')} windings; held-out rays: one winding per model sheet {ah.get('frac_exact_d1')} of "
          f"{ah.get('pairs_d1')} pairs (count dist {ah.get('count_d1_distribution')}), 5-sheet spans exact {ah.get('frac_exact_d5')} "
          f"(median {ah.get('median_count_d5')}), crossing to nearest winding {ah.get('median_crossing_to_nearest_winding_vox')} vox, "
          f"by radius {ah.get('by_radius_vox')}; training rays {at.get('frac_exact_d1')}, nearest {at.get('median_crossing_to_nearest_winding_vox')} vox")
print('SUMMARY ' + json.dumps(summary))
print(f'total {(time.time() - T0) / 3600:.2f} h')
""", Z0=Z0, Z1=Z1, SCROLL=SCROLL, VILLA8=VILLA_COMMIT[:8], AMODE=AUDIT_MODE)
(code if AUDIT_MODE not in ("light", "none") else (lambda *a, **k: None))("""
# 7. Audit every run (seed fit, refits, and earlier runs attached as inputs): winding pitch (overall and per
#    90-degree sector), radius per winding, render contrast on three windings x three blocks, CT cross-sections
#    with the windings drawn on, and winding_agreement.py on the held-out (and the training) crossing stores.
TOOLS = WORK + '/atlas/tools'
os.makedirs(TOOLS, exist_ok=True)
for fname, src in TOOLS_SRC.items():
    open(os.path.join(TOOLS, fname), 'w').write(src)
sys.path.insert(0, TOOLS)
sh('pip install -q tifffile numcodecs')
import tifffile
from vcz import OmeZarr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ZMID = (__Z0__ + __Z1__) // 2
L1 = OmeZarr(S3 + '/__VOLUME__', cache_dir=WORK + '/l1cache').level('1')

def plane_polyline(xx, yy, zz, zc):
    # where a winding mesh crosses the plane z = zc: one point per grid column (linear along the rows), NaN where it does not
    ok = (xx >= 0) & (yy >= 0) & (zz >= 0)
    d = zz - zc
    s = (d[:-1] * d[1:] <= 0) & ok[:-1] & ok[1:] & (d[:-1] != d[1:])
    has = s.any(0)
    r = np.argmax(s, 0)
    c = np.arange(zz.shape[1])
    t = d[r, c] / np.where(has, d[r, c] - d[np.minimum(r + 1, zz.shape[0] - 1), c], 1)
    rx = np.minimum(r + 1, zz.shape[0] - 1)
    px = np.where(has, xx[r, c] + t * (xx[rx, c] - xx[r, c]), np.nan)
    py = np.where(has, yy[r, c] + t * (yy[rx, c] - yy[r, c]), np.nan)
    return px, py


def audit(name, g):
    ws = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(g + '/w[0-9][0-9][0-9]') if os.path.exists(d + '/x.tif'))
    res = {'mesh_dir': g, 'windings': len(ws)}
    if not ws:
        return res
    for tag, store in (('heldout', HELD), ('train', TRAIN)):
        out = sh(f'python {TOOLS}/winding_agreement.py {g} {store} --z __Z0__ __Z1__ --max-rays 4000 --umbilicus {ROOT}/umbilicus.json', check=False, tail=1500)
        res['agreement_' + tag] = next((json.loads(l) for l in out.splitlines() if l.startswith('{')), None)
    if '__AUDIT_MODE__' == 'light':  # the rest runs in the CPU audit notebook (make_audit_notebook.py), not on GPU time
        return res
    starts = [w for w in ws if (w + 1) in ws]
    out = sh(f'python {TOOLS}/winding_pitch.py {g} --start {" ".join(map(str, starts))} --sheet-um __SHEET_UM__ --umbilicus {ROOT}/umbilicus.json', check=False, tail=1500)
    res['pitch'] = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    rad, traces = {}, []
    for w in ws:
        xx, yy, zz = (tifffile.imread(f'{g}/w{w:03d}/{c}.tif') for c in 'xyz')
        ok = (xx >= 0) & (yy >= 0) & (zz >= 0)
        if ok.sum() >= 10:
            rad[w] = round(float(np.median(np.hypot(yy[ok] - np.interp(zz[ok], uz, uy_), xx[ok] - np.interp(zz[ok], uz, ux_)))), 1)
        traces.append((w,) + plane_polyline(xx, yy, zz, ZMID))
    res['radius_vox_by_winding'] = rad
    res['render_contrast'] = []
    for w in [ws[len(ws) // 4], ws[len(ws) // 2], ws[3 * len(ws) // 4]]:
        ok = tifffile.imread(f'{g}/w{w:03d}/x.tif') >= 0
        blocks = [(r, c) for r in range(0, ok.shape[0] - 21, 3) for c in range(0, ok.shape[1] - 21, 3) if ok[r:r + 21, c:c + 21].all()]
        for r, c in ([blocks[i] for i in np.linspace(0, len(blocks) - 1, 3).astype(int)] if blocks else []):
            rc = sh(f'python {TOOLS}/render_contrast.py {g}/w{w:03d} {S3}/__VOLUME__ --rows {r}:{r + 21} --cols {c}:{c + 21} --smooth 0', check=False, tail=1500)
            res['render_contrast'] += [json.loads(l) for l in rc.splitlines() if l.startswith('{')]
    try:
        cy0, cx0 = float(np.interp(ZMID, uz, uy_)), float(np.interp(ZMID, uz, ux_))
        cols = plt.cm.tab10(np.arange(10))
        res['xsection_png'] = []
        for tag, (cy, cx, half) in {'full': (cy0, cx0, 1536), 'zoom': (cy0, cx0 + 1000, 256)}.items():
            y0, x0 = int(cy / 2) - half, int(cx / 2) - half
            img = L1.read(ZMID // 2, ZMID // 2 + 1, max(y0, 0), y0 + 2 * half, max(x0, 0), x0 + 2 * half)[0]
            ext = (max(x0, 0) * 2, (max(x0, 0) + img.shape[1]) * 2, (max(y0, 0) + img.shape[0]) * 2, max(y0, 0) * 2)
            fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
            ax.imshow(img, cmap='gray', extent=ext)
            for w, x_, y_ in traces:
                ax.plot(x_, y_, lw=0.6 if tag == 'full' else 1.4, color=cols[w % 10])
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(f'__SCROLL__ z={ZMID} ({name}, {tag}): fitted windings (colours cycle every 10) over the CT')
            ax.set_xlabel('x (voxels)'); ax.set_ylabel('y (voxels)')
            fn = f'{name}_xsection_{tag}_z{ZMID}.png'
            fig.savefig('/kaggle/working/' + fn, bbox_inches='tight'); plt.close(fig)
            res['xsection_png'].append(fn)
    except Exception as e:
        print('cross-section failed:', e)
    return res

summary = {'scroll': '__SCROLL__', 'z': [__Z0__, __Z1__], 'villa': '__VILLA8__', 'sheet_um': __SHEET_UM__,
           'bench': BENCH, 'autocast': AUTOCAST, 'seed_spacing': SPACING, 'stores': STORES,
           'runs': {}, 'earlier_runs': {}}
for n in ['S_seed'] + names:
    groups = sorted(set(os.path.dirname(os.path.dirname(p)) for p in glob.glob(f'{OUT}/{n}/meshes/*/w[0-9][0-9][0-9]/x.tif')))
    res = {'exit': codes.get(n), 'overrides': SEED if n == 'S_seed' else RUNS[n]}
    if not os.path.exists(f'{OUT}/{n}.log'):
        summary['runs'][n] = res
        continue
    if groups:
        res.update(audit(n, groups[0]))
        shutil.copytree(groups[0], f'/kaggle/working/{n}/meshes', dirs_exist_ok=True)
    for f in glob.glob(f'{OUT}/{n}/*.json'):
        os.makedirs(f'/kaggle/working/{n}', exist_ok=True)
        shutil.copy2(f, f'/kaggle/working/{n}/')
    shutil.copy2(f'{OUT}/{n}.log', '/kaggle/working/')
    summary['runs'][n] = res
for m in sorted(glob.glob('/kaggle/input/**/meshes', recursive=True)):
    if glob.glob(m + '/w[0-9][0-9][0-9]/x.tif'):
        label = os.path.relpath(os.path.dirname(m), '/kaggle/input').replace('/', '__')
        summary['earlier_runs'][label] = audit('earlier__' + label, m)
for x in __EXTRA_MESHES__:  # published fits of the same band in git repos
    d = WORK + '/extra/' + x['label']
    if not os.path.isdir(d):
        sh(f"git clone --depth 1 {x['repo']} {d}", check=False)
    m = d + '/' + x['path']
    if glob.glob(m + '/w[0-9][0-9][0-9]/x.tif'):
        summary['earlier_runs'][x['label']] = dict(audit('earlier__' + x['label'], m), source=x['repo'])
json.dump(summary, open('/kaggle/working/summary.json', 'w'), indent=1)
for n, r in list(summary['runs'].items()) + list(summary['earlier_runs'].items()):
    p = [x for x in r.get('pitch', []) if 'gap_over_sheet_spacing' in x and not x.get('summary')]
    sm = [x for x in r.get('pitch', []) if x.get('summary')]
    g_ = np.array([x['gap_over_sheet_spacing'] for x in p]) if p else np.array([np.nan])
    cv = [x['contrast_median'] for x in r.get('render_contrast', []) if x.get('contrast_median') is not None]
    ah, at = r.get('agreement_heldout') or {}, r.get('agreement_train') or {}
    print(f"RESULT {n}: {r.get('windings')} windings; held-out rays: exactly one winding per model sheet in "
          f"{ah.get('frac_exact_d1')} of {ah.get('pairs_d1')} pairs (count dist {ah.get('count_d1_distribution')}), "
          f"5-sheet spans exact {ah.get('frac_exact_d5')} (median {ah.get('median_count_d5')}), by radius {ah.get('by_radius_vox')}; "
          f"training rays {at.get('frac_exact_d1')}; pitch/sheet median {np.nanmedian(g_):.2f} "
          f"(IQR {np.nanpercentile(g_, 25):.2f}-{np.nanpercentile(g_, 75):.2f}, {len(p)} pairs), by sector "
          f"{sm[0].get('gap_over_sheet_spacing_by_sector') if sm else None}; render contrast median "
          f"{np.median(cv) if cv else float('nan'):.3f} ({len(cv)} blocks)")
print('SUMMARY ' + json.dumps(summary))
print(f'total {(time.time() - T0) / 3600:.2f} h')
""", Z0=Z0, Z1=Z1, VOLUME=VOLUME, SHEET_UM=spec["sheet_um"], SCROLL=SCROLL, VILLA8=VILLA_COMMIT[:8],
     EXTRA_MESHES=repr(spec.get("extra_meshes", [])), AUDIT_MODE=spec.get("audit", "full"))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"))
print("wrote", sys.argv[2], len(cells), "cells;", SCROLL, Z0, Z1, "seed", SEED_STEPS, "runs", list(RUNS), "steps", STEPS)
if os.path.getsize(sys.argv[2]) > 59000:  # Kaggle's File -> Import Notebook fails silently above ~60 KB
    print("WARNING:", os.path.getsize(sys.argv[2]), "bytes: Kaggle's importer may reject it silently (keep it under ~59 KB,"
          " e.g. with \"audit\": \"none\" in the spec)")
