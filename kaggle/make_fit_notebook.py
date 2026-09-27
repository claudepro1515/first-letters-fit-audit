"""Build a Kaggle notebook that runs one or two spiral fits of a band (one per T4) and audits them.

The spec is a JSON file, for example:
  {"scroll": "PHerc0211", "run": "20250821151803", "umbilicus": "20250821151803-umbilicus-20260808112626.json",
   "lasagna": "20250821151803-lasagna-20260419180421", "volume": "20250821151803-9.362um-1.2m-113keV-masked.zarr",
   "z": [8000, 9000], "sheet_um": 141, "sense": "CW", "steps": 30000,
   "base": {...config overrides shared by all runs...},
   "runs": {"B_spacing": {...extra overrides...}},
   "title": "...", "notes": "..."}

Runs are queued on the two GPUs (run i on GPU (i + 1) % 2, one at a time per GPU). The first run starts alone and
builds the shared cache in its first step (~10 min); the others start once that step is done, each as soon as its
GPU is free. "audit": "pitch_only" in the spec skips the render contrast and cross-sections (use the CPU audit
notebook, make_audit_notebook.py, for the full audit without GPU time). A deadline stops the fits in time
for the audit inside Kaggle's 12 h limit. The audit uses first-letters-scan-atlas: winding pitch on every consecutive pair
(gap / the scan's sheet spacing) and render contrast on three blocks of three windings. Meshes, logs,
fitter metrics and summary.json go to /kaggle/working; the last printed line starts with 'SUMMARY'.

Usage: python make_fit_notebook.py spec.json out.ipynb
"""
import json
import os
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
SHEET_UM = spec["sheet_um"]
SENSE = spec["sense"]
STEPS = spec.get("steps", 30000)
VILLA_COMMIT = spec.get("villa", "75c79ac5f506d4b9a89bcfbef8e8c0f2f0c3acb3")
BASE = dict(spec["base"], z_begin=Z0, z_end=Z1, optimizer_num_training_steps=STEPS)
RUNS = spec["runs"]
VARIANTS = spec.get("variants", {})  # per-run grad_mag rewrites: gradmag_constant_spacing_vox, gradmag_clamp, gradmag_core_radius_vox
assert len(RUNS) >= 1
AUDIT_MODE = spec.get("audit", "full")

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip()})


md(f"""# {spec.get('title', f'Spiral fit of {SCROLL}, z [{Z0}, {Z1})')}

Settings: **Accelerator GPU T4 x2**, **Internet on**. Runs as a background commit (Save Version -> Save & Run All).
{spec.get('notes', '')}
Audited with [first-letters-scan-atlas](https://github.com/claudepro1515/first-letters-scan-atlas): winding pitch
against the scan's sheet spacing ({SHEET_UM} um for {SCROLL}) and render contrast.""")

code("""
import os, subprocess, sys, time, json, shutil, glob, re
T0 = time.time()
def sh(cmd, check=True, cwd=None, env=None):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-6000:], flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')
    return p.stdout
sh('nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv')
sh('df -h /kaggle/working / | cat; free -g; nproc')
WORK = '/kaggle/temp'
os.makedirs(WORK, exist_ok=True)
""")

code(f"""
# 1. uv + Python 3.14 + villa at a pinned commit (spiral-fitting only)
sh('curl -LsSf https://astral.sh/uv/install.sh | sh')
os.environ['PATH'] = os.path.expanduser('~/.local/bin') + ':' + os.environ['PATH']
sh('uv python install 3.14')
os.chdir(WORK)
if not os.path.isdir('villa'):
    sh('git clone --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git')
    sh('git sparse-checkout init --cone && git sparse-checkout set spiral-fitting lasagna vesuvius', cwd='villa')
    sh('git checkout {VILLA_COMMIT}', cwd='villa')
SF = WORK + '/villa/spiral-fitting'
t = time.time(); sh('uv sync', cwd=SF); print(f'uv sync {{(time.time()-t)/60:.1f}} min')
CHECK = '''
import torch
a = torch.randn(512, 512, device="cuda"); b = (a @ a).sum().item(); assert b == b
print("KERNEL OK", torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))
'''
open(WORK + '/gpu_check.py', 'w').write(CHECK)
sh(SF + '/.venv/bin/python ' + WORK + '/gpu_check.py')
""")

code(f"""
# 2. Data: umbilicus, tracks (DBM + packed store + crossings), Lasagna nx/ny/grad_mag for the band only
import urllib.request, urllib.parse, concurrent.futures as cf, xml.etree.ElementTree as ET
S3 = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
DL = 'https://dl.ash2txt.org/datasets/spiral_datasets/{SCROLL}/{RUN}/tracks'
ROOT = WORK + '/dataset_{SCROLL}'
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
                raise IOError(f'size mismatch {{os.path.getsize(tmp)}} != {{n}}')
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception as e:
            print('retry', k, url, e, flush=True); time.sleep(5)
    raise RuntimeError('download failed: ' + url)

t = time.time()
fetch('{UMB_URL}', ROOT + '/umbilicus.json')
for name in ['{TRACKS}.dbm', '{TRACKS}.dbm.crossings.npz', '{TRACKS}.extract.json']:
    print(name, fetch(DL + '/' + name, ROOT + '/tracks/' + name) / 2**30, 'GB', flush=True)
VT = ROOT + '/tracks/{TRACKS}.dbm.vctracks'
for name in ['metadata.json', 'header.bin', 'offsets.i64', 'source_ids.u64', 'family_codes.i8', 'z_bounds.i32',
             'arclengths.f64', 'tortuosities.f64', 'coordinates.i32']:
    print(name, fetch(DL + '/{TRACKS}.dbm.vctracks/' + name, VT + '/' + name) / 2**30, 'GB', flush=True)
# the packed store and crossings are tied to the DBM's (name, size, mtime_ns): restore the publisher's mtime
sig = json.load(open(VT + '/metadata.json'))['source_db_signature'][0]
dbm_path = ROOT + '/tracks/' + sig[0]
assert os.path.getsize(dbm_path) == sig[1], 'DBM size differs from the packed store signature'
os.utime(dbm_path, ns=(sig[2], sig[2]))
print('tracks ready', (time.time() - t) / 60, 'min')

def list_keys(prefix):
    keys, token = [], None
    while True:
        q = f'{{S3}}/?list-type=2&prefix={{prefix}}' + (f'&continuation-token={{urllib.parse.quote(token)}}' if token else '')
        x = ET.fromstring(urllib.request.urlopen(q, timeout=120).read())
        ns = {{'s': x.tag.split('}}')[0].strip('{{')}}
        keys += [c.find('s:Key', ns).text for c in x.findall('s:Contents', ns)]
        if x.find('s:IsTruncated', ns).text != 'true':
            return keys
        token = x.find('s:NextContinuationToken', ns).text
zc0, zc1 = ({Z0} // 4 - 32) // 32, ({Z1} // 4 + 32) // 32
for ch in ['nx', 'ny', 'grad_mag']:
    src = '{LAS}/{SCROLL}_' + ch + '.ome.zarr'
    dst = ROOT + '/lasagna_inputs/las_008_' + ch + '.ome.zarr'
    for meta in ['.zgroup', '.zattrs', '2/.zarray', '2/.zattrs']:
        try:
            fetch(S3 + '/' + src + '/' + meta, dst + '/' + meta, tries=1)
        except Exception as e:
            print('no', meta, e)
    keys = []
    for zc in range(zc0, zc1 + 1):
        keys += list_keys(src + f'/2/{{zc}}/')
    with cf.ThreadPoolExecutor(32) as ex:
        sizes = list(ex.map(lambda k: fetch(S3 + '/' + k, dst + '/' + k[len(src) + 1:]), keys))
    print(ch, len(keys), 'chunks', sum(sizes) / 2**20, 'MB', flush=True)
json.dump({{"schema_version": 1, "name": "{SCROLL}", "voxel_size_um": 9.362, "spiral_outward_sense": "{SENSE}",
           "normal_zarr_group": "2", "lasagna_scale": 4, "paths": {{"tracks_dbm": "tracks/{TRACKS}.dbm"}}}},
          open(ROOT + '/spiral-scroll.json', 'w'), indent=2)
sh(f'du -sh {{ROOT}}/* | cat')
""")

if spec.get("store_from_input"):
    code("""
# 2b. A winding-model crossing store attached as input (the winding_inference_train folder written by a
#     *winding-model-supervision* notebook) for dense_spacing_mode winding_model, and the outer shell that mode
#     requires: the scan's own boundary (for every z and angle, where the ray from the umbilicus leaves the volume),
#     so it removes nothing.
import numpy as np
from PIL import Image
STORE_IN = sorted(p for p in glob.glob('/kaggle/input/**/winding_inference_train', recursive=True)
                  if os.path.exists(p + '/manifest.json'))
assert STORE_IN, 'no winding_inference_train store attached'
shutil.copytree(STORE_IN[0], ROOT + '/winding_inference', dirs_exist_ok=True)
print('winding-model store:', STORE_IN[0], json.load(open(ROOT + '/winding_inference/manifest.json')).get('fingerprint'))
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
""".replace("__VOLUME__", VOLUME).replace("__Z0__", str(Z0)).replace("__Z1__", str(Z1)))

code(f"""
# 3. Configurations. Every key exists in villa {VILLA_COMMIT[:8]} (unknown keys abort the fit).
BASE = {BASE!r}
RUNS = {{n: dict(BASE, **o) for n, o in {RUNS!r}.items()}}
VARIANTS = {VARIANTS!r}
OUT = WORK + '/spiral_out'
sh('pip install -q numcodecs tifffile')
import numcodecs, numpy as np

def build_modified_gradmag(src_root, dst_root, v):
    # Same dataset, with the Lasagna grad_mag winding density (u8 = 1000 x windings per full-res voxel,
    # 0 = invalid) rewritten chunk by chunk: 'gradmag_constant_spacing_vox' replaces every valid value by
    # 1000 / spacing; 'gradmag_clamp' [lo, hi] clips valid values (e.g. [40, 80] = 12.5-25 voxels per
    # winding); 'gradmag_core_radius_vox' marks as invalid everything closer to the umbilicus than that.
    os.makedirs(dst_root + '/lasagna_inputs', exist_ok=True)
    for name in os.listdir(src_root):
        if name != 'lasagna_inputs' and not os.path.exists(dst_root + '/' + name):
            os.symlink(src_root + '/' + name, dst_root + '/' + name)
    for name in os.listdir(src_root + '/lasagna_inputs'):
        if 'grad_mag' not in name and not os.path.exists(dst_root + '/lasagna_inputs/' + name):
            os.symlink(src_root + '/lasagna_inputs/' + name, dst_root + '/lasagna_inputs/' + name)
    src = src_root + '/lasagna_inputs/las_008_grad_mag.ome.zarr'
    dst = dst_root + '/lasagna_inputs/las_008_grad_mag.ome.zarr'
    meta = json.load(open(src + '/2/.zarray'))
    codec = numcodecs.get_codec(meta['compressor'])
    cs = meta['chunks']
    cp = sorted(json.load(open(src_root + '/umbilicus.json'))['control_points'], key=lambda q: q['z'])
    uz = np.array([q['z'] for q in cp], float); uy = np.array([q['y'] for q in cp], float); ux = np.array([q['x'] for q in cp], float)
    core = float(v.get('gradmag_core_radius_vox', 0))
    n = changed = 0
    for dirpath, dirs, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            p, q = os.path.join(dirpath, f), os.path.join(dst, rel, f)
            if f.startswith('.') or not rel.startswith('2'):
                shutil.copy2(p, q)
                continue
            a = np.frombuffer(codec.decode(open(p, 'rb').read()), np.uint8).copy().reshape(cs)
            valid = a > 0
            if 'gradmag_constant_spacing_vox' in v:
                a[valid] = int(round(1000 / float(v['gradmag_constant_spacing_vox'])))
            if 'gradmag_clamp' in v:
                lo, hi = v['gradmag_clamp']
                a[valid] = np.clip(a[valid], lo, hi)
            if core > 0:
                cz, cy, cx = (int(t) for t in rel.split('/')[1:] + [f])
                zz = (cz * cs[0] + np.arange(cs[0])) * 4.0
                yy = (cy * cs[1] + np.arange(cs[1])) * 4.0
                xx = (cx * cs[2] + np.arange(cs[2])) * 4.0
                r = np.hypot(yy[None, :, None] - np.interp(zz, uz, uy)[:, None, None],
                             xx[None, None, :] - np.interp(zz, uz, ux)[:, None, None])
                a[r < core] = 0
            changed += int((a != np.frombuffer(codec.decode(open(p, 'rb').read()), np.uint8).reshape(cs)).sum())
            open(q, 'wb').write(codec.encode(a.ravel()))
            n += 1
    print('modified grad_mag', v, 'in', n, 'chunks,', changed, 'voxels changed:', dst, flush=True)

ROOTS = {{}}
for n in RUNS:
    v = VARIANTS.get(n, {{}})
    if any(k.startswith('gradmag_') for k in v):
        ROOTS[n] = WORK + '/dataset_' + n
        build_modified_gradmag(ROOT, ROOTS[n], v)
    else:
        ROOTS[n] = ROOT

def launch(name, cfg, gpu, steps=None):
    c = dict(cfg)
    if steps: c['optimizer_num_training_steps'] = steps
    env = dict(os.environ, FIT_SPIRAL_CONFIG_OVERRIDES=json.dumps(c), FIT_SPIRAL_RUN_DIR=f'{{OUT}}/{{name}}',
               FIT_SPIRAL_CACHE_DIR=WORK + '/spiral_cache', WANDB_MODE='disabled', CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONUNBUFFERED='1')
    os.makedirs(f'{{OUT}}/{{name}}', exist_ok=True)
    json.dump(c, open(f'{{OUT}}/{{name}}/overrides.json', 'w'), indent=1)
    log = open(f'{{OUT}}/{{name}}.log', 'w')
    return subprocess.Popen([SF + '/.venv/bin/python', '-u', 'fit_spiral.py', '--dataset', ROOTS.get(name, ROOT)],
                            cwd=SF, env=env, stdout=log, stderr=subprocess.STDOUT)

def mem_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return round(int(line.split()[1]) / 2**20, 1)

def wait(procs, every=180, deadline=None):
    # deadline: stop the fits in time for the audit to finish inside Kaggle's 12 h limit
    while any(p.poll() is None for p in procs.values()):
        time.sleep(every)
        for n in procs:
            tail = open(f'{{OUT}}/{{n}}.log').read()[-300:].replace('\\r', ' | ').replace('\\n', ' | ')
            print(f'[{{(time.time()-T0)/60:.0f}} min, {{mem_gb()}} GB free] {{n}}: {{tail}}', flush=True)
        if deadline and time.time() > deadline:
            print('DEADLINE reached: stopping the fits', flush=True)
            for p in procs.values():
                if p.poll() is None:
                    p.terminate()
            time.sleep(60)
            for p in procs.values():
                if p.poll() is None:
                    p.kill()
    return {{n: p.returncode for n, p in procs.items()}}
""")

code(f"""
# 4. Fits, one per GPU. The first run's first optimisation step builds the shared cache (crossing maps,
#    ~10 min on a T4); the other run starts once that step is done and reuses it. Then ~5 steps/s, so
#    30,000 steps take ~1.7 h. A deadline stops the fits in time for the audit if anything runs long
#    (a commit killed at Kaggle's 12 h limit keeps nothing).
LIMIT_S, AUDIT_S = 11.75 * 3600, 0.75 * 3600
names = list(RUNS)
procs = {{}}
first = names[0]
procs[first] = launch(first, RUNS[first], 1 if len(names) > 1 else 0)
t = time.time()
while procs[first].poll() is None and time.time() - t < 3 * 3600:
    if re.search(r'Optimizing — [1-9][\\d,]*/', open(f'{{OUT}}/{{first}}.log').read()):
        break
    time.sleep(30)
print(first, 'past its first step after', round((time.time() - t) / 60, 1), 'min; exit code so far:', procs[first].poll(), flush=True)
if procs[first].poll() not in (None, 0):
    print(open(f'{{OUT}}/{{first}}.log').read()[-3000:].replace('\\r', '\\n'))
    raise RuntimeError(first + ' failed early: see the log above')
# the other runs: run i goes on GPU (i + 1) % 2, one run at a time per GPU
queue = {{g: [n for i, n in enumerate(names) if i > 0 and (i + 1) % 2 == g] for g in (0, 1)}}
running = {{first: 1 if len(names) > 1 else 0}}
deadline = T0 + LIMIT_S - AUDIT_S
while True:
    for g in (0, 1):
        if queue[g] and g not in running.values():
            n = queue[g].pop(0)
            procs[n] = launch(n, RUNS[n], g)
            running[n] = g
            print('started', n, 'on GPU', g, flush=True)
    if not running:
        break
    time.sleep(180)
    for n in list(running):
        tail = open(f'{{OUT}}/{{n}}.log').read()[-300:].replace('\\r', ' | ').replace('\\n', ' | ')
        print(f'[{{(time.time()-T0)/60:.0f}} min, {{mem_gb()}} GB free] {{n}}: {{tail}}', flush=True)
        if procs[n].poll() is not None:
            print(n, 'finished with exit code', procs[n].returncode, flush=True)
            del running[n]
    if time.time() > deadline:
        print('DEADLINE reached: stopping the fits and dropping the queue', flush=True)
        queue = {{0: [], 1: []}}
        for n in running:
            procs[n].terminate()
        time.sleep(60)
        for n in running:
            if procs[n].poll() is None:
                procs[n].kill()
        running = {{}}
codes = {{n: p.returncode for n, p in procs.items()}}
steps = {{n: RUNS[n]['optimizer_num_training_steps'] for n in names}}
print('runs', codes)
for n in RUNS:
    print('=====', n); print(open(f'{{OUT}}/{{n}}.log').read()[-2500:].replace('\\r', '\\n'))
""")

AUDIT = """
# 5. Audit: winding pitch on every consecutive pair (overall, per 90-degree sector and per winding radius),
#    render contrast on three blocks of three windings, and CT cross-sections with the windings drawn on.
#    The tools are written from this notebook (first-letters-scan-atlas, the versions used here), so the
#    audit does not depend on the repository's current state. Earlier runs attached as inputs
#    (/kaggle/input/*/<run>/meshes/wNNN) are audited the same way.
sh('pip install -q tifffile numcodecs')
TOOLS = WORK + '/atlas/tools'
os.makedirs(TOOLS, exist_ok=True)
for fname, src in TOOLS_SRC.items():
    open(os.path.join(TOOLS, fname), 'w').write(src)
sys.path.insert(0, TOOLS)
import tifffile, numpy as np
from vcz import OmeZarr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cp = sorted(json.load(open(ROOT + '/umbilicus.json'))['control_points'], key=lambda q: q['z'])
uz = np.array([q['z'] for q in cp], float); uy_ = np.array([q['y'] for q in cp], float); ux_ = np.array([q['x'] for q in cp], float)
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
    starts = [w for w in ws if (w + 1) in ws]
    out = sh(f'python {TOOLS}/winding_pitch.py {g} --start {" ".join(map(str, starts))} --sheet-um __SHEET_UM__ --umbilicus {ROOT}/umbilicus.json', check=False)
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
    if '__AUDIT_MODE__' == 'pitch_only':
        return res
    for w in [ws[len(ws) // 4], ws[len(ws) // 2], ws[3 * len(ws) // 4]]:
        ok = tifffile.imread(f'{g}/w{w:03d}/x.tif') >= 0
        blocks = [(r, c) for r in range(0, ok.shape[0] - 21, 3) for c in range(0, ok.shape[1] - 21, 3) if ok[r:r + 21, c:c + 21].all()]
        for r, c in ([blocks[i] for i in np.linspace(0, len(blocks) - 1, 3).astype(int)] if blocks else []):
            rc = sh(f'python {TOOLS}/render_contrast.py {g}/w{w:03d} {S3}/__VOLUME__ --rows {r}:{r+21} --cols {c}:{c+21} --smooth 0', check=False)
            res['render_contrast'] += [json.loads(l) for l in rc.splitlines() if l.startswith('{')]
    try:
        cy0, cx0 = float(np.interp(ZMID, uz, uy_)), float(np.interp(ZMID, uz, ux_))
        cols = plt.cm.tab10(np.arange(10))
        res['xsection_png'] = []
        for tag, (cy, cx, half) in {'full': (cy0, cx0, 1536), 'zoom': (cy0, cx0 + 1000, 256)}.items():
            y0, x0 = int(cy / 2) - half, int(cx / 2) - half   # level-1 pixels = 2 full-res voxels
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

summary = {'scroll': '__SCROLL__', 'z': [__Z0__, __Z1__], 'steps': steps, 'villa': '__VILLA__',
           'sheet_um': __SHEET_UM__, 'sense': '__SENSE__', 'runs': {}, 'earlier_runs': {}}
for n in RUNS:
    groups = sorted(set(os.path.dirname(os.path.dirname(p)) for p in glob.glob(f'{OUT}/{n}/meshes/*/w[0-9][0-9][0-9]/x.tif')))
    res = {'exit': codes.get(n), 'overrides': RUNS[n], 'variant': VARIANTS.get(n, {})}
    if groups:
        res.update(audit(n, groups[0]))
        shutil.copytree(groups[0], f'/kaggle/working/{n}/meshes', dirs_exist_ok=True)
    for f in glob.glob(f'{OUT}/{n}/*.json'):
        shutil.copy2(f, f'/kaggle/working/{n}/' if os.path.isdir(f'/kaggle/working/{n}') else '/kaggle/working/')
    shutil.copy2(f'{OUT}/{n}.log', '/kaggle/working/')
    summary['runs'][n] = res
for m in sorted(glob.glob('/kaggle/input/**/meshes', recursive=True)):
    if glob.glob(m + '/w[0-9][0-9][0-9]/x.tif'):
        label = os.path.relpath(os.path.dirname(m), '/kaggle/input').replace('/', '__')
        summary['earlier_runs'][label] = audit('earlier__' + label, m)
json.dump(summary, open('/kaggle/working/summary.json', 'w'), indent=1)
for n, r in list(summary['runs'].items()) + list(summary['earlier_runs'].items()):
    p = [x for x in r.get('pitch', []) if 'gap_over_sheet_spacing' in x and not x.get('summary')]
    sm = [x for x in r.get('pitch', []) if x.get('summary')]
    g_ = np.array([x['gap_over_sheet_spacing'] for x in p]) if p else np.array([np.nan])
    cv = [x['contrast_median'] for x in r.get('render_contrast', []) if x.get('contrast_median') is not None]
    print(f"RESULT {n}: {r.get('windings')} windings, pitch/sheet median {np.nanmedian(g_):.2f} "
          f"(IQR {np.nanpercentile(g_, 25):.2f}-{np.nanpercentile(g_, 75):.2f}, {len(p)} pairs), by sector "
          f"{sm[0].get('gap_over_sheet_spacing_by_sector') if sm else None}, render contrast median "
          f"{np.median(cv) if cv else float('nan'):.3f} ({len(cv)} blocks)")
print('SUMMARY ' + json.dumps(summary))
print(f'total {(time.time()-T0)/3600:.2f} h')
"""
TOOL_FILES = ["vcz.py", "mesh_roughness.py", "winding_pitch.py", "render_contrast.py"]
TOOLS_DIR = spec.get("tools_dir") or _find(os.path.join(HERE, "..", "tools"), "/home/claude/vc/release/tools")
tools_src = {f: open(os.path.join(TOOLS_DIR, f)).read() for f in TOOL_FILES}
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": "# Tools used by the audit (first-letters-scan-atlas, embedded verbatim)\nTOOLS_SRC = " + repr(tools_src)})
code(AUDIT.replace("__SCROLL__", SCROLL).replace("__Z0__", str(Z0)).replace("__Z1__", str(Z1))
     .replace("__VOLUME__", VOLUME).replace("__SHEET_UM__", str(SHEET_UM)).replace("__SENSE__", SENSE)
     .replace("__VILLA__", VILLA_COMMIT[:8]).replace("__AUDIT_MODE__", AUDIT_MODE))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2], len(cells), "cells;", SCROLL, Z0, Z1, "runs", list(RUNS), "steps", STEPS)
if os.path.getsize(sys.argv[2]) > 59000:  # Kaggle's File -> Import Notebook fails silently above ~60 KB
    print("WARNING:", os.path.getsize(sys.argv[2]), "bytes: Kaggle's importer may reject it silently (keep it under ~59 KB,"
          " e.g. with \"audit\": \"none\" in the spec)")
