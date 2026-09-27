"""Build a Kaggle notebook that renders spiral-fit windings and runs the public ink model on them.

  0. control: a crop of the curated PHerc0139 w035 segment (9.362 um canvas, ink labels published) rendered
     two ways, the published surface volume and our renderer on the published mesh, both through ink_9um,
     scored against the labels (AUC). This checks the whole render -> ink chain before it is used on meshes
     without labels;
  1. for every run named in the spec, found among the attached notebook outputs (/kaggle/input/**/<run>/meshes),
     and every winding in the spec ('all' = every exported winding): inkjob.py = render_sv.py (28 layers along
     the normal, published convention, CT from a local copy of the slices the meshes touch) -> koine_machines
     infer.py with scrollprize/ink_9um, both layer orders -> PNGs in /kaggle/working, one JSON line per job,
     and one overview image per run and layer order.

The ink work runs in subprocesses, and the missing packages are installed with the kernel's numpy/scipy/torch versions
pinned (an unpinned install upgraded numpy and broke scipy in the kernel on Kaggle; Kaggle's python has no venv).

Usage: python make_ink_notebook.py spec.json out.ipynb   (spec: see spec_0826_ink.json)
"""
import base64
import hashlib
import io
import json
import os
import lzma
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(*cands):
    """First existing path: the repo layout (../wm, ../ink, ../tools next to kaggle/) or the dev layout."""
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(cands)


spec = json.load(open(sys.argv[1]))
SCROLL = spec["scroll"]
VOLUME = f"{SCROLL}/volumes/{spec['volume']}"
VILLA = spec.get("villa", "75c79ac5f506d4b9a89bcfbef8e8c0f2f0c3acb3")
INK_COMMIT = spec.get("ink_commit", "3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e")
INK_CKPT = spec.get("ink_ckpt", "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth")
CTRL = spec["control"]
cells = []


def packed(var, files, what):
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode(), preset=9 | lzma.PRESET_EXTREME)).decode()
    lines = [f"# {what}, embedded as xz-compressed JSON {{path: source}} (sha256 of each file below)"]
    lines += [f"#   {p}  {hashlib.sha256(t.encode()).hexdigest()[:16]}" for p, t in files.items()]
    lines += ["import base64 as _b64, json as _json, lzma as _lzma",
              f"{var} = _json.loads(_lzma.decompress(_b64.b85decode("]
    lines += [f"    {blob[i:i + 120]!r}" for i in range(0, len(blob), 120)]
    lines += [")).decode())"]
    return "\n".join(lines)


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t, **subs):
    for k, v in subs.items():
        t = t.replace(f"__{k}__", str(v))
    left = re.findall(r'__[A-Z][A-Z0-9_]*__', t)
    assert not left, left
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip()})


def png_b64(a):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray((a > 0).astype(np.uint8) * 255).convert("1").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# the control crop's labels travel inside the notebook (the published label store is zarr v3 with sharding)
import zarr  # noqa: E402
r0, r1 = CTRL["rows"]
c0, c1 = CTRL["cols"]
LAB = png_b64(zarr.open_group(CTRL["labels_local"], mode="r")["0"][r0:r1, c0:c1])
SUP = png_b64(zarr.open_group(CTRL["supervision_local"], mode="r")["0"][r0:r1, c0:c1])

md(f"""# {spec['title']}

Settings: **Accelerator GPU T4 x2** (or None: everything also runs on CPU, slowly), **Internet on**.
Runs as a background commit (Save Version -> Save & Run All).
Inputs: the outputs of the fit notebooks whose meshes are rendered (Add Input -> Your Work -> Notebook).

{spec['notes']}

Pipeline:
0. **Control** (checks the chain): PHerc0139 w035 (curated segment, ink labels published on the 9.362 um canvas),
   rows {r0}-{r1}, cols {c0}-{c1}: the published surface volume and our render of the published mesh, both through
   ink_9um; AUC against the labels inside the labelled area.
1. **Windings**: `inkjob.py` renders each winding's full canvas with `render_sv.py` (28 layers, layer L = surface +
   (L - 13) n, n = normalize(cross(dR, dC)), the published convention, checked at r = 0.91-0.96 per layer against a
   published surface volume), then runs `koine_machines/inference/infer.py` (villa `{INK_COMMIT[:8]}`) with
   [`scrollprize/ink_9um`](https://huggingface.co/scrollprize/ink_9um) (layers 6-22, both layer orders).
   Outputs: `<run>_w<NNN>_layer13.png`, `_ink_forward.png`, `_ink_reverse.png`, `overview_<run>_ink_<order>.jpg`
   and `ink_summary.json`.""")

code("""
import os, subprocess, sys, time, json, shutil, glob, re, threading
T0 = time.time()
def sh(cmd, check=True, cwd=None, env=None, tail=4000):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-tail:], flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')
    return p.stdout
def log(*a):
    print(f'[{(time.time() - T0) / 60:.1f} min]', *a, flush=True)
GPUS = [l.strip() for l in sh('nvidia-smi --query-gpu=index --format=csv,noheader', check=False).splitlines() if l.strip().isdigit()]
print('GPUs:', GPUS)
WORK = '/kaggle/temp'
os.makedirs(WORK, exist_ok=True)
sh('df -h /kaggle/working / | cat; free -g; nproc')
""")

code("""
# 1. Code and environment: villa's vesuvius package (network builder) at __VILLA8__ and villa's ink-detection
#    (koine_machines) at __INK8__, a minimal `vesuvius` package exposing only models/utils/image_proc, the ink
#    dependencies (kernel packages pinned), the checkpoint. The ink work runs in subprocesses.
os.chdir(WORK)
for d, c, paths in [('villa', '__VILLA__', 'vesuvius lasagna' if __FLATTEN__ else 'vesuvius'), ('villa_ink', '__INK__', 'ink-detection')]:
    if not os.path.isdir(d):
        sh(f'git clone --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git {d}')
        sh(f'git sparse-checkout init --cone && git sparse-checkout set {paths}', cwd=d)
        sh(f'git checkout {c}', cwd=d)
STUB = WORK + '/stub/vesuvius'
os.makedirs(STUB, exist_ok=True)
open(STUB + '/__init__.py', 'w').close()
for m in ['models', 'utils', 'image_proc']:
    if not os.path.exists(f'{STUB}/{m}'):
        os.symlink(f'{WORK}/villa/vesuvius/src/vesuvius/{m}', f'{STUB}/{m}')
# Kaggle's python has no ensurepip (no venv): install the missing packages into it, but with the versions of the
# packages the kernel already uses pinned (an unpinned install upgraded numpy and broke scipy in the kernel).
VPY = sys.executable
sh(f'{VPY} -m pip freeze | grep -iE "^(numpy|scipy|torch|torchvision|scikit-learn|pillow|pandas)==" > {WORK}/pins.txt; cat {WORK}/pins.txt')
sh(f'{VPY} -m pip install -q -c {WORK}/pins.txt einops timm imagecodecs tifffile numcodecs zarr requests 2>&1 | tail -3', check=False)
INKENV = dict(os.environ, PYTHONPATH=WORK + '/stub:' + WORK + '/villa_ink/ink-detection:' + WORK + '/inktools')
print(subprocess.run([VPY, '-c', 'import numpy, torch, torchvision, timm, zarr, numcodecs, tifffile, imagecodecs, '
                      'koine_machines.inference.infer; print(numpy.__version__, torch.__version__, torchvision.__version__, '
                      'timm.__version__, zarr.__version__, torch.cuda.device_count())'],
                     env=INKENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout[-3000:], flush=True)
CKPT = WORK + '/ink_9um_step075000.pth'
if not os.path.exists(CKPT):
    sh(f'curl -sSL --retry 4 -o {CKPT} __CKPT_URL__')
print('checkpoint', os.path.getsize(CKPT) / 2**20, 'MB')
""", VILLA=VILLA, VILLA8=VILLA[:8], INK=INK_COMMIT, INK8=INK_COMMIT[:8], CKPT_URL=INK_CKPT, FLATTEN=bool(spec.get("flatten")))

INK_DIR = _find(os.path.join(HERE, "..", "ink"), "/home/claude/vc/ink")
VCZ = _find(os.path.join(HERE, "..", "tools", "vcz.py"), "/home/claude/vc/release/tools/vcz.py")
files = {"inkjob.py": open(os.path.join(INK_DIR, "inkjob.py")).read(),
         "render_sv.py": open(os.path.join(INK_DIR, "render_sv.py")).read(),
         "tools/vcz.py": open(VCZ).read()}
ALIGN = spec.get("align") or {}
if ALIGN:
    files["tools/phase_align.py"] = open(os.path.join(os.path.dirname(VCZ), "phase_align.py")).read()
code(packed("SRC", files, "inkjob.py (one render + ink job, and the control), render_sv.py (renderer), vcz.py (open-data reader)") + """
TOOLS = WORK + '/inktools'
for p, t in SRC.items():
    os.makedirs(os.path.dirname(f'{TOOLS}/{p}'), exist_ok=True)
    open(f'{TOOLS}/{p}', 'w').write(t)
S3 = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
CACHE = WORK + '/ctcache'
OUTW = '/kaggle/working'
def run_inkjob(args, gpu=None, tail=3000):
    env = dict(INKENV)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    p = subprocess.run([VPY, TOOLS + '/inkjob.py'] + args, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [l for l in p.stdout.splitlines() if l.startswith('{')]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f'inkjob failed ({p.returncode}): ' + p.stdout[-tail:])
    return json.loads(lines[-1])
""")

code("""
# 2. Control: PHerc0139 w035, published surface volume vs our render of the published mesh, through ink_9um
import base64
CT = WORK + '/control'
os.makedirs(CT, exist_ok=True)
open(CT + '/labels.png', 'wb').write(base64.b64decode('__LAB__'))
open(CT + '/supervision.png', 'wb').write(base64.b64decode('__SUP__'))
try:
    CONTROL = run_inkjob(['control', '--seg', S3 + '/__CTRL_SEG__', '--mesh-name', '__CTRL_MESH__', '--sv-name', '__CTRL_SV__',
                          '--volume', S3 + '/__CTRL_VOL__/0', '--box', '__R0__', '__R1__', '__C0__', '__C1__',
                          '--labels', CT + '/labels.png', '--supervision', CT + '/supervision.png',
                          '--prefix', OUTW + '/control_w035', '--work', CT, '--ckpt', CKPT, '--cache', CACHE],
                         gpu=GPUS[0] if GPUS else None)
except Exception as e:
    CONTROL = {'error': repr(e)[-1500:]}
print('CONTROL ' + json.dumps(CONTROL), flush=True)
""", LAB=LAB, SUP=SUP, CTRL_SEG=CTRL["segment"], CTRL_MESH=CTRL["mesh"], CTRL_SV=CTRL["sv"], CTRL_VOL=CTRL["volume"],
     R0=r0, R1=r1, C0=c0, C1=c1)

code("""
# 3. Windings of the attached fits. Jobs: every (run, winding) in the spec; 'all' = every exported winding.
RUNS = __RUNS__
WINDINGS = __WINDINGS__
VARIANT = '__VARIANT__'
PNG_SCALE = __PNG_SCALE__
DEADLINE_H = __DEADLINE_H__
jobs = []
for run, ws in RUNS.items():
    roots = sorted(glob.glob(f'/kaggle/input/**/{run}/meshes', recursive=True))
    if not roots:
        print('not attached:', run); continue
    if ws == 'all':
        ws = sorted({int(m.group(1)) for d in os.listdir(roots[0]) for m in [re.match(r'^w(\\d{3})', d)] if m})
    for w in (ws or WINDINGS):
        for v in (VARIANT, ''):
            m = f'{roots[0]}/w{w:03d}{v}'
            if os.path.exists(m + '/x.tif'):
                jobs.append((run, w, m)); break
        else:
            print('missing', run, w)
PRIORITY = __PRIORITY__  # windings rendered first (every run of a winding together), then the rest in order
def job_key(j):
    return (PRIORITY.index(j[1]) if j[1] in PRIORITY else len(PRIORITY), j[1], list(RUNS).index(j[0]) if j[0] in RUNS else 99)
jobs.sort(key=job_key)
print(len(jobs), 'jobs', flush=True)

# 3a. Local copy of the CT slices the meshes touch (uncompressed 128^3 chunks; PHerc0826: ~2 GB per 128 slices)
import urllib.request, urllib.parse, xml.etree.ElementTree as ET, concurrent.futures as cf
def fetch(url, dest, tries=4):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                tmp = dest + '.part'
                with open(tmp, 'wb') as f:
                    shutil.copyfileobj(r, f, 1 << 22)
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
ZSCRIPT = ('import sys, json, tifffile, numpy as np\\nzs = []\\nfor m in sys.argv[1:]:\\n'
           '    z = tifffile.imread(m + "/z.tif"); z = z[np.isfinite(z) & (z > 0)]\\n'
           '    zs += [float(z.min()), float(z.max())] if z.size else []\\nprint(json.dumps([min(zs), max(zs)] if zs else None))')
ZR = None
if jobs:
    zr = subprocess.run([VPY, '-c', ZSCRIPT] + sorted({m for _, _, m in jobs}), env=INKENV, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    ZR = json.loads(zr.strip().splitlines()[-1])
print('mesh z range', ZR, flush=True)
VOLSRC = S3 + '/__VOLUME__/0'
if ZR:
    cz0, cz1 = max(0, int(ZR[0]) - 16) // 128, (int(ZR[1]) + 16) // 128
    VOL = WORK + '/volume_l0/0'
    fetch(VOLSRC + '/.zarray', VOL + '/.zarray')
    keys = []
    for cz in range(cz0, cz1 + 1):
        keys += list_keys(f'__VOLUME__/0/{cz}/')
    if os.environ.get('CT_KEY_FILTER'):  # dry runs only
        keys = [k for k in keys if re.search(os.environ['CT_KEY_FILTER'], k)]
    t = time.time()
    with cf.ThreadPoolExecutor(48) as ex:
        sizes = list(ex.map(lambda k: fetch(S3 + '/' + k, VOL + '/' + k[len('__VOLUME__/0/'):]), keys))
    log(f'CT slices {cz0 * 128}-{(cz1 + 1) * 128}: {len(keys)} chunks, {sum(sizes) / 2**30:.1f} GB in {(time.time() - t) / 60:.1f} min')
    VOLSRC = VOL

# 3a'. The windings of the runs listed in the spec's "align" are also rendered after tools/phase_align.py moves each
#      onto the sheet it runs along (CT profiles along the normals; a winding parallel to the sheets but between them
#      comes back onto a sheet, one that crosses sheets does not change): run "<run>_aligned"
ALIGN_RUNS, ALIGN_PERIOD = __ALIGN_RUNS__, __ALIGN_PERIOD__
if ALIGN_RUNS and jobs:
    def align_one(job):
        run, w, m = job
        out_m = f'{WORK}/aligned/{run}/{os.path.basename(m)}'
        o = subprocess.run([VPY, f'{TOOLS}/tools/phase_align.py', m, VOLSRC, out_m, '--period', str(ALIGN_PERIOD)],
                           env=INKENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        print('ALIGN', run, w, o.strip().splitlines()[-1][:400] if o.strip() else '', flush=True)
        return (run + '_aligned', w, out_m) if os.path.exists(out_m + '/x.tif') else None
    with cf.ThreadPoolExecutor(4) as ex:
        extra = [j for j in ex.map(align_one, [j for j in jobs if j[0] in ALIGN_RUNS]) if j]
    for run in ALIGN_RUNS:
        if run in RUNS:
            RUNS[run + '_aligned'] = RUNS[run]
    jobs = sorted(jobs + extra, key=job_key)
    log(len(extra), 'aligned windings added;', len(jobs), 'jobs')

# 3b. render + ink, one worker per GPU (one on CPU). With the scroll's umbilicus, each job reports its normal
#     orientation and the layer order chosen from it before inference (primary_direction; see inkjob.py)
# with "flatten" in the spec, every winding is flattened with villa's Lasagna flattener before it is rendered (the
# First Letters requirement of a low-distortion parametrisation); the job reports the grid distortion before/after
FLAT_ARGS = ['--flatten-lasagna', WORK + '/villa/lasagna'] if __FLATTEN__ else []
UMB_ARGS = []
if '__UMB_URL__':
    try:
        fetch('__UMB_URL__', WORK + '/umbilicus.json')
        UMB_ARGS = ['--umbilicus', WORK + '/umbilicus.json']
    except Exception as e:
        print('umbilicus not available:', e, flush=True)
RESULTS = []
lock = threading.Lock()
def worker(gpu, my_jobs):
    for run, w, m in my_jobs:
        if time.time() - T0 > DEADLINE_H * 3600:
            print('deadline: skipping', run, w, flush=True); continue
        tag = re.sub('[^A-Za-z0-9_]+', '_', run) + f'_w{w:03d}'
        try:
            r = run_inkjob(['job', '--mesh', m, '--volume', VOLSRC, '--prefix', f'{OUTW}/{tag}', '--work', f'{WORK}/job_{tag}',
                            '--ckpt', CKPT, '--png-scale', str(PNG_SCALE), '--cache', CACHE] + UMB_ARGS + FLAT_ARGS, gpu=gpu)
            r.update(run=run, winding=w, mesh=os.path.relpath(m, '/kaggle/input'), gpu=gpu)
        except Exception as e:
            r = {'run': run, 'winding': w, 'error': repr(e)[-800:]}
        with lock:
            RESULTS.append(r)
            log('JOB ' + json.dumps(r))
slots = GPUS or [None]
threads = [threading.Thread(target=worker, args=(g, jobs[i::len(slots)])) for i, g in enumerate(slots)]
[t.start() for t in threads]; [t.join() for t in threads]
""", RUNS=json.dumps(spec["runs"]), WINDINGS=json.dumps(spec["windings"]), PRIORITY=json.dumps(spec.get("priority", [])), VARIANT=spec.get("variant", "_spliced"),
     VOLUME=VOLUME, PNG_SCALE=spec.get("png_scale", 1.0), DEADLINE_H=spec.get("deadline_h", 10.5),
     FLATTEN=bool(spec.get("flatten")), ALIGN_RUNS=json.dumps(ALIGN.get("runs", [])), ALIGN_PERIOD=ALIGN.get("period_vox", 0),
     UMB_URL=("" if not spec.get("umbilicus") else spec["umbilicus"] if spec["umbilicus"].startswith("http")
              else f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{SCROLL}/representations/umbilicus/{spec['umbilicus']}"))

code("""
# 4. One overview per run and image kind (ink in both layer orders, the middle layer of the render, and the max of
#    layers 10-16): every winding's strip at 1/8 size, stacked by winding; summary
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
for run in RUNS:
    tagr = re.sub('[^A-Za-z0-9_]+', '_', run)
    for d in ['ink_forward', 'ink_reverse', 'layer13', 'max10-16']:
        strips = []
        for f in sorted(glob.glob(f'{OUTW}/{tagr}_w[0-9][0-9][0-9]_{d}.png')):
            im = Image.open(f).convert('L'); k = 0.125 / PNG_SCALE
            im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
            strips.append((re.search(r'_w(\\d{3})_', f).group(1), im))
        if not strips:
            continue
        W = max(im.width for _, im in strips) + 60
        H = sum(im.height + 6 for _, im in strips)
        ov = Image.new('L', (W, H), 255); dr = ImageDraw.Draw(ov); y = 0
        for w, im in strips:
            ov.paste(im, (60, y)); dr.text((4, y + im.height // 2 - 5), 'w' + w, fill=0); y += im.height + 6
        ov.save(f'{OUTW}/overview_{tagr}_{d}.jpg', quality=90)
    rr = [r for r in RESULTS if r.get('run') == run and 'error' not in r]
    if rr:
        prim = [r.get('primary_direction') for r in rr]
        cm = [r['contrast_median'] for r in rr if r.get('contrast_median') is not None]
        pd_ = max(set(prim), key=prim.count)
        p99 = [r.get(f'ink_{pd_}_p99') for r in rr if r.get(f'ink_{pd_}_p99') is not None]
        print(f"RUN {run}: {len(rr)} windings; render contrast median {np.median(cm) if cm else float('nan'):.3f}; "
              f"normal orientation -> primary layer order {pd_} ({prim.count(pd_)}/{len(prim)}); ink p99 median "
              f"{np.median(p99) if p99 else float('nan'):.1f} ({pd_})", flush=True)
# 5. Side by side at the render's resolution: the same winding in every run (the middle third along the winding;
#    runs whose meshes share a grid, e.g. a fit and its aligned copies, show the same place)
for w in sorted({j[1] for j in jobs}):
    for d in ('layer13', 'ink_forward', 'ink_reverse'):
        bands = []
        for run in RUNS:
            f = f"{OUTW}/{re.sub('[^A-Za-z0-9_]+', '_', run)}_w{w:03d}_{d}.png"
            if os.path.exists(f):
                im = Image.open(f).convert('L')
                x0 = im.width // 3
                bands.append((run, im.crop((x0, 0, x0 + min(im.width // 3, 1200), im.height))))
        if len(bands) < 2:
            continue
        W = max(b.width for _, b in bands); H = sum(b.height + 22 for _, b in bands)
        cv = Image.new('L', (W, H), 255); dr = ImageDraw.Draw(cv); y = 0
        for run, b in bands:
            dr.text((4, y + 4), f'{run}  w{w:03d}  {d}', fill=0); cv.paste(b, (0, y + 20)); y += b.height + 22
        cv.save(f'{OUTW}/compare_w{w:03d}_{d}.jpg', quality=92)
json.dump({'control': CONTROL, 'jobs': RESULTS}, open(f'{OUTW}/ink_summary.json', 'w'), indent=1)
print('SUMMARY ' + json.dumps({'control': CONTROL, 'jobs': RESULTS}))
print(f'total {(time.time() - T0) / 3600:.2f} h')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"))
print("wrote", sys.argv[2], len(cells), "cells;", sum(len(c["source"]) for c in cells), "chars")
if os.path.getsize(sys.argv[2]) > 59000:  # Kaggle's File -> Import Notebook fails silently above ~60 KB
    print("WARNING:", os.path.getsize(sys.argv[2]), "bytes: Kaggle's importer may reject it silently (keep it under ~59 KB,"
          " e.g. with \"audit\": \"none\" in the spec)")
