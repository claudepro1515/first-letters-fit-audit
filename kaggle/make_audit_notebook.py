"""Build a CPU-only Kaggle notebook that audits spiral fits attached as inputs (no GPU quota used).

Every attached directory named 'meshes' that holds wNNN tifxyz windings is audited with the same tools as
the fit notebooks: winding_agreement.py against a winding-inference crossing store (the first attached
directory named winding_inference_heldout, else winding_inference_train), winding pitch (overall and per
sector), radius per winding, render contrast (three windings x three blocks), CT cross-sections, and
sheet_hits.py (model-free: is the CT brighter where the fit's windings cross a ray than half-way between them?
on radial rays from the umbilicus and on the store's rays, plus the phase of the model's crossings relative to
the fit).

Usage: python make_audit_notebook.py spec.json out.ipynb
  spec: {"scroll", "umbilicus", "volume", "z": [z0, z1], "sheet_um", "title", "notes",
         optional "sheet_hits": {"radial", "r0", "r1", "store_rays"},
         optional "extra_meshes": [{"repo": git URL, "path": "meshes", "label": name}] (published fits, cloned)}
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
SCROLL = spec["scroll"]
UMB = f"{SCROLL}/representations/umbilicus/{spec['umbilicus']}"
# an umbilicus given as a full URL (e.g. a community file on GitHub) is fetched from there instead of the S3 bucket
UMB_URL = spec["umbilicus"] if spec["umbilicus"].startswith("http") else "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/" + UMB
VOLUME = f"{SCROLL}/volumes/{spec['volume']}"
Z0, Z1 = spec["z"]
TOOLS_DIR = spec.get("tools_dir") or _find(os.path.join(HERE, "..", "tools"), "/home/claude/vc/release/tools")
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip()})


def code(t):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip()})


md(f"# {spec['title']}\n\nSettings: **no accelerator**, **Internet on**.\n\n{spec.get('notes', '')}")
TOOL_FILES = ["vcz.py", "mesh_roughness.py", "winding_pitch.py", "render_contrast.py", "winding_agreement.py", "sheet_hits.py",
              "fit_audit.py", "phase_align.py", "surface_void.py"]
ALIGN = spec.get("align", {})  # {"windings": 16, "period_vox": 16.3}: phase-align a block of windings and re-score it
LAS_COS = (f"https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/{SCROLL}/representations/predictions/lasagna/"
           f"{spec['lasagna']}/{SCROLL}_cos.ome.zarr") if spec.get("lasagna") else ""
SH = spec.get("sheet_hits", {"radial": 80, "r0": 300, "r1": 2200, "store_rays": 600})
tools_src = {f: open(os.path.join(TOOLS_DIR, f)).read() for f in TOOL_FILES}


def packed(var, files, what):
    """A code cell defining var = {path: source}, stored xz-compressed (Kaggle's importer silently drops notebooks
    above ~60 KB)."""
    import base64
    import hashlib
    import lzma
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode(), preset=9 | lzma.PRESET_EXTREME)).decode()
    lines = [f"# {what}, embedded as xz-compressed JSON {{path: source}} (sha256 of each file below)"]
    lines += [f"#   {p}  {hashlib.sha256(t.encode()).hexdigest()[:16]}" for p, t in files.items()]
    lines += ["import base64 as _b64, json as _json, lzma as _lzma",
              f"{var} = _json.loads(_lzma.decompress(_b64.b85decode("]
    lines += [f"    {blob[i:i + 120]!r}" for i in range(0, len(blob), 120)]
    lines += [")).decode())"]
    return "\n".join(lines)


cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": packed("TOOLS_SRC", tools_src, "Audit tools (first-letters-scan-atlas)")})
code("""
import os, subprocess, sys, time, json, shutil, glob, urllib.request
import numpy as np
T0 = time.time()
def sh(cmd, check=True, tail=3000):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-tail:], flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')
    return p.stdout
sh('pip install -q tifffile numcodecs')
WORK = '/kaggle/temp'; os.makedirs(WORK, exist_ok=True)
TOOLS = WORK + '/atlas/tools'; os.makedirs(TOOLS, exist_ok=True)
for fname, src in TOOLS_SRC.items():
    open(os.path.join(TOOLS, fname), 'w').write(src)
sys.path.insert(0, TOOLS)
import tifffile
from vcz import OmeZarr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S3 = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
UMBF = WORK + '/umbilicus.json'
urllib.request.urlretrieve('__UMB_URL__', UMBF)
cp = sorted(json.load(open(UMBF))['control_points'], key=lambda q: q['z'])
uz = np.array([q['z'] for q in cp], float); uy_ = np.array([q['y'] for q in cp], float); ux_ = np.array([q['x'] for q in cp], float)
ZMID = (__Z0__ + __Z1__) // 2
L1 = OmeZarr(S3 + '/__VOLUME__', cache_dir=WORK + '/l1cache').level('1')
stores = {os.path.basename(p): p for p in sorted(glob.glob('/kaggle/input/**/winding_inference_*', recursive=True))
          if os.path.exists(p + '/manifest.json')}
print('stores', stores)
HELD = stores.get('winding_inference_heldout') or stores.get('winding_inference_train')
SH_RADIAL, SH_R0, SH_R1, SH_STORE = __SH_RADIAL__, __SH_R0__, __SH_R1__, __SH_STORE__
TRAIN = stores.get('winding_inference_train')
LAS_COS = '__LAS_COS__'  # Lasagna cos prediction (sheet crests = 1), for the sheets-slid-per-turn check
ALIGN_N, ALIGN_PERIOD = __ALIGN_N__, __ALIGN_PERIOD__
ORDER = __ORDER__  # fits whose label ends with these names are audited first, in this order
AGREE_TRAIN = __AGREE_TRAIN__  # agreement on the training store too (slow: ~1 s per ray)
""".replace("__ALIGN_N__", str(ALIGN.get("windings", 0))).replace("__ORDER__", repr(spec.get("order", []))).replace("__AGREE_TRAIN__", repr(bool(spec.get("agreement_train", True)))).replace("__ALIGN_PERIOD__", str(ALIGN.get("period_vox", 0)))
     .replace("__LAS_COS__", LAS_COS).replace("__UMB__", UMB).replace("__UMB_URL__", UMB_URL).replace("__Z0__", str(Z0)).replace("__Z1__", str(Z1)).replace("__VOLUME__", VOLUME)
     .replace("__SH_RADIAL__", str(SH["radial"])).replace("__SH_R0__", str(SH["r0"])).replace("__SH_R1__", str(SH["r1"]))
     .replace("__SH_STORE__", str(SH["store_rays"])))
code("""
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


def block(name, g):
    # a block of ALIGN_N consecutive windings around the middle, scored as they are and after tools/phase_align.py
    # moves each onto the sheet it runs along (same radial rays): windings parallel to the sheets gain, crossing ones do
    # not. Gaps near 0 between consecutive crossings afterwards = two windings moved onto one sheet
    ws = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(g + '/w[0-9][0-9][0-9]') if os.path.exists(d + '/x.tif'))
    if not ws or not ALIGN_N:
        return {}
    mid = len(ws) // 2
    sel = ws[max(0, mid - ALIGN_N // 2): max(0, mid - ALIGN_N // 2) + ALIGN_N]
    res = {'block_windings': [sel[0], sel[-1]]}
    sub, al = f'{WORK}/align/{name}/raw', f'{WORK}/align/{name}/aligned'
    procs = []
    for k in range(4):  # four processes, one CPU each; the CT chunk cache is shared
        part = f'{WORK}/align/{name}/part{k}'
        os.makedirs(part, exist_ok=True)
        for w in sel[k::4]:
            for dst in (f'{sub}/w{w:03d}', f'{part}/w{w:03d}'):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.lexists(dst):
                    os.symlink(f'{g}/w{w:03d}', dst)
        procs.append(subprocess.Popen(f'python {TOOLS}/phase_align.py {part} {S3}/__VOLUME__ {al} --period {ALIGN_PERIOD} --cache {WORK}/ct0cache',
                                      shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
    outs = [p.communicate()[0] for p in procs]
    res['phase_align'] = [json.loads(l) for o in outs for l in o.splitlines() if l.startswith('{')]
    bad = [o[-600:] for o, p in zip(outs, procs) if p.returncode != 0]
    if bad:
        print('phase_align failed:', bad, flush=True)
    for tag, m_ in (('raw', sub), ('aligned', al)):
        out = sh(f'python {TOOLS}/sheet_hits.py {m_} {S3}/__VOLUME__ --z __Z0__ __Z1__ --radial {SH_RADIAL} --r0 {SH_R0} --r1 {SH_R1} --null 200 '
                 f'--umbilicus {UMBF} --cache {WORK}/ct0cache --label {name}_{tag}_w{sel[0]:03d}-w{sel[-1]:03d}', check=False, tail=1500)
        res['sheet_hits_block_' + tag] = next((json.loads(l) for l in out.splitlines() if l.startswith('{')), None)
        # share of the surface with no papyrus within 6 voxels along the normal (in the voids between sheet bundles)
        out = sh(f'python {TOOLS}/surface_void.py {m_} {S3}/__VOLUME__ --z __Z0__ __Z1__ --stride 2 --cache {WORK}/ct0cache '
                 f'--label {name}_{tag}', check=False, tail=600)
        res['void_block_' + tag] = next((json.loads(l).get('void_frac') for l in out.splitlines() if l.startswith('{')), None)
    if os.path.isdir(al):
        shutil.copytree(al, f'/kaggle/working/aligned/{name}', dirs_exist_ok=True)
    return res


def audit(name, g):
    ws = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(g + '/w[0-9][0-9][0-9]') if os.path.exists(d + '/x.tif'))
    res = {'mesh_dir': g, 'windings': len(ws)}
    if not ws:
        return res
    for tag, store in (('heldout', HELD), ('train', TRAIN if AGREE_TRAIN else None)):
        if store:
            out = sh(f'python {TOOLS}/winding_agreement.py {g} {store} --z __Z0__ __Z1__ --max-rays 4000 --umbilicus {UMBF}', check=False, tail=1500)
            res['agreement_' + tag] = next((json.loads(l) for l in out.splitlines() if l.startswith('{')), None)
    for tag, extra in (('radial', f'--radial {SH_RADIAL} --r0 {SH_R0} --r1 {SH_R1} --umbilicus {UMBF} --null 200'),
                       ('store', f'--store {HELD} --max-rays {SH_STORE} --umbilicus {UMBF}' if HELD else None)):
        if extra:
            out = sh(f'python {TOOLS}/sheet_hits.py {g} {S3}/__VOLUME__ --z __Z0__ __Z1__ {extra} --cache {WORK}/ct0cache --label {name}', check=False, tail=1500)
            res['sheet_hits_' + tag] = next((json.loads(l) for l in out.splitlines() if l.startswith('{')), None)
    starts = [w for w in ws if (w + 1) in ws]
    out = sh(f'python {TOOLS}/winding_pitch.py {g} --start {" ".join(map(str, starts))} --sheet-um __SHEET_UM__ --umbilicus {UMBF}', check=False, tail=1500)
    res['pitch'] = [json.loads(l) for l in out.splitlines() if l.startswith('{')]
    if LAS_COS:
        # sheets slid per turn (fit_audit.py, Lasagna cos crests along the normals, unwrapped around each row): ~0 for a
        # winding that stays on its sheet, about 2 per turn for a fit with the wrong spiral sense (it cannot follow the
        # sheets with an orientation-preserving warp); independent of the phase at which the winding sits
        res['lasagna_drift'] = []
        for w in sorted(set(ws[min(len(ws) - 1, int(len(ws) * q))] for q in (0.2, 0.4, 0.6, 0.8))):
            out = sh(f'python {TOOLS}/fit_audit.py {g}/w{w:03d} --lasagna-cos {LAS_COS} --level 1 --every 2 '
                     f'--cache {WORK}/lascache --out {WORK}/fit_audit_{name}_w{w:03d}.json', check=False, tail=1500)
            res['lasagna_drift'] += [json.loads(l) for l in out.splitlines() if l.startswith('{')]
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
        res.setdefault('xsection_png', [])
        L0 = OmeZarr(S3 + '/__VOLUME__', cache_dir=WORK + '/l1cache').level('0')
        rmid = 0.5 * (SH_R0 + SH_R1)  # the same three places for every fit
        for k, ang in enumerate((0.0, 120.0, 240.0)):
            cy, cx = cy0 + rmid * np.sin(np.radians(ang)), cx0 + rmid * np.cos(np.radians(ang))
            y0, x0 = int(cy) - 192, int(cx) - 192
            img = L0.read(ZMID, ZMID + 1, y0, y0 + 384, x0, x0 + 384)[0]
            fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
            ax.imshow(img, cmap='gray', extent=(x0, x0 + img.shape[1], y0 + img.shape[0], y0))
            for w, x_, y_ in traces:
                ax.plot(x_, y_, lw=1.2, color=cols[w % 10])
            ax.set_xlim(x0, x0 + img.shape[1]); ax.set_ylim(y0 + img.shape[0], y0)
            ax.set_title(f'__SCROLL__ z={ZMID} ({name}): full resolution, 384 x 384 voxels at {ang:.0f} deg, radius {rmid:.0f}')
            fn = f'{name}_xsection_l0_{k}_z{ZMID}.png'
            fig.savefig('/kaggle/working/' + fn, bbox_inches='tight'); plt.close(fig)
            res.setdefault('xsection_png', []).append(fn)
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
            fn = f'{name}_xsection_{tag}_z{ZMID}.png'
            fig.savefig('/kaggle/working/' + fn, bbox_inches='tight'); plt.close(fig)
            res['xsection_png'].append(fn)
    except Exception as e:
        print('cross-section failed:', e)
    return res

summary = {'scroll': '__SCROLL__', 'z': [__Z0__, __Z1__], 'held_store': HELD, 'train_store': TRAIN, 'runs': {}}
targets = []  # (label, mesh dir, source repo)
for m in sorted(glob.glob('/kaggle/input/**/meshes', recursive=True)):
    if glob.glob(m + '/w[0-9][0-9][0-9]/x.tif'):
        targets.append((os.path.relpath(os.path.dirname(m), '/kaggle/input').replace('/', '__'), m, None))
for x in __EXTRA_MESHES__:  # published fits of the same band in git repos
    d = WORK + '/extra/' + x['label']
    if not os.path.isdir(d):
        sh(f"git clone --depth 1 {x['repo']} {d}", check=False)
    m = d + '/' + x['path']
    if glob.glob(m + '/w[0-9][0-9][0-9]/x.tif'):
        targets.append((x['label'], m, x['repo']))
targets.sort(key=lambda t: next((i for i, k in enumerate(ORDER) if t[0].endswith(k)), len(ORDER)))
print('audit order', [t[0] for t in targets], flush=True)
BLOCKS = {}
for label, m, _ in targets:  # the quick decisive check first, for every fit
    b = BLOCKS[label] = block(label, m)
    if b:
        g_ = lambda t, k: (b.get('sheet_hits_block_' + t) or {}).get(k)
        print(f"BLOCK {label}: windings {b.get('block_windings')}; CT brighter at the crossings than half-way (radial rays) "
              f"{g_('raw', 'fit')} -> after phase_align {g_('aligned', 'fit')}; random-shift null {g_('raw', 'null_random_phase')} -> {g_('aligned', 'null_random_phase')}; "
              f"gaps {g_('raw', 'gaps_vox')} -> {g_('aligned', 'gaps_vox')}; surface in voids {b.get('void_block_raw')} -> {b.get('void_block_aligned')}; "
              f"{(time.time() - T0) / 60:.0f} min", flush=True)
for label, m, src in targets:
    r = audit(label, m)
    r.update(BLOCKS.get(label, {}))
    if src:
        r['source'] = src
    summary['runs'][label] = r
    json.dump(summary, open('/kaggle/working/summary.json', 'w'), indent=1)  # partial results survive a timeout
json.dump(summary, open('/kaggle/working/summary.json', 'w'), indent=1)
for n, r in summary['runs'].items():
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
          f"{np.median(cv) if cv else float('nan'):.3f} ({len(cv)} blocks); CT brighter at the fit's crossings than "
          f"half-way between them: radial rays {(r.get('sheet_hits_radial') or {}).get('fit')} (random-shift null "
          f"{(r.get('sheet_hits_radial') or {}).get('null_random_phase')}), store rays "
          f"{(r.get('sheet_hits_store') or {}).get('fit')}; model crossings {(r.get('sheet_hits_store') or {}).get('model')}; "
          f"phase model->fit {(r.get('sheet_hits_store') or {}).get('phase_model_to_fit')}; sheets slid per turn "
          f"(Lasagna cos, median over rows) {[(x['mesh'], x.get('drift_layers_median') if x.get('drift_layers_median') is None else round(x['drift_layers_median'], 2)) for x in r.get('lasagna_drift', [])]}; "
          f"block of windings before/after phase_align: {(r.get('sheet_hits_block_raw') or {}).get('fit')} -> "
          f"{(r.get('sheet_hits_block_aligned') or {}).get('fit')} (gaps {(r.get('sheet_hits_block_raw') or {}).get('gaps_vox')} -> "
          f"{(r.get('sheet_hits_block_aligned') or {}).get('gaps_vox')})")
print('SUMMARY ' + json.dumps(summary))
print(f'total {(time.time() - T0) / 3600:.2f} h')
""".replace("__Z0__", str(Z0)).replace("__Z1__", str(Z1)).replace("__VOLUME__", VOLUME)
     .replace("__SHEET_UM__", str(spec["sheet_um"])).replace("__SCROLL__", SCROLL)
     .replace("__EXTRA_MESHES__", repr(spec.get("extra_meshes", []))))
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], "w"), indent=1)
print("wrote", sys.argv[2], len(cells), "cells")
if os.path.getsize(sys.argv[2]) > 59000:  # Kaggle's File -> Import Notebook fails silently above ~60 KB
    print("WARNING:", os.path.getsize(sys.argv[2]), "bytes: Kaggle's importer may reject it silently (keep it under ~59 KB,"
          " e.g. with \"audit\": \"none\" in the spec)")
