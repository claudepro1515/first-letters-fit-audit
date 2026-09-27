"""Dry check of the phase-term wiring in a generated winding-model notebook (dev tool): runs the notebook's own
config/launch cell, its packed runner files and the patch lines of cell 4 against a copy of villa's spiral-fitting
sources, with a stand-in 'python' that records the environment each fit would get. No GPU, no data.

Usage: python check_phase_integration.py <notebook.ipynb> [villa dir]
"""
import json, os, shutil, subprocess, sys, tempfile

nb = json.load(open(sys.argv[1]))
villa = sys.argv[2] if len(sys.argv) > 2 else '/home/claude/vc/villa'
code = [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
cfg_cell = next(c for c in code if c.lstrip().startswith('# 3. Run configurations'))
wm_cell = next(c for c in code if 'WM_FILES = ' in c)
seed_cell = next(c for c in code if c.lstrip().startswith('# 4. Seed fit'))
patch_lines = seed_cell[seed_cell.index('if any(r.get('):seed_cell.index('SEED_FROM =')]

tmp = tempfile.mkdtemp()
WORK = tmp + '/work'
SF = WORK + '/villa/spiral-fitting'
os.makedirs(SF)
for f in os.listdir(villa + '/spiral-fitting'):
    if f.endswith('.py'):
        shutil.copy2(f'{villa}/spiral-fitting/{f}', SF)
subprocess.run('git init -q && git add -A && git -c user.email=x@y -c user.name=x commit -qm base', shell=True, cwd=SF, check=True)
ROOT = WORK + '/dataset'
os.makedirs(ROOT)
json.dump({'spiral_outward_sense': 'CW'}, open(ROOT + '/spiral-scroll.json', 'w'))
fake_py = tmp + '/fakepy.sh'
open(fake_py, 'w').write('#!/bin/sh\necho "WM_PHASE_WEIGHT=$WM_PHASE_WEIGHT"; echo "OVERRIDES=$FIT_SPIRAL_CONFIG_OVERRIDES"\n')
os.chmod(fake_py, 0o755)
WM = WORK + '/wm'


def sh(cmd, check=True, cwd=None, env=None, tail=6000):
    print('$', cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-tail:], flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f'command failed ({p.returncode}): {cmd}')
    return p.stdout


g = dict(os=os, sys=sys, json=json, subprocess=subprocess, shutil=shutil, time=__import__('time'), sh=sh, WORK=WORK,
         SF=SF, ROOT=ROOT, PY=fake_py, WM=WM, log=print)
exec(cfg_cell, g)
exec(wm_cell, g)
for rel, src in g['WM_FILES'].items():
    os.makedirs(os.path.dirname(os.path.join(WM, rel)), exist_ok=True)
    open(os.path.join(WM, rel), 'w').write(src)
exec(patch_lines, g)
ok = 'phase_patch' in open(SF + '/winding_supervision.py').read() and 'phase_patch' in open(SF + '/fit_spiral.py').read()
print('patched in place:', ok)
import py_compile
py_compile.compile(SF + '/fit_spiral.py', doraise=True)
py_compile.compile(SF + '/winding_supervision.py', doraise=True)
for n, r in g['RUNS'].items():
    p = g['launch'](n, r, 0)
    p.wait()
    out = open(f"{g['OUT']}/{n}.log").read()
    over = json.loads(out.split('OVERRIDES=', 1)[1])
    print(n, out.splitlines()[0], '| __phase_weight passed to the fitter:', '__phase_weight' in over,
          '| overrides.json:', json.load(open(f"{g['OUT']}/{n}/overrides.json")).get('wm_phase_weight'))
