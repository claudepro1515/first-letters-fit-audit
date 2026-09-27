"""Dry run of a generated Kaggle notebook on CPU: executes its code cells in order in one namespace.

Setup (see make_ink_notebook.py): /kaggle/{temp,working,input/<fake>/<run>/meshes/wNNN} with a small synthetic mesh,
symlinks /kaggle/temp/villa and /kaggle/temp/villa_ink to local checkouts, the checkpoint copied, fake nvidia-smi and
pip on PATH, RENDER_DEVICE=cpu.
"""
import json, sys, os, time
nb = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/home/claude/vc/kaggle/ink0826.ipynb'))
ns = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    t = time.time()
    print(f'===== cell {i}', flush=True)
    exec(compile(c['source'], f'cell{i}', 'exec'), ns)
    print(f'===== cell {i} done in {time.time() - t:.0f} s', flush=True)
