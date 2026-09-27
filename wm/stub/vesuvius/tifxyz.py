"""Minimal tifxyz reader for the winding-model seeding code (read_tifxyz -> _x, _y, _z, valid_vertex_mask).

Same validity rule as vesuvius.tifxyz: mask.tif when present (same shape), else z > 0; invalid vertices
are set to -1. Avoids the full package's cv2/numba imports on Kaggle's Python 3.14 environment.
"""
import os
from types import SimpleNamespace

import numpy as np
from PIL import Image


def read_tifxyz(path, **_):
    path = str(path)
    x, y, z = (np.array(Image.open(os.path.join(path, c + ".tif")), dtype=np.float32) for c in "xyz")
    mask = None
    if os.path.exists(os.path.join(path, "mask.tif")):
        m = np.array(Image.open(os.path.join(path, "mask.tif")))
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape == z.shape:
            mask = m != 0
    if mask is None:
        mask = (z > 0) & np.isfinite(z)
    for a in (x, y, z):
        a[~mask] = -1.0
    return SimpleNamespace(_x=x, _y=y, _z=z, valid_vertex_mask=mask, path=path)
