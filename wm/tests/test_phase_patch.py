"""phase_patch.py on a copy of villa's spiral-fitting: the phase term is 0 for crossings on the exported windings,
2 for crossings half-way between them, 1 at a quarter; its gradient moves the windings onto the crossings; with
WM_PHASE_WEIGHT unset villa's losses are unchanged; and the patched fit_spiral.py compiles.

Needs a villa checkout (VILLA env var, default /home/claude/vc/villa). Run: python -m pytest wm/tests/test_phase_patch.py
"""
import importlib
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VILLA = os.environ.get("VILLA", "/home/claude/vc/villa")
SF = os.path.join(VILLA, "spiral-fitting")
pytestmark = pytest.mark.skipif(not os.path.isdir(SF), reason="needs a villa checkout")


@pytest.fixture(scope="module")
def patched():
    d = tempfile.mkdtemp()
    for f in os.listdir(SF):
        if f.endswith(".py"):
            shutil.copy2(os.path.join(SF, f), d)
    out = subprocess.run([sys.executable, os.path.join(HERE, "..", "phase_patch.py"), d], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    again = subprocess.run([sys.executable, os.path.join(HERE, "..", "phase_patch.py"), d], capture_output=True, text=True)
    assert again.stdout.count("already patched") == 2  # idempotent
    py_compile.compile(os.path.join(d, "fit_spiral.py"), doraise=True)
    sys.path.insert(0, d)
    for m in ("winding_supervision", "sample_spiral", "loss_maps"):
        sys.modules.pop(m, None)
    ws = importlib.import_module("winding_supervision")
    yield ws
    sys.path.remove(d)
    for m in ("winding_supervision", "sample_spiral", "loss_maps"):
        sys.modules.pop(m, None)


DR = 16.0


def spiral_points(n, frac, rng):
    """Pairs of consecutive crossings of a ray at theta, on shifted radius (k + frac) * DR (spiral space = scroll)."""
    theta = rng.uniform(0.3, 2 * np.pi - 0.3, n)
    k = rng.integers(5, 40, n)
    pts = np.zeros((n, 2, 3))
    for j in range(2):
        r = DR * (k + j + frac + theta / (2 * np.pi))
        pts[:, j, 0] = rng.uniform(10, 90, n)  # z
        pts[:, j, 1] = r * np.sin(theta)  # y
        pts[:, j, 2] = r * np.cos(theta)  # x
    return torch.tensor(pts, dtype=torch.float32)


class FakeStore:
    def __init__(self, pts):
        self.pts = pts

    def sample_relative(self, count, a, b, generator=None):
        return {"points": self.pts[:0], "target": self.pts.new_zeros((0,))}

    def sample_adjacent(self, count, generator=None):
        return {"points": self.pts, "target": torch.ones(len(self.pts))}


class FakeShell:
    def lookup(self, p):
        n = p.shape[:-1]
        return torch.full(n, 1e9), torch.linalg.norm(p[..., 1:], dim=-1), torch.ones(n), torch.ones(n, dtype=torch.bool)


CFG = {"sample_count_winding_model_relative_pairs": 0, "sample_count_winding_model_density_pairs": 64,
       "loss_weight_dense_spacing": 0.0, "loss_weight_dense_spacing_density": 12.0,
       "winding_model_relative_pair_delta": [3, 15], "winding_model_huber_delta": 0.5}


def run(ws, pts, transform=lambda p: p):
    return ws.get_winding_inference_losses(transform, torch.tensor(DR), FakeStore(pts), FakeShell(), CFG, 0, 100)


@pytest.mark.parametrize("frac,expected", [(0.0, 0.0), (0.5, 2.0), (0.25, 1.0), (0.75, 1.0)])
def test_phase_values(patched, frac, expected, monkeypatch):
    monkeypatch.setenv("WM_PHASE_WEIGHT", "1")
    losses, metrics = run(patched, spiral_points(64, frac, np.random.default_rng(0)))
    assert losses["dense_spacing_winding_model_phase"].item() == pytest.approx(expected, abs=1e-3)
    assert metrics["dense_spacing_winding_model_phase_residual_abs_p50"] == pytest.approx(min(frac, 1 - frac), abs=1e-3)
    # the density term (one winding between consecutive crossings) is satisfied whatever the phase
    assert losses["dense_spacing_winding_model_density"].item() == pytest.approx(0.0, abs=1e-3)


def test_unset_weight_keeps_villa(patched, monkeypatch):
    monkeypatch.delenv("WM_PHASE_WEIGHT", raising=False)
    losses, _ = run(patched, spiral_points(64, 0.3, np.random.default_rng(1)))
    assert set(losses) == {"dense_spacing_winding_model_relative", "dense_spacing_winding_model_density"}


def test_gradient_moves_windings_onto_crossings(patched, monkeypatch):
    """A transform that only adds a radial offset: gradient descent on the phase term finds the offset that puts
    the crossings (generated at frac 0.3) on integer shifted radii."""
    monkeypatch.setenv("WM_PHASE_WEIGHT", "1")
    pts = spiral_points(64, 0.3, np.random.default_rng(2))
    off = torch.zeros((), requires_grad=True)

    def transform(p):
        r = torch.linalg.norm(p[..., 1:], dim=-1, keepdim=True)
        return torch.cat([p[..., :1], p[..., 1:] * (1 + off / r)], dim=-1)  # radius + off

    opt = torch.optim.SGD([off], lr=2.0)
    for _ in range(300):
        opt.zero_grad()
        losses, _ = run(patched, pts, transform)
        losses["dense_spacing_winding_model_phase"].backward()
        opt.step()
    # crossings at shifted radius (k + 0.3) DR move to an integer: off = -0.3 DR (or +0.7 DR)
    assert min(abs(off.item() + 0.3 * DR), abs(off.item() - 0.7 * DR)) < 0.05
    assert losses["dense_spacing_winding_model_phase"].item() < 1e-4
