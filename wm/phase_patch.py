"""Add a phase term to fit_spiral's winding-model loss (patch for a villa checkout; idempotent).

villa's winding-model supervision (spiral-fitting/winding_supervision.py) is relative: for two crossings of a ray
it asks the fit for the right NUMBER of windings between them, so it fixes the density of the windings but not
where they sit between the sheets. In the First Letters recipe the only term that pins the exported integer windings
is the track DT loss (weight 10, from step 25,000: each track is pulled to one integer winding); the track radius
loss pulls a track to its mean shifted radius, "continuous, not snapped to an integer winding", and patches with
absolute windings are off. Tracks end 11-14 % satisfied in our fits and in Miller & Mueller's. This patch adds

    phase loss = mean over crossings c of (1 - cos(2 pi s(c) / dr)),   s = shifted radius of c in spiral space,

which is zero when every crossing of the winding model lies on some exported winding (s / dr an integer) and
does not need to know which winding. It is evaluated on the crossings already sampled for the density pairs (both
ends of each pair), with the same validity mask, so it costs no extra transform evaluations.

Weight: environment variable WM_PHASE_WEIGHT (default 0 = villa's behaviour unchanged). Logged as the loss family
'dense_spacing_winding_model_phase' and the metrics dense_spacing_winding_model_phase_* (mean |s/dr - round(s/dr)|).

Usage: python phase_patch.py <villa>/spiral-fitting
"""
import sys
from pathlib import Path

MARK = "# phase_patch: winding-model phase term"

WS_OLD_1 = """    density = store.sample_adjacent(density_count, generator=generator)
"""
WS_NEW_1 = """    density = store.sample_adjacent(density_count, generator=generator)
    """ + MARK + """
    import os as _os
    phase_weight = float(_os.environ.get("WM_PHASE_WEIGHT", "0") or 0)
"""
WS_OLD_2 = """        if count:
            record_loss_samples(
                name, component_spiral.mean(dim=1), residual.detach().abs(), valid)
    return losses, metrics
"""
WS_NEW_2 = """        if count:
            record_loss_samples(
                name, component_spiral.mean(dim=1), residual.detach().abs(), valid)
        if name == "dense_spacing_winding_model_density" and phase_weight > 0:
            """ + MARK + """: every sampled density crossing should lie on an
            # exported winding, i.e. have an integer shifted radius / dr_per_winding.
            _theta, _radius, shifted = get_theta_and_radii(
                component_spiral[..., 1:], dr_per_winding)
            w = shifted / dr_per_winding
            point_valid = (
                (component_points[..., 0] >= float(z_begin))
                & (component_points[..., 0] < float(z_end))
                & torch.isfinite(component_points).all(dim=-1)
                & torch.isfinite(component_spiral).all(dim=-1)
                & torch.isfinite(w)
                & (shell_valid & (sample_radius <= shell_radius))[
                    cursor - count : cursor]
            )
            safe_w = torch.where(point_valid, w, torch.zeros_like(w))
            per_point = 1.0 - torch.cos(2.0 * np.pi * safe_w)
            pv = point_valid.to(per_point.dtype)
            losses["dense_spacing_winding_model_phase"] = (
                (per_point * pv).sum() / pv.sum().clamp(min=1))
            if with_metrics:
                frac = (w - torch.round(w)).abs()
                metrics.update(_metrics(
                    "dense_spacing_winding_model_phase", frac, point_valid))
    return losses, metrics
"""

FS_OLD = """                'dense_spacing_winding_model_density': (
                    inference_losses['dense_spacing_winding_model_density']
                    * self.config['loss_weight_dense_spacing_density']),
            })
"""
# The phase term joins the same backward_family call: the family's graph (one transform evaluation shared by all
# three terms) is released by its backward pass, so a second call could not reach it.
FS_NEW = """                'dense_spacing_winding_model_density': (
                    inference_losses['dense_spacing_winding_model_density']
                    * self.config['loss_weight_dense_spacing_density']),
                """ + MARK + """
                **({'dense_spacing_winding_model_phase': (
                    inference_losses['dense_spacing_winding_model_phase']
                    * float(__import__('os').environ.get('WM_PHASE_WEIGHT', '0') or 0))}
                   if 'dense_spacing_winding_model_phase' in inference_losses else {}),
            })
"""


def patch(path, pairs):
    src = path.read_text()
    if MARK in src:
        return "already patched"
    for old, new in pairs:
        if src.count(old) != 1:
            raise SystemExit(f"phase_patch: anchor not found exactly once in {path}:\n{old}")
        src = src.replace(old, new)
    path.write_text(src)
    return "patched"


def main(spiral_fitting):
    root = Path(spiral_fitting)
    print("winding_supervision.py:", patch(root / "winding_supervision.py", [(WS_OLD_1, WS_NEW_1), (WS_OLD_2, WS_NEW_2)]))
    print("fit_spiral.py:", patch(root / "fit_spiral.py", [(FS_OLD, FS_NEW)]))


if __name__ == "__main__":
    main(sys.argv[1])
