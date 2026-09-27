"""Figure: the model-free test on a block of 16 windings, as fitted and after phase alignment, for the curated control,
every published spiral fit of an eligible scroll we audited, and our own fits (dev tool; data files of the repo).

Usage: python scoreboard_figure.py <data dir> <out.png>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

data, out = sys.argv[1], sys.argv[2]
pub = [json.loads(line) for line in open(os.path.join(data, "block_published_fits_and_control.jsonl"))][1:]
ours = [json.loads(line) for line in open(os.path.join(data, "block_our_fits_kaggle.jsonl"))]
NULL = {}  # set -> as-fitted random-phase null (mean, 2.5 %, 97.5 %), where measured
if os.path.exists(os.path.join(data, "sheet_hits_random_phase_null.jsonl")):
    for line in open(os.path.join(data, "sheet_hits_random_phase_null.jsonl")):
        r = json.loads(line)
        if "as_fitted" in r:
            NULL[r["set"]] = (r["as_fitted"]["null_c_pos_mean"], *r["as_fitted"]["null_c_pos_95"])
NULL_OF = {  # scoreboard rows -> null rows
    "PHerc0139 curated windings (control)": "PHerc0139 curated windings w034-w041 (control)",
    "PHerc0139 curated windings moved by a smooth field uniform in +-7 voxels": "the same, moved by a smooth field uniform in +-7 voxels",
    "PHerc0139 curated windings moved 7 voxels (half a sheet)": "the same, moved 7 voxels (half a sheet)",
    "PHerc0211, armando-gaona published recipe fit (z 8000-9000)": "PHerc0211 z 8000-9000, armando-gaona published recipe fit, w062-w077",
    "PHerc0826, rodriguescarson published fit (z 8928-9728)": "PHerc0826 z 8928-9728, rodriguescarson published fit, w037-w052",
    "PHerc0826, rodriguescarson published fit (z 6528-7328)": "PHerc0826 z 6528-7328, rodriguescarson published fit, w037-w052",
    "PHerc0826, rodriguescarson published fit (z 9328-10128)": "PHerc0826 z 9328-10128, rodriguescarson published fit, w037-w052",
}
LABELS = {
    "PHerc0139 curated windings (control)": "PHerc0139 curated windings",
    "PHerc0139 curated windings moved by a smooth field uniform in +-7 voxels": "  the same, moved ±7 voxels (smooth)",
    "PHerc0139 curated windings moved 7 voxels (half a sheet)": "  the same, moved half a sheet",
    "PHerc0211, armando-gaona published recipe fit (z 8000-9000)": "PHerc0211 armando-gaona",
    "PHerc0826, rodriguescarson published fit (z 8928-9728)": "PHerc0826 rodriguescarson z 8928",
    "PHerc0826, rodriguescarson published fit (z 6528-7328)": "PHerc0826 rodriguescarson z 6528",
    "PHerc0826, rodriguescarson published fit (z 9328-10128)": "PHerc0826 rodriguescarson z 9328",
}
rows = []  # (group, label, as fitted, aligned, n as fitted, n aligned, null or None)
for r in pub:
    lab = LABELS.get(r["set"]) or ("PHerc0191 rodriguescarson z 11600" if "PHerc0191" in r["set"] else r["set"])
    key = NULL_OF.get(r["set"]) or ("PHerc0191 z 11600-12400, rodriguescarson published fit, w037-w052" if "PHerc0191" in r["set"] else None)
    rows.append(("control" if "PHerc0139" in r["set"] else "published", lab, r["as_fitted"]["c_pos"],
                 r["aligned_final"]["c_pos"], r["as_fitted"]["n"], r["aligned_final"]["n"], NULL.get(key)))
OURS = {"A_recipe (30k steps)": "PHerc0826 recipe, 30k steps", "C2_clampcore_w100 (30k steps)": "PHerc0826 recipe + spacing prior",
        "S_seed (2k steps)": "PHerc0826 recipe, 2k steps", "W12_winding_model": "PHerc0826 winding model, w 12",
        "W48_winding_model": "PHerc0826 winding model, w 48", "W48_CW": "PHerc0211 winding model, CW",
        "W48_ACW": "PHerc0211 winding model, ACW", "S_seed_0211 (2k steps)": "PHerc0211 recipe, 2k steps",
        "W100g2_phase48": "PHerc0826 w 100 + phase (model crossings)",
        "W100g2_phase48_snap": "PHerc0826 w 100 + phase (snapped crossings)",
        "W48_CW_snap": "PHerc0211 CW + phase (snapped)", "W48_ACW_snap": "PHerc0211 ACW + phase (snapped)"}
for r in ours:
    nr = r.get("null_raw")  # the random-shift null of the fit as fitted, when the audit ran it
    nul = (nr["frac_c_pos_mean"], nr["frac_c_pos_p2.5"], nr["frac_c_pos_p97.5"]) if nr else None
    rows.append(("ours", OURS.get(r["fit"], r["fit"]), r["raw"]["frac_c_pos"], r["aligned"]["frac_c_pos"], r["raw"]["n"],
                 r["aligned"]["n"], nul))
P4 = os.path.join(data, "sheet_hits_paris4_team_fit_and_staff.jsonl")
if os.path.exists(P4):  # PHerc Paris 4 at level 2 (9.6 um): full bands, as fitted only (no aligned run)
    for line in open(P4):
        r = json.loads(line)
        lab = {"staff_z9000": "PHerc Paris 4 staff windings (z 9000, full band)",
               "paris4_team_fit_w028-058_z9000": "PHerc Paris 4 team spiral fit (z 9000, full band)"}.get(r.get("label"))
        if lab:
            n = r["null_random_phase"]
            rows.append(("reference", lab, r["fit"]["frac_c_pos"], None, r["fit"]["n"], None,
                         (n["frac_c_pos_mean"], n["frac_c_pos_p2.5"], n["frac_c_pos_p97.5"])))
order = {"control": 0, "reference": 1, "published": 2, "ours": 3}
OURS_ORDER = ["PHerc0826 recipe, 2k steps", "PHerc0826 recipe, 30k steps", "PHerc0826 recipe + spacing prior",
              "PHerc0826 winding model, w 12", "PHerc0826 winding model, w 48", "PHerc0826 w 100 + phase (model crossings)",
              "PHerc0826 w 100 + phase (snapped crossings)", "PHerc0211 recipe, 2k steps", "PHerc0211 winding model, CW",
              "PHerc0211 winding model, ACW", "PHerc0211 CW + phase (snapped)", "PHerc0211 ACW + phase (snapped)"]
rows.sort(key=lambda x: (order[x[0]], OURS_ORDER.index(x[1]) if x[1] in OURS_ORDER else 99))

INK, MUTED, GRID = "#1f2328", "#57606a", "#d0d7de"
COL = {"control": "#2f6f4e", "reference": "#6f42c1", "published": "#8a4b0f", "ours": "#1f5f99"}
fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(rows) + 1.3), dpi=150)
y = list(range(len(rows)))[::-1]
for yy, (g, lab, a, b, na, nb, nul) in zip(y, rows):
    if nul:  # the same windings at a random phase on the same rays: mean and 95 % range of 200 copies
        ax.plot([nul[1], nul[2]], [yy - 0.3, yy - 0.3], color=MUTED, lw=1.2, solid_capstyle="butt", zorder=1)
        ax.plot([nul[0]], [yy - 0.3], marker="|", color=MUTED, ms=6, mew=1.2, zorder=1)
    for v, n in ((a, na), (b, nb)):  # 95 % binomial interval (a lower bound: crossings on one ray are correlated)
        if v is None:
            continue
        h = 1.96 * (v * (1 - v) / n) ** 0.5
        ax.plot([v - h, v + h], [yy, yy], color=COL[g], alpha=0.28, lw=5, solid_capstyle="butt", zorder=2)
    if b is not None:
        ax.plot([a, b], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.scatter([a], [yy], s=42, color=COL[g], zorder=3, edgecolor="white", linewidth=1.2)
    if b is not None:
        ax.scatter([b], [yy], s=42, facecolor="white", edgecolor=COL[g], linewidth=1.6, zorder=3)
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
ax.text(0.504, len(rows) - 0.55, "0.5", color=MUTED, fontsize=8, ha="left", va="bottom")
ax.set_yticks(y)
ax.set_yticklabels([r[1] for r in rows], fontsize=8.5, color=INK)
for t, r in zip(ax.get_yticklabels(), rows):
    t.set_color(COL[r[0]])
ax.set_xlim(0.2, 0.86)
ax.set_ylim(-0.7, len(rows) - 0.1)
ax.set_xlabel("crossings where the CT is brighter than half-way to the neighbours (fraction)", fontsize=8.5, color=INK)
ax.tick_params(axis="x", labelsize=8, colors=MUTED)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6)
ax.scatter([], [], s=42, color=MUTED, label="as fitted")
ax.scatter([], [], s=42, facecolor="white", edgecolor=MUTED, linewidth=1.6, label="after tools/phase_align.py")
ax.plot([], [], color=MUTED, lw=1.2, label="random phase (95 % of 200 copies)")
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.045), ncol=3, fontsize=7.5, frameon=False)  # below the axis
ax.set_title("Model-free test on 16 consecutive windings (8 for the control)\ngreen: curated control; purple: PHerc Paris 4, "
             "full bands (4,208 and 801 crossings)\nbrown: published fits of eligible scrolls; blue: ours\nbands: 95 % "
             "binomial intervals (lower bounds: crossings on one ray are correlated)",
             fontsize=9, color=INK, loc="left")
fig.tight_layout()
fig.savefig(out)
print("wrote", out, len(rows), "rows")
