"""Print the README's block table (§4) from the data files, so that new rows (new audits) are added without hand
copying (dev tool). Usage: python block_table.py <data dir>"""
import json
import os
import sys

d = sys.argv[1]
pub = [json.loads(l) for l in open(os.path.join(d, "block_published_fits_and_control.jsonl"))][1:]
ours = [json.loads(l) for l in open(os.path.join(d, "block_our_fits_kaggle.jsonl"))]
nul = {}
for l in open(os.path.join(d, "sheet_hits_random_phase_null.jsonl")):
    r = json.loads(l)
    if "as_fitted" in r:
        nul[r["set"]] = r["as_fitted"]
NULL_OF = {"PHerc0139 curated windings (control)": "PHerc0139 curated windings w034-w041 (control)",
           "PHerc0139 curated windings moved by a smooth field uniform in +-7 voxels": "the same, moved by a smooth field uniform in +-7 voxels",
           "PHerc0139 curated windings moved 7 voxels (half a sheet)": "the same, moved 7 voxels (half a sheet)",
           "PHerc0211, armando-gaona published recipe fit (z 8000-9000)": "PHerc0211 z 8000-9000, armando-gaona published recipe fit, w062-w077",
           "PHerc0826, rodriguescarson published fit (z 8928-9728)": "PHerc0826 z 8928-9728, rodriguescarson published fit, w037-w052",
           "PHerc0826, rodriguescarson published fit (z 6528-7328)": "PHerc0826 z 6528-7328, rodriguescarson published fit, w037-w052",
           "PHerc0826, rodriguescarson published fit (z 9328-10128)": "PHerc0826 z 9328-10128, rodriguescarson published fit, w037-w052"}
LAB = {"PHerc0139 curated windings (control)": "PHerc0139, curated w034–w041 (control)",
       "PHerc0139 curated windings moved by a smooth field uniform in +-7 voxels": "the same, moved by the ±7-voxel field",
       "PHerc0139 curated windings moved 7 voxels (half a sheet)": "the same, moved half a sheet",
       "PHerc0211, armando-gaona published recipe fit (z 8000-9000)": "PHerc0211, armando-gaona's published recipe fit, w062–w077",
       "PHerc0826, rodriguescarson published fit (z 8928-9728)": "PHerc0826 z 8928–9728, rodriguescarson's published fit, w037–w052",
       "PHerc0826, rodriguescarson published fit (z 6528-7328)": "the same author's fit of PHerc0826 z 6528–7328, w037–w052",
       "PHerc0826, rodriguescarson published fit (z 9328-10128)": "the same author's fit of PHerc0826 z 9328–10128, w037–w052"}
OURS = {"A_recipe (30k steps)": "PHerc0826, our recipe fit (30,000 steps), w062–w077",
        "C2_clampcore_w100 (30k steps)": "PHerc0826, recipe + clamped grad_mag spacing prior, weight 100",
        "S_seed (2k steps)": "PHerc0826, recipe seed (2,000 steps)", "W12_winding_model": "PHerc0826, winding-model refit, weight 12",
        "W48_winding_model": "PHerc0826, winding-model refit, weight 48",
        "W100g2_phase48": "PHerc0826, second-generation store: weight 100 + phase term on the model's crossings",
        "W100g2_phase48_snap": "PHerc0826, the same, phase term on the snapped crossings",
        "S_seed_0211 (2k steps)": "PHerc0211, recipe seed (2,000 steps, clockwise), w062–w077",
        "W48_CW": "PHerc0211, winding-model refit, weight 48, clockwise",
        "W48_ACW": "PHerc0211, the same, anticlockwise",
        "W48_CW_snap": "PHerc0211, weight 48 + phase term on the snapped crossings, clockwise",
        "W48_ACW_snap": "PHerc0211, the same, anticlockwise"}


from decimal import ROUND_HALF_UP, Decimal


def r3(x):
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def r2(x):
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def pct(x):
    return str((Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + " %"


def nstr(n):
    return f"{n:,}"


def nulls(nr, n=200):
    if not nr:
        return "next audit"
    k = round(nr.get("null_frac_at_or_above", nr.get("frac_at_or_above_fit", 0)) * n)
    return f"{r3(nr.get('null_c_pos_mean', nr.get('frac_c_pos_mean')))} ({k}/{n})"


print("| block of windings | crossings | c > 0, as fitted → aligned | random-shift null, as fitted: mean (copies ≥ fit) | "
      "c > 0.1, as fitted | gaps under 3 voxels | surface in voids |")
print("|---|---|---|---|---|---|---|")
PUB_ORDER = ["PHerc0139 curated windings (control)", "PHerc0139 curated windings moved by a smooth field uniform in +-7 voxels",
             "PHerc0139 curated windings moved 7 voxels (half a sheet)", "PHerc0211, armando-gaona published recipe fit (z 8000-9000)",
             "PHerc0826, rodriguescarson published fit (z 8928-9728)", "PHerc0826, rodriguescarson published fit (z 6528-7328)",
             "PHerc0826, rodriguescarson published fit (z 9328-10128)"]
pub.sort(key=lambda r: PUB_ORDER.index(r["set"]) if r["set"] in PUB_ORDER else len(PUB_ORDER))
OURS_ORDER = list(OURS)
ours.sort(key=lambda r: OURS_ORDER.index(r["fit"]) if r["fit"] in OURS_ORDER else len(OURS_ORDER))
for r in pub:
    lab = LAB.get(r["set"]) or ("the same author's fit of PHerc0191 z 11600–12400, w037–w052" if "PHerc0191" in r["set"] else r["set"])
    key = NULL_OF.get(r["set"]) or ("PHerc0191 z 11600-12400, rodriguescarson published fit, w037-w052" if "PHerc0191" in r["set"] else None)
    a, b = r["as_fitted"], r["aligned_final"]
    print(f"| {lab} | {nstr(a['n'])} | {r3(a['c_pos'])} → {r3(b['c_pos'])} | {nulls(nul.get(key))} | {r2(a['c_gt_0.1'])} | "
          f"{pct(a['gaps_lt_3vox'])} → {pct(b['gaps_lt_3vox'])} | {pct(a['void'])} → {pct(b['void'])} |")
for r in ours:
    a, b = r["raw"], r["aligned"]
    print(f"| {OURS.get(r['fit'], r['fit'])} | {nstr(a['n'])} | {r3(a['frac_c_pos'])} → {r3(b['frac_c_pos'])} | {nulls(r.get('null_raw'))} | "
          f"{r2(a['frac_c_gt_0.1'])} | {pct(r['gaps_raw']['frac_lt_3vox'])} → {pct(r['gaps_aligned']['frac_lt_3vox'])} | "
          f"{pct(r['void_raw'])} → {pct(r['void_aligned'])} |")
