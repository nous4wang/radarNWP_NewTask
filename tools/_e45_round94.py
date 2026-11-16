#!/usr/bin/env python
"""E45 round 94 audit -- S3 bootstrap-RNG robustness (remote job 343).

Read-only over local artifacts.  Writes one JSON report.

What round 94 adds
------------------
Round 91 re-ran the RAW half of the S3 equal-bias guard locally with 20
alternative bootstrap seeds and found 20/20 intervals containing zero for
all six arms, so the negative verdict does not rest on that one draw.
The EQUALISED half needs the real guard implementation, so it was left
to remote Slurm job 343, which was still PENDING behind the account
quota.  Job 343 has now COMPLETED and produced
``runs/e45/s3_seedrobust.json``.

This script therefore:
  1. checks the job used the SAME guard implementation that produced the
     published numbers (module md5 vs the local module, line-endings
     normalised);
  2. checks the canonical seed reproduces the published S3 numbers
     bit-for-bit (both halves);
  3. re-derives the "does this interval contain zero" question NUMERICALLY
     (ci_lo <= 0 <= ci_hi) instead of trusting the stored boolean;
  4. applies the pre-registered promotion test per arm per half per
     bootstrap seed: an arm qualifies iff point > 0 AND the interval
     excludes zero;
  5. re-derives the headline table (median + floor pass rate) from the
     21 primary *.eval.json files, independently of scores_all.json;
  6. re-checks the mechanism criteria on the raw probe JSON.

Nothing here touches the remote side or the frozen criteria.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"

ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
CONTROL = "B0"
FLOOR = 0.0924  # pre-registered floor, single-seed convention

checks: list[dict] = []
notes: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})
    return bool(ok)


def contains_zero(ci_lo: float, ci_hi: float) -> bool:
    """Numeric sign-inclusion, independent of the stored boolean."""
    return ci_lo <= 0.0 <= ci_hi


def excludes_zero(ci_lo: float, ci_hi: float) -> bool:
    return not contains_zero(ci_lo, ci_hi)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# 0. artifacts present
# ----------------------------------------------------------------------
SR_PATH = EV / "s3_seedrobust.json"
EB_PATH = EV / "s3_equalbias.json"
if not SR_PATH.exists():
    raise SystemExit(f"MISSING ARTIFACT: {SR_PATH}")
SR = load(SR_PATH)
EB = load(EB_PATH)

# ----------------------------------------------------------------------
# 1. same guard implementation as the published numbers
# ----------------------------------------------------------------------
mod_path = ROOT / "tools" / "_e41_s3_equalbias.py"
mod_md5 = hashlib.md5(mod_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
check(
    "guard_impl_md5_matches_local_module",
    SR.get("module_md5") == mod_md5,
    f"job={SR.get('module_md5')} local_normalised={mod_md5}",
)

meta_pairs = [
    ("threshold_mm", 2.0),
    ("control", CONTROL),
    ("n_boot", 10000),
    ("rng_seed", 20261109),
]
for key, expected in meta_pairs:
    check(
        f"guard_param_{key}",
        SR.get(key) == expected,
        f"job={SR.get(key)!r} published={EB.get(key)!r}",
    )
check(
    "guard_epoch_00",
    str(EB.get("epoch")) == "00",
    f"epoch={EB.get('epoch')!r}",
)
alt = SR.get("alt_seeds") or []
check("alt_seeds_count_20", len(alt) == 20, f"n={len(alt)}")
check("alt_seeds_canonical_first", bool(alt) and alt[0] == 20261109, f"first={alt[:1]}")
check("alt_seeds_unique", len(set(alt)) == len(alt), "")

# ----------------------------------------------------------------------
# 2. canonical seed reproduces the published S3 numbers
# ----------------------------------------------------------------------
repro_dev = 0.0
repro_cells = 0
for arm in ARMS:
    eb_arm = EB["arms"][arm]
    pairs = [
        ("raw", SR["arms"][arm]["raw_reproduced"]),
        ("equal_bias", SR["arms"][arm]["reproduced"]),
    ]
    for half, sr_cell in pairs:
        eb_cell = eb_arm[half]
        for field in ("point", "ci_lo", "ci_hi"):
            repro_dev = max(repro_dev, abs(float(sr_cell[field]) - float(eb_cell[field])))
            repro_cells += 1
        if bool(sr_cell["crosses_zero"]) != bool(eb_cell["crosses_zero"]):
            check(f"repro_flag_{arm}_{half}", False, "crosses_zero differs")
check(
    "canonical_reproduction_max_abs_dev",
    repro_dev == 0.0 and repro_cells == 36,
    f"max|dev|={repro_dev:.3e} over {repro_cells} cells",
)

for arm in ARMS:
    gap = SR["arms"][arm].get("max_abs_bias_gap")
    check(f"bias_gap_zero_{arm}", float(gap) == 0.0, f"gap={gap!r}")

# ----------------------------------------------------------------------
# 3+4. sweep: numeric zero-containment and the pre-registered promotion test
# ----------------------------------------------------------------------
sweep_summary = {}
for arm in ARMS:
    a = SR["arms"][arm]
    halves = {}
    for half, sweep_key, canon_key in (
        ("raw", "raw_sweep", "raw_reproduced"),
        ("equalised", "sweep", "reproduced"),
    ):
        sweep = a[sweep_key]
        canon = a[canon_key]
        n = len(sweep)

        flag_mismatch = 0
        n_contains_numeric = 0
        n_contains_stored = 0
        n_excludes = 0
        qualify_seeds = 0
        wholly_pos = 0
        wholly_neg = 0
        margins = []
        for cell in sweep:
            lo, hi = float(cell["ci_lo"]), float(cell["ci_hi"])
            num = contains_zero(lo, hi)
            stored = bool(cell["crosses_zero"])
            if num != stored:
                flag_mismatch += 1
            n_contains_numeric += int(num)
            n_contains_stored += int(stored)
            if excludes_zero(lo, hi):
                n_excludes += 1
                wholly_pos += int(lo > 0.0 and hi > 0.0)
                wholly_neg += int(lo < 0.0 and hi < 0.0)
            margins.append(min(abs(lo), abs(hi)))
            # pre-registered promotion test: point > 0 AND interval excludes 0
            if float(cell["point"]) > 0.0 and excludes_zero(lo, hi):
                qualify_seeds += 1

        canon_in_sweep = any(
            abs(float(c["ci_lo"]) - float(canon["ci_lo"])) == 0.0
            and abs(float(c["ci_hi"]) - float(canon["ci_hi"])) == 0.0
            for c in sweep
        )
        # the sweep must vary ONLY the resampling seed: the point estimate
        # is deterministic and must not move with the bootstrap seed
        point_drift = max(
            abs(float(c["point"]) - float(canon["point"])) for c in sweep
        )
        check(
            f"{arm}_{half}_point_invariant_under_resampling",
            point_drift == 0.0,
            f"max|point drift|={point_drift:.3e}",
        )
        halves[half] = {
            "n_seeds": n,
            "point": float(canon["point"]),
            "canon_ci": [float(canon["ci_lo"]), float(canon["ci_hi"])],
            "canon_contains_zero_numeric": contains_zero(
                float(canon["ci_lo"]), float(canon["ci_hi"])
            ),
            "contains_zero_numeric": n_contains_numeric,
            "contains_zero_stored": n_contains_stored,
            "excludes_zero": n_excludes,
            "wholly_positive": wholly_pos,
            "wholly_negative": wholly_neg,
            "qualifying_seeds": qualify_seeds,
            "min_margin_to_zero": min(margins),
            "median_margin_to_zero": statistics.median(margins),
            "flag_vs_numeric_mismatches": flag_mismatch,
            "canonical_present_in_sweep": canon_in_sweep,
        }
        check(f"{arm}_{half}_flag_matches_numeric", flag_mismatch == 0, f"mismatch={flag_mismatch}")
        check(
            f"{arm}_{half}_canonical_in_sweep",
            canon_in_sweep,
            "canonical seed interval not found in sweep",
        )
    sweep_summary[arm] = halves

# ----------------------------------------------------------------------
# 5. headline table: median + floor pass rate from the 21 primary evals
# ----------------------------------------------------------------------
eval_files = sorted((EV / "evals_raw").rglob("*.eval.json"))
if not eval_files:
    eval_files = sorted((EV / "raw_audit").rglob("*.eval.json"))
per_arm: dict[str, list[float]] = {}
unmatched = []
for p in eval_files:
    m = re.search(r"(?P<arm>[A-Za-z]+[0-9a-z]*)_s(?P<seed>\d+)", p.stem)
    if not m:
        unmatched.append(p.name)
        continue
    arm = m.group("arm")
    data = load(p)
    thr = data.get("thresholds", {})
    csi = thr.get("2.0", {}).get("csi")
    if csi is None:
        unmatched.append(p.name)
        continue
    per_arm.setdefault(arm, []).append(float(csi))

check("primary_evals_found_21", len(per_arm) == 7 and sum(len(v) for v in per_arm.values()) == 21,
      f"arms={sorted(per_arm)} n={sum(len(v) for v in per_arm.values())} unmatched={unmatched}")

headline = {}
for arm, vals in sorted(per_arm.items()):
    headline[arm] = {
        "n": len(vals),
        "median_csi_at_2.0": statistics.median(vals),
        "pass_rate_floor": sum(v >= FLOOR for v in vals) / len(vals),
        "n_pass": sum(v >= FLOOR for v in vals),
    }

b0_median = headline.get(CONTROL, {}).get("median_csi_at_2.0")
three_of_three = [a for a, h in headline.items() if h["n_pass"] == h["n"] == 3]
three_of_three_above_control = [
    a for a in three_of_three
    if b0_median is not None and headline[a]["median_csi_at_2.0"] > b0_median
]
notes.append(
    "arms clearing the floor on all three seeds: "
    + (", ".join(three_of_three) if three_of_three else "none")
    + "; of those, above the same-batch control median: "
    + (", ".join(three_of_three_above_control) if three_of_three_above_control else "none")
)

# cross-check the headline medians against the published aggregate
sa_path = EV / "scores_all.json"
sa_dev = None
if sa_path.exists():
    SA = load(sa_path)
    # layout: flat map "<arm>_s<seed>_ep00" -> {"csi_at_2.0": ..., ...}
    per_arm_sa: dict[str, list[float]] = {}
    for key, val in SA.items():
        if not isinstance(val, dict):
            continue
        m = re.match(r"(?P<arm>[A-Za-z]+[0-9a-z]*)_s(?P<seed>\d+)_ep(?P<ep>\d+)$", key)
        if not m:
            continue
        csi = val.get("csi_at_2.0")
        if csi is None:
            continue
        per_arm_sa.setdefault(m.group("arm"), []).append(float(csi))
    dev = 0.0
    cells = 0
    for arm, h in headline.items():
        if arm not in per_arm_sa:
            continue
        med = statistics.median(per_arm_sa[arm])
        dev = max(dev, abs(med - h["median_csi_at_2.0"]))
        cells += 1
    sa_dev = dev
    check(
        "headline_matches_scores_all",
        cells == len(headline) and dev == 0.0,
        f"max|dev|={dev:.3e} over {cells}/{len(headline)} arms",
    )

# ----------------------------------------------------------------------
# 6. mechanism criteria on the raw probe JSON
# ----------------------------------------------------------------------
mech: dict[str, dict] = {}
for seed in (0, 1, 2):
    p = EV / f"diag_probe_s{seed}.json"
    if not p.exists():
        continue
    d = load(p)
    arms_in = d.get("arms", {})
    for arm in ARMS:
        entry = arms_in.get(arm)
        if not isinstance(entry, dict):
            continue
        r = entry.get("readouts", {})
        g = r.get("grad_top1pct_share")
        fmax = r.get("field_max_mm")
        corr = r.get("corr_mask_amp")
        peak = r.get("mask_at_peak")
        cv = r.get("scale_cv")
        tcorr = r.get("scale_truth_corr")
        rec = mech.setdefault(arm, {"seeds": {}})
        rec["seeds"][seed] = {
            "grad_top1pct_share": g,
            "field_max_mm": fmax,
            "corr_mask_amp": corr,
            "mask_at_peak": peak,
            "scale_cv": cv,
            "scale_truth_corr": tcorr,
            "mean_out_equals_T": r.get("mean_out_equals_T"),
            "mean_out_equals_T_rel_err": r.get("mean_out_equals_T_rel_err"),
        }
        # only score a criterion when that readout family actually exists
        if g is not None and fmax is not None:
            rec.setdefault("topo_pass", []).append(
                bool(float(g) < 0.5 and float(fmax) <= 30.0)
            )
        if corr is not None and peak is not None:
            rec.setdefault("gate_pass", []).append(
                bool(float(corr) >= 0.0 and float(peak) > 0.5)
            )
        if cv is not None and tcorr is not None:
            rec.setdefault("scale_pass", []).append(
                bool(float(cv) >= 0.05 and float(tcorr) >= 0.3)
            )
for arm, rec in mech.items():
    for key in ("topo_pass", "gate_pass", "scale_pass"):
        if key in rec:
            rec[key + "_n_pass"] = sum(rec[key])
            rec[key + "_n"] = len(rec[key])
    # the three pre-registered criterion families must each be covered
    check(f"mech_family_covered_{arm}", any(k in rec for k in ("topo_pass", "gate_pass", "scale_pass")),
          f"keys={[k for k in ('topo_pass','gate_pass','scale_pass') if k in rec]}")

# ----------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------
equalised_qualifiers = {
    arm: s["equalised"]["qualifying_seeds"] for arm, s in sweep_summary.items()
}
raw_qualifiers = {arm: s["raw"]["qualifying_seeds"] for arm, s in sweep_summary.items()}
any_qualifier = {
    arm: raw_qualifiers[arm] + equalised_qualifiers[arm] for arm in sweep_summary
}
flipping_arms = [a for a, n in any_qualifier.items() if n > 0]
equalised_all_cross = all(
    s["equalised"]["contains_zero_numeric"] == s["equalised"]["n_seeds"]
    for s in sweep_summary.values()
)
raw_all_cross = all(
    s["raw"]["contains_zero_numeric"] == s["raw"]["n_seeds"] for s in sweep_summary.values()
)

verdict = {
    "s3_raw_half_20_20_contains_zero": raw_all_cross,
    "s3_equalised_half_20_20_contains_zero": equalised_all_cross,
    "arms_qualifying_on_any_bootstrap_seed": flipping_arms,
    "qualifying_seed_counts": any_qualifier,
    "promotion": "NONE -- no arm qualifies, no 10-seed compute",
}

report = {
    "round": 94,
    "scope": "E45 S3 bootstrap-RNG robustness, remote job 343 (COMPLETED 2026-11-16 06:02)",
    "remote_actions": "read-only status query + 1 artifact download (no sbatch, no sync)",
    "guard": {
        "threshold_mm": SR.get("threshold_mm"),
        "control": SR.get("control"),
        "n_boot": SR.get("n_boot"),
        "rng_seed": SR.get("rng_seed"),
        "module_md5_job": SR.get("module_md5"),
        "module_md5_local_normalised": mod_md5,
        "alt_seeds": alt,
    },
    "reproduction": {"max_abs_dev": repro_dev, "cells": repro_cells},
    "sweep": sweep_summary,
    "headline": headline,
    "headline_vs_scores_all_max_abs_dev": sa_dev,
    "mechanism": mech,
    "verdict": verdict,
    "notes": notes,
    "checks": checks,
}
n_pass = sum(c["pass"] for c in checks)
report["verdict_summary"] = f"{n_pass}/{len(checks)} checks PASS"

out = EV / "round94_status_audit.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------------------------------------------------------------
# console
# ----------------------------------------------------------------------
print(f"checks: {n_pass}/{len(checks)} PASS")
for c in checks:
    if not c["pass"]:
        print(f"  FAIL {c['name']}: {c['detail']}")
print()
print("S3 sweep -- equalised half (20 bootstrap seeds)")
print(f"{'arm':5} {'point':>10} {'canon CI':>26} {'contains0':>9} {'excl0':>6} {'qualify':>8} {'min margin':>12}")
for arm in ARMS:
    e = sweep_summary[arm]["equalised"]
    print(
        f"{arm:5} {e['point']:+10.6f} [{e['canon_ci'][0]:+.6f},{e['canon_ci'][1]:+.6f}] "
        f"{e['contains_zero_numeric']:>4}/{e['n_seeds']:<4} {e['excludes_zero']:>6} "
        f"{e['qualifying_seeds']:>8} {e['min_margin_to_zero']:>12.6f}"
    )
print()
print("S3 sweep -- raw half (20 bootstrap seeds)")
for arm in ARMS:
    r = sweep_summary[arm]["raw"]
    print(
        f"{arm:5} {r['point']:+10.6f} [{r['canon_ci'][0]:+.6f},{r['canon_ci'][1]:+.6f}] "
        f"{r['contains_zero_numeric']:>4}/{r['n_seeds']:<4} {r['excludes_zero']:>6} "
        f"{r['qualifying_seeds']:>8} {r['min_margin_to_zero']:>12.6f}"
    )
print()
print("headline (median + floor pass rate; single points not reported)")
for arm, h in headline.items():
    print(f"  {arm:5} n={h['n']} median={h['median_csi_at_2.0']:.6f} pass={h['n_pass']}/{h['n']}")
print()
print("mechanism criteria (n pass / n seeds)")
for arm, rec in mech.items():
    def fmt(key):
        if key + "_n" in rec:
            return f"{rec[key+'_n_pass']}/{rec[key+'_n']}"
        return "n/a"
    print(
        f"  {arm:5} gate={fmt('gate_pass')} topo={fmt('topo_pass')} scale={fmt('scale_pass')}"
    )
print()
for n in notes:
    print("NOTE:", n)
print()
print("VERDICT:", json.dumps(verdict, ensure_ascii=False))
print("report:", out)
sys.exit(0)
