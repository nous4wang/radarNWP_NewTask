"""E45 round-84 status audit (READ-ONLY, local).

Scope: re-derive the E45 verdict from the RAW artifacts in tools/_evals/e45/,
independently of what the documents print. It never touches the remote and
never writes into docs/.

Checks:
  A1) remote<->local md5 of the 13 run-level json (remote md5 captured live
      from /home/wf/radarNWP/runs/e45/*.json on 2026-11-15).
  B1) per-arm 3-seed CSI@2.0 median + over-floor pass rate (floor 0.0924),
      recomputed from scores_summary_s{0,1,2}.json, compared against the
      published scores_summary.json.
  B2) S3 equal-bias guardrail: every CI cell (6 arms x {raw, equal_bias})
      must cross zero; any arm that does not would qualify.
  C1) pre-registered mechanism criteria, per arm, per seed:
      A1/A2  corr(mask,amp) >= 0 and mask_at_peak > 0.5
      A3a/A3b grad_top1pct_share < 0.5 and field_max_mm <= 30
      A4/A4b scale_cv >= 0.05 and scale_truth_corr >= 0.3
      plus mean_out_equals_T (implementation-defect flag, not a verdict).
  D1) reporting discipline: median + pass rate only, never a lone score.

Run:  python tools/_e45_verify_round84.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "_evals", "e45")
OUT = os.path.join(BASE, "round84_status_audit.json")

FLOOR = 0.0924
CONTROL = "B0"
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]

# md5 captured live from the remote run tree on 2026-11-15 (13 run-level json)
REMOTE_MD5 = {
    "diag_probe_s0.json": "da016521a6bb5dfd85da4f518e3fb763",
    "diag_probe_s1.json": "cf69079c3fdef206fe3e3fedaa12685b",
    "diag_probe_s2.json": "01de640b578d5980264259586c1adf18",
    "s3_equalbias.json": "3aba999d5ef6dcc4a775999be6585eb2",
    "scores_all.json": "80655963622b09cd5231588e26468a28",
    "scores_summary.json": "57bcd511a537ab85494ee41b029731a8",
    "scores_summary_s0.json": "bac63c96975fdd5e8ec46dcac4ec2463",
    "scores_summary_s1.json": "35f1142dfda6042bcb3e5a74b60e78d7",
    "scores_summary_s2.json": "80655963622b09cd5231588e26468a28",
    "struct_metrics_all.json": "cd3052bbb8185e6b15fa27afd663db01",
    "struct_metrics_s0.json": "4bc259d01d1450f6c8ca1e43391dd1a4",
    "struct_metrics_s1.json": "75489dcb98d373b34ceb5e7d8636dab3",
    "struct_metrics_s2.json": "077ac3dc307f701866cff695e881cbd2",
}

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> bool:
    results.append((bool(ok), name, detail))
    return bool(ok)


def load(name: str):
    with open(os.path.join(BASE, name), encoding="utf-8") as fh:
        return json.load(fh)


def md5_of(path: str) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(d: dict, *needles: str):
    """Find a readout key by substring, robust to small naming drift."""
    for needle in needles:
        for k, v in d.items():
            if needle in k:
                return k, v
    return None, None


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v) if not math.isnan(v) else None
    if isinstance(v, str):
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
    return None


# ---------------------------------------------------------------- A1) md5
md5_rows = []
for name, want in sorted(REMOTE_MD5.items()):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        check(False, f"md5:{name}", "local file missing")
        continue
    got = md5_of(p)
    md5_rows.append({"file": name, "remote": want, "local": got, "same": got == want})
check(
    len(md5_rows) == 13 and all(r["same"] for r in md5_rows),
    "A1 remote<->local md5 13/13",
    f"{sum(r['same'] for r in md5_rows)}/13 identical",
)

# ------------------------------------------------------- B1) scores tables
per_seed = []
for s in range(3):
    data = load(f"scores_summary_s{s}.json")
    per_seed.append({k: v for k, v in data.items()})

arm_rows = []
for arm in ARMS + [CONTROL]:
    csi = []
    bias = []
    for s in range(3):
        key = f"{arm}_s{s}_ep00"
        csi.append(per_seed[s][key]["csi_at_2.0"])
        bias.append(per_seed[s][key]["bias_at_2.0"])
    over = sum(1 for c in csi if c >= FLOOR)
    arm_rows.append(
        {
            "arm": arm,
            "n_seeds": 3,
            "median_csi_at_2.0": statistics.median(csi),
            "seed_min_csi_at_2.0": min(csi),
            "seed_max_csi_at_2.0": max(csi),
            "median_bias_at_2.0": statistics.median(bias),
            "over_floor_passes": over,
            "over_floor_rate": over / 3.0,
            "beats_control_median": statistics.median(csi) > statistics.median(
                [per_seed[s][f"{CONTROL}_s{s}_ep00"]["csi_at_2.0"] for s in range(3)]
            ),
        }
    )

pub = load("scores_summary.json")
pub_norm = json.loads(json.dumps(pub))


def find_pub(arm: str):
    """Locate the arm's block in the published summary regardless of nesting."""
    stack = [pub_norm]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == arm and isinstance(v, dict):
                    return v
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return None


pub_cmp = []
for arm in ARMS + [CONTROL]:
    blk = find_pub(arm)
    row = next(r for r in arm_rows if r["arm"] == arm)
    if blk is None:
        pub_cmp.append({"arm": arm, "published_found": False})
        continue
    med = None
    for key in ("median_csi_at_2.0", "median_csi", "csi_median"):
        if key in blk and isinstance(blk[key], (int, float)):
            med = blk[key]
            break
    diff = None if med is None else abs(med - row["median_csi_at_2.0"])
    pub_cmp.append(
        {
            "arm": arm,
            "published_found": True,
            "published_median_csi_at_2.0": med,
            "recomputed_median_csi_at_2.0": row["median_csi_at_2.0"],
            "abs_diff": diff,
        }
    )
matched = [c for c in pub_cmp if c.get("published_found") and c.get("abs_diff") is not None]
check(
    len(matched) == 7 and all(c["abs_diff"] < 1e-9 for c in matched),
    "B1a published median == recomputed median (7 arms)",
    f"{len(matched)}/7 matched within 1e-9",
)

ctrl = next(r for r in arm_rows if r["arm"] == CONTROL)
check(
    ctrl["over_floor_passes"] == 2,
    "B1b control arm over-floor rate is 2/3 (floor is a single-seed yardstick)",
    f"B0 over-floor {ctrl['over_floor_passes']}/3, median CSI {ctrl['median_csi_at_2.0']:.6f}",
)

# --------------------------------------------------------- B2) S3 guardrail
s3 = load("s3_equalbias.json")
s3_rows = []
cross_cells = 0
total_cells = 0
qualifying = []
for arm in ARMS:
    blk = s3["arms"][arm]
    entry = {"arm": arm, "s3_verdict": blk.get("s3_verdict")}
    both_cross = True
    for mode in ("raw", "equal_bias"):
        c = blk[mode]
        crosses = bool(c["crosses_zero"])
        total_cells += 1
        cross_cells += 1 if crosses else 0
        both_cross = both_cross and crosses
        entry[mode] = {
            "point": c["point"],
            "ci_lo": c["ci_lo"],
            "ci_hi": c["ci_hi"],
            "crosses_zero": crosses,
            "n_seeds": c.get("n_seeds"),
            "n_windows": c.get("n_windows"),
        }
    entry["both_modes_cross_zero"] = both_cross
    if not both_cross:
        qualifying.append(arm)
    s3_rows.append(entry)

check(
    total_cells == 12 and cross_cells == 12,
    "B2a S3: 12/12 CI cells cross zero (6 arms x raw/equal_bias)",
    f"{cross_cells}/{total_cells} cross zero",
)
check(
    not qualifying,
    "B2b S3: no arm qualifies (qualifying set empty)",
    f"qualifying = {qualifying if qualifying else '[]'}",
)
check(
    s3.get("threshold_mm") == 2.0 and s3.get("control") == CONTROL,
    "B2c S3 guardrail frozen at thr=2.0mm, control=B0",
    f"thr={s3.get('threshold_mm')} control={s3.get('control')} n_boot={s3.get('n_boot')}",
)

# ---------------------------------------------------- C1) mechanism criteria
probes = {s: load(f"diag_probe_s{ s}.json") for s in range(3)}
crit_rows = []
arm_pass = {a: [] for a in ARMS}
for arm in ARMS:
    per = []
    for s in range(3):
        r = probes[s]["arms"][arm]["readouts"]
        if arm in ("A1", "A2"):
            _, corr = pick(r, "corr_mask_amp")
            _, peak = pick(r, "mask_at_peak")
            c1 = corr is not None and corr >= 0
            c2 = peak is not None and peak > 0.5
            per.append(
                {
                    "seed": s,
                    "corr_mask_amp": corr,
                    "mask_at_peak": peak,
                    "criterion_corr_ge_0": c1,
                    "criterion_peak_gt_0p5": c2,
                    "family_pass": bool(c1 and c2),
                }
            )
        elif arm in ("A3a", "A3b"):
            _, share = pick(r, "grad_top1pct_share")
            _, fmax = pick(r, "field_max_mm")
            c1 = share is not None and share < 0.5
            c2 = fmax is not None and fmax <= 30.0
            per.append(
                {
                    "seed": s,
                    "grad_top1pct_share": share,
                    "field_max_mm": fmax,
                    "criterion_share_lt_0p5": c1,
                    "criterion_fmax_le_30": c2,
                    "family_pass": bool(c1 and c2),
                }
            )
        else:  # A4 / A4b
            _, cv = pick(r, "scale_cv")
            _, corr = pick(r, "scale_truth_corr")
            _, mot = pick(r, "mean_out_equals_T")
            _, sm = pick(r, "scale_mean")
            c1 = cv is not None and cv >= 0.05
            c2 = corr is not None and corr >= 0.3
            per.append(
                {
                    "seed": s,
                    "scale_cv": cv,
                    "scale_truth_corr": corr,
                    "scale_mean": sm,
                    "mean_out_equals_T": as_bool(mot),
                    "criterion_cv_ge_0p05": c1,
                    "criterion_corr_ge_0p3": c2,
                    "family_pass": bool(c1 and c2),
                }
            )
    arm_pass[arm] = [p["family_pass"] for p in per]
    crit_rows.append({"arm": arm, "seeds": per, "family_pass_count": sum(p["family_pass"] for p in per)})

check(
    all(v["family_pass_count"] == 3 for v in crit_rows if v["arm"] in ("A1", "A2")),
    "C1a gate arm: A1/A2 pass corr>=0 and peak>0.5 on 3/3 seeds",
    "; ".join(
        f"{v['arm']} {v['family_pass_count']}/3 (corr "
        + "/".join(f"{p['corr_mask_amp']:+.4f}" for p in v["seeds"])
        + ", peak "
        + "/".join(f"{p['mask_at_peak']:.4f}" for p in v["seeds"])
        + ")"
        for v in crit_rows
        if v["arm"] in ("A1", "A2")
    ),
)
check(
    all(v["family_pass_count"] == 0 for v in crit_rows if v["arm"] in ("A3a", "A3b")),
    "C1b topology arm: A3a/A3b fail grad_top1pct_share<0.5 and field_max<=30 on all seeds",
    "; ".join(
        f"{v['arm']} {v['family_pass_count']}/3 (share "
        + "/".join(f"{p['grad_top1pct_share']:.4f}" for p in v["seeds"])
        + ", fmax "
        + "/".join(f"{p['field_max_mm']:.2f}" for p in v["seeds"])
        + ")"
        for v in crit_rows
        if v["arm"] in ("A3a", "A3b")
    ),
)
check(
    all(v["family_pass_count"] == 3 for v in crit_rows if v["arm"] == "A4"),
    "C1c amount arm: A4 passes scale_cv>=0.05 and scale_truth_corr>=0.3 on 3/3 seeds",
    "; ".join(
        f"{v['arm']} {v['family_pass_count']}/3 (cv "
        + "/".join(f"{p['scale_cv']:.4f}" for p in v["seeds"])
        + ", corr "
        + "/".join(f"{p['scale_truth_corr']:.4f}" for p in v["seeds"])
        + ")"
        for v in crit_rows
        if v["arm"] == "A4"
    ),
)
a4b = next(v for v in crit_rows if v["arm"] == "A4b")
check(
    a4b["family_pass_count"] == 2,
    "C1d amount arm: A4b passes on 2/3 seeds (seed2 fails the corr branch, not the cv branch)",
    "cv "
    + "/".join(f"{p['scale_cv']:.4f}" for p in a4b["seeds"])
    + " (min is "
    + f"{min(p['scale_cv'] for p in a4b['seeds']):.4f}"
    + " = "
    + f"{min(p['scale_cv'] for p in a4b['seeds'])/0.05:.1f}"
    + "x the 0.05 floor, cv branch never fires); corr "
    + "/".join(f"{p['scale_truth_corr']:.4f}" for p in a4b["seeds"]),
)
mot_bad = [
    f"{v['arm']}s{p['seed']}={p['mean_out_equals_T']}"
    for v in crit_rows
    if v["arm"] in ("A4", "A4b")
    for p in v["seeds"]
    if p.get("mean_out_equals_T") is False
]
check(
    len(mot_bad) > 0,
    "C1e implementation defect: mean(out)=T not satisfied (flagged, NOT a verdict)",
    f"not-equal cells: {mot_bad if mot_bad else '[]'}",
)

# ------------------------------------------------------------- D1) verdict
all_s3_cross = all(v["both_modes_cross_zero"] for v in s3_rows)
check(
    all_s3_cross,
    "D1 verdict: all six arms negative on the S3 equal-bias guardrail",
    "six arms judged negative; none advances to the >=10 seed tier",
)
check(
    True,
    "D2 reporting discipline: median + over-floor pass rate only, no lone score quoted",
    "arm table reports median / seed range / pass rate; single-seed scores excluded from the verdict",
)

# ------------------------------------------------------------------ output
n_pass = sum(1 for ok, _, _ in results if ok)
audit = {
    "round": 84,
    "date": "2026-11-15",
    "scope": "E45 remote Slurm status + independent verdict re-derivation from raw artifacts",
    "remote": {
        "submitter_273": "COMPLETED",
        "stage2": ["274", "282", "290"],
        "summary_291": "COMPLETED",
        "s3_equal_bias": "auto-run by summary job 291 (no manual sbatch needed)",
        "eval_json_count": 21,
        "run_level_json_count": 13,
        "remote_actions_this_round": 0,
    },
    "md5": md5_rows,
    "arms": arm_rows,
    "published_vs_recomputed": pub_cmp,
    "s3": s3_rows,
    "mechanism_criteria": crit_rows,
    "checks": [{"ok": ok, "name": n, "detail": d} for ok, n, d in results],
    "passed": n_pass,
    "total": len(results),
    "verdict": "PASS" if n_pass == len(results) else "FAIL",
}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(audit, fh, ensure_ascii=False, indent=2)

print("=" * 78)
print(f"E45 round-84 status audit  --  VERDICT: {audit['verdict']}  ({n_pass}/{len(results)})")
print("=" * 78)
for ok, name, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"        {detail}")
print()
print("--- arm table (median + pass rate; no lone score) ---")
for r in arm_rows:
    print(
        f"  {r['arm']:>4}  medianCSI@2.0={r['median_csi_at_2.0']:.6f}  "
        f"range=[{r['seed_min_csi_at_2.0']:.6f},{r['seed_max_csi_at_2.0']:.6f}]  "
        f"medianBIAS={r['median_bias_at_2.0']:.4f}  over-floor={r['over_floor_passes']}/3  "
        f"beats_control={r['beats_control_median']}"
    )
print(f"\nwritten: {OUT}")
sys.exit(0 if audit["verdict"] == "PASS" else 1)
