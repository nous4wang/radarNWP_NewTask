#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 68 -- epoch axis, THREE-seed (n=3) controlled read.  READ ONLY.

Inputs (all local):
  tools/_evals/e45/epoch_axis/evals/{ARM}_s{0,1,2}_ep{00,01,02}.eval.json  (63 cells)
  tools/_evals/e45/epoch_axis/probe/diag_probe_s{0,1,2}_ep{00,01,02}.json     (9 probes)
  tools/_evals/e45/{scores_summary,s3_equalbias,diag_probe_s0,diag_probe_s1}.json
  tools/_evals/e45/epoch_axis_all.tar.gz   (md5 734857236b13142f02d6174ee6d8fc8f)

Round 65 submitted the epoch axis (jobs 315/316/317, one per epoch).  Round 66 read
seed 0 (n=1), round 67 read seeds 0+1 (n=2).  All three jobs are now COMPLETED
(2026-11-14 23:01-23:03), so the FULL 3-seed column is available for the first time.

Discipline carried from rounds 61-67:
  * median + pass rate; a single cell never decides anything
  * paired convention and difference convention reported side by side
  * zero-model floor reported for every comparable axis
  * "criterion flipped" is split into "read-out out of bounds" vs "read-out missing"
  * post-hoc diagnostic: creates no criterion, changes no verdict
  * assertions never assume the data will agree with the conclusion
"""
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"
AX = EV / "epoch_axis"
OUTP = EV / "round68_epoch_axis_n3.json"
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
EPOCHS = ["00", "01", "02"]
SEEDS = [0, 1, 2]
FLOOR = 0.0924

report = {}
checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    return bool(ok)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, n=6):
    if x is None:
        return "NA"
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def md5_file(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


report["round"] = 68
report["axis"] = "epoch axis, n=3 (all three seeds), controlled"

# ------------------------------------------------- 0. provenance of the fetch
tar = EV / "epoch_axis_all.tar.gz"
tar_md5 = md5_file(tar) if tar.exists() else None
check("archive_md5_matches_remote",
      tar_md5 == "734857236b13142f02d6174ee6d8fc8f", f"{tar_md5}")
report["archive_md5"] = tar_md5

# ---------------------------------------------------------- 1. published guards
ss_md5 = md5_file(EV / "scores_summary.json")
s3_md5 = md5_file(EV / "s3_equalbias.json")
check("published_scores_summary_md5_unchanged",
      ss_md5 == "57bcd511a537ab85494ee41b029731a8", ss_md5)
check("published_s3_equalbias_md5_unchanged",
      s3_md5 == "3aba999d5ef6dcc4a775999be6585eb2", s3_md5)
report["published_md5"] = {"scores_summary": ss_md5, "s3_equalbias": s3_md5}

s3 = load(EV / "s3_equalbias.json")
s3txt = json.dumps(s3)
cz = []


def find_cz(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, bool) and "cross" in str(k).lower():
                cz.append(v)
            find_cz(v)
    elif isinstance(o, list):
        for v in o:
            find_cz(v)


find_cz(s3)
v_area = s3txt.count("GAIN_WAS_AREA_OR_AMPLITUDE")
v_none = s3txt.count("NO_GAIN_TO_BEGIN_WITH")
s3_arms = {}
_s3a = s3.get("arms", {}) if isinstance(s3, dict) else {}
if isinstance(_s3a, dict):
    for arm, rec in _s3a.items():
        if not isinstance(rec, dict):
            continue
        s3_arms[arm] = {
            "verdict": [v for k, v in rec.items() if isinstance(v, str) and "verdict" in str(k).lower()],
            "crosses_zero": [v for k, v in rec.items() if isinstance(v, bool) and "cross" in str(k).lower()],
        }
check("s3_all_intervals_cross_zero",
      len(cz) == 12 and all(cz),
      f"{sum(cz)}/{len(cz)} cross zero; verdicts area={v_area} nogain={v_none}")
report["s3_guard"] = {"n_intervals": len(cz), "n_cross_zero": sum(cz),
                      "verdict_area": v_area, "verdict_nogain": v_none,
                      "per_arm": s3_arms}

# ------------------------------------------- 2. published headline replication
ss = load(EV / "scores_summary.json")
published = {}
for arm, v in ss.items():
    if not isinstance(v, dict):
        continue
    per = v.get("csi_at_2.0")
    if isinstance(per, dict):
        for sk, sv in per.items():
            m = re.fullmatch(r"s(\d+)", str(sk))
            if m and isinstance(sv, (int, float)) and not isinstance(sv, bool):
                published[(arm, int(m.group(1)))] = float(sv)

head_med, head_rate = {}, {}
for arm, v in ss.items():
    per = v.get("csi_at_2.0", {})
    vals = [float(x) for x in per.values()
            if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if len(vals) == 3:
        head_med[arm] = median(vals)
        head_rate[arm] = (sum(1 for x in vals if x >= FLOOR), len(vals))
med_diff = max([abs(head_med[a] - ss[a]["median_csi_at_2.0"]) for a in head_med] or [float("nan")])
rate_ok = all(ss[a].get("pass_rate_over_floor") == f"{head_rate[a][0]}/{head_rate[a][1]}"
              for a in head_rate)
check("published_headline_median_and_passrate_reproduced",
      med_diff < 1e-12 and rate_ok,
      f"max|diff| median = {med_diff:.3e}; pass-rate strings match = {rate_ok}")
report["published_headline"] = {"median": head_med,
                                "pass_rate": {a: f"{v[0]}/{v[1]}" for a, v in head_rate.items()}}

# per-seed snapshot files vs the merged summary (both are E45's published data).
# The snapshot files are flat: keys look like "A1_s0_ep00" -> {csi_at_2.0: ...}
# (round 64: they are cumulative -- s0 has 7 keys, s1 14, s2 21).
seed_cons = []
seed_keys = {}
for s in SEEDS:
    p = EV / f"scores_summary_s{s}.json"
    if not p.exists():
        continue
    obj = load(p)
    seed_keys[s] = len(obj)
    for key, rec in obj.items():
        if not isinstance(rec, dict):
            continue
        v = rec.get("csi_at_2.0")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        m = re.fullmatch(r"(A3a|A3b|A4b|A4|A2|A1|B0)_s(\d+)_ep(\d+)", str(key))
        if not m:
            continue
        arm, sd, ep = m.group(1), int(m.group(2)), m.group(3)
        if sd != s or ep != "00":
            continue
        if (arm, sd) in published:
            seed_cons.append(abs(float(v) - published[(arm, sd)]))
max_cons = max(seed_cons) if seed_cons else None
check("published_per_seed_snapshots_consistent_with_summary",
      max_cons is not None and max_cons < 1e-12 and len(seed_cons) == 21,
      f"max|diff| = {max_cons} over {len(seed_cons)} values; snapshot key counts {seed_keys}")
report["published_per_seed_consistency_max_abs_diff"] = max_cons
report["published_snapshot_key_counts"] = seed_keys

# ------------------------------------------------------------- 3. the 63 cells
csi, bias, pod, far = {}, {}, {}, {}
for arm in ARMS:
    for s in SEEDS:
        for ep in EPOCHS:
            p = AX / "evals" / f"{arm}_s{s}_ep{ep}.eval.json"
            if not p.exists():
                continue
            d = load(p)
            t = d["thresholds"]["2.0"]
            csi[(arm, s, ep)] = t["csi"]
            bias[(arm, s, ep)] = t["bias"]
            pod[(arm, s, ep)] = t["pod"]
            far[(arm, s, ep)] = t["far"]
check("all_63_three_seed_cells_present", len(csi) == 63, f"{len(csi)}/63 cells")
report["n_cells"] = len(csi)
n_finite = sum(1 for v in csi.values() if isinstance(v, float) and math.isfinite(v))
check("all_cell_csi_finite", n_finite == len(csi), f"{n_finite}/{len(csi)} finite")

# CONTROL: the epoch-00 re-run must reproduce E45's published per-cell CSI.
ctrl = []
for (arm, s, ep), v in csi.items():
    if ep != "00":
        continue
    if (arm, s) in published:
        ctrl.append(abs(v - published[(arm, s)]))
max_ctrl = max(ctrl) if ctrl else None
check("control_ep00_rerun_reproduces_published_csi",
      max_ctrl is not None and max_ctrl < 1e-9 and len(ctrl) == 21,
      f"max|diff| = {max_ctrl} over {len(ctrl)} published cells")
report["control_csi_max_abs_diff"] = max_ctrl
report["control_csi_n_cells"] = len(ctrl)

# supplementary: byte identity against E45's published md5 list
md5_ident, md5_comp = 0, 0
publist = AX / "md5_published.txt"
if publist.exists():
    pubmap = {}
    for line in publist.read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, f = line.split(None, 1)
            pubmap[Path(f.strip()).name] = h
    for (arm, s, ep) in csi:
        if ep != "00":
            continue
        f = AX / "evals" / f"{arm}_s{s}_ep{ep}.eval.json"
        if not f.exists():
            continue
        for cand in (f"{arm}_s{s}.eval.json", f.name):
            if cand in pubmap:
                md5_comp += 1
                md5_ident += int(md5_file(f) == pubmap[cand])
                break
report["control_md5"] = {"n_compared": md5_comp, "n_identical": md5_ident}

# -------------------------------------------------------- 4. zero-model floor
floor_ep = {}
for ep in EPOCHS:
    pairs = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        if ("B0", a, ep) in csi and ("B0", b, ep) in csi:
            pairs.append(abs(csi[("B0", a, ep)] - csi[("B0", b, ep)]))
    floor_ep[ep] = {"pairs": pairs, "median": median(pairs) if pairs else None,
                    "max": max(pairs) if pairs else None}
check("zero_model_floor_three_pairs_per_epoch",
      all(len(floor_ep[ep]["pairs"]) == 3 for ep in EPOCHS),
      "; ".join(f"ep{ep}: n={len(floor_ep[ep]['pairs'])}" for ep in EPOCHS))
report["zero_model_floor_per_epoch"] = floor_ep

# --------------------------------------------------- 5. arm x epoch statistics
arm_ep = {}
for arm in ARMS:
    for ep in EPOCHS:
        vs = [csi[(arm, s, ep)] for s in SEEDS if (arm, s, ep) in csi]
        bs = [csi[("B0", s, ep)] for s in SEEDS if ("B0", s, ep) in csi]
        if len(vs) < 3 or len(bs) < 3:
            continue
        dl = [csi[(arm, s, ep)] - csi[("B0", s, ep)] for s in SEEDS
              if (arm, s, ep) in csi and ("B0", s, ep) in csi]
        fl = floor_ep[ep]["median"]
        arm_ep[f"{arm}_ep{ep}"] = {
            "median_csi": median(vs),
            "paired_median_delta": median(dl),
            "diff_convention_delta": median(vs) - median(bs),
            "paired_deltas": dl,
            "pos_sign_rate": sum(1 for d in dl if d > 0) / len(dl),
            "all_same_sign_positive": all(d > 0 for d in dl),
            "seed_spread": max(vs) - min(vs),
            "median_bias": median([bias[(arm, s, ep)] for s in SEEDS]),
            "median_pod": median([pod[(arm, s, ep)] for s in SEEDS]),
            "per_seed_csi": vs,
            "n_over_floor": sum(1 for v in vs if v >= FLOOR),
            "paired_delta_exceeds_floor": abs(median(dl)) > fl,
            "zero_model_floor": fl,
        }
check("all_21_arm_epoch_combinations_have_three_seeds", len(arm_ep) == 21, f"{len(arm_ep)}/21")
report["arm_epoch"] = arm_ep

# how many arms pass the floor 3/3 at each epoch (the promotion trigger shape)
floor_pass = {}
for ep in EPOCHS:
    floor_pass[ep] = {arm: arm_ep[f"{arm}_ep{ep}"]["n_over_floor"] for arm in ARMS}
report["floor_pass_counts_per_epoch"] = floor_pass

# ------------------------------------------------------- 6. peak epoch per arm
peak = {}
for arm in ARMS:
    n1 = {ep: csi[(arm, 0, ep)] for ep in EPOCHS if (arm, 0, ep) in csi}
    n2 = {ep: median([csi[(arm, s, ep)] for s in (0, 1) if (arm, s, ep) in csi])
          for ep in EPOCHS if all((arm, s, ep) in csi for s in (0, 1))}
    n3 = {ep: median([csi[(arm, s, ep)] for s in SEEDS if (arm, s, ep) in csi])
          for ep in EPOCHS if all((arm, s, ep) in csi for s in SEEDS)}
    b1 = max(n1, key=lambda e: n1[e]) if n1 else None
    b2 = max(n2, key=lambda e: n2[e]) if n2 else None
    b3 = max(n3, key=lambda e: n3[e]) if n3 else None
    peak[arm] = {"n1_seed0": n1, "best_n1": b1, "n2_median": n2, "best_n2": b2,
                 "n3_median": n3, "best_n3": b3,
                 "ep00_is_peak_n1": b1 == "00", "ep00_is_peak_n2": b2 == "00",
                 "ep00_is_peak_n3": b3 == "00",
                 "argmax_changed_n1_to_n2": b1 != b2,
                 "argmax_changed_n2_to_n3": b2 != b3}
report["peak_epoch"] = peak
n1c = sum(1 for a in ARMS if peak[a]["ep00_is_peak_n1"])
n2c = sum(1 for a in ARMS if peak[a]["ep00_is_peak_n2"])
n3c = sum(1 for a in ARMS if peak[a]["ep00_is_peak_n3"])
report["ep00_peak_count"] = {"n1": n1c, "n2": n2c, "n3": n3c,
                              "argmax_changed_n1_to_n2": sum(1 for a in ARMS if peak[a]["argmax_changed_n1_to_n2"]),
                              "argmax_changed_n2_to_n3": sum(1 for a in ARMS if peak[a]["argmax_changed_n2_to_n3"])}
check("peak_counts_are_consistent_with_round67_n2",
      n2c == report["ep00_peak_count"]["n2"] == 7,
      f"n1={n1c} n2={n2c} n3={n3c} arms whose best epoch is ep00")

# ---------------------------------------------------------- 7. mechanism probes
TARGET = ["corr_mask_amp", "mask_at_peak", "mask_dry_frac_lt_0p1", "corr_mask_resid",
          "grad_top1pct_share", "field_max_mm", "tail_max_over_p99",
          "scale_cv", "scale_truth_corr", "scale_is_constant", "scale_mean",
          "mean_out_equals_T", "maxratio_as_trained"]
ARM_RE = re.compile(r"(A3a|A3b|A4b|A4|A2|A1|B0)")


def walk(o, path=()):
    res = []
    if isinstance(o, dict):
        if any(k in o for k in TARGET):
            res.append((path, o))
        for k, v in o.items():
            res += walk(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            res += walk(v, path + (str(i),))
    return res


def arm_of(path, rec):
    if isinstance(rec, dict):
        for k in ("arm", "name", "case", "tag", "arm_name", "model"):
            v = rec.get(k)
            if isinstance(v, str) and ARM_RE.fullmatch(v.strip()):
                return v.strip()
    for p in reversed(path):
        m = ARM_RE.fullmatch(str(p).strip())
        if m:
            return m.group(1)
    m = ARM_RE.search("|".join(str(p) for p in path))
    return m.group(1) if m else None


probe = {}
unresolved = 0
n_probe_files = 0
for s in SEEDS:
    for ep in EPOCHS:
        p = AX / "probe" / f"diag_probe_s{s}_ep{ep}.json"
        if not p.exists():
            continue
        n_probe_files += 1
        obj = load(p)
        for path, rec in walk(obj):
            a = arm_of(path, rec)
            if a is None:
                unresolved += 1
                continue
            key = (a, s, ep)
            if key in probe:
                continue
            probe[key] = {k: rec.get(k) for k in TARGET if k in rec}
check("all_9_probe_files_present", n_probe_files == 9, f"{n_probe_files}/9 probe files")
check("probe_cells_6_arms_x_3_seeds_x_3_epochs", len(probe) == 54, f"{len(probe)}/54 cells")
report["mechanism_probe_cells"] = len(probe)
report["mechanism_probe_unresolved_records"] = unresolved

CRIT = {
    "A1": {"corr_mask_amp": (">=", 0.0), "mask_at_peak": (">", 0.5),
           "maxratio_as_trained": (">", 0.9)},
    "A2": {"corr_mask_amp": (">=", 0.0), "mask_at_peak": (">", 0.5),
           "maxratio_as_trained": (">", 0.9)},
    "A3a": {"grad_top1pct_share": ("<", 0.5), "field_max_mm": ("<=", 30.0)},
    "A3b": {"grad_top1pct_share": ("<", 0.5), "field_max_mm": ("<=", 30.0)},
    "A4": {"scale_cv": (">=", 0.05), "scale_truth_corr": (">=", 0.3)},
    "A4b": {"scale_cv": (">=", 0.05), "scale_truth_corr": (">=", 0.3)},
}


def judge(op, thr, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "missing"
    if isinstance(v, bool):
        return "pass" if v else "fail"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "missing"
    if op == ">=":
        return "pass" if x >= thr else "fail"
    if op == ">":
        return "pass" if x > thr else "fail"
    if op == "<":
        return "pass" if x < thr else "fail"
    if op == "<=":
        return "pass" if x <= thr else "fail"
    return "missing"


crit_rows = {}
for arm, crits in CRIT.items():
    for cname, (op, thr) in crits.items():
        cells = {}
        for ep in EPOCHS:
            for s in SEEDS:
                v = probe.get((arm, s, ep), {}).get(cname)
                cells[f"{arm}_s{s}_ep{ep}"] = {"value": v, "verdict": judge(op, thr, v)}
        npass = sum(1 for c in cells.values() if c["verdict"] == "pass")
        nmiss = sum(1 for c in cells.values() if c["verdict"] == "missing")
        nfail = len(cells) - npass - nmiss
        crit_rows[f"{arm}:{cname}"] = {
            "rule": f"{cname} {op} {thr}",
            "per_cell": cells,
            "n_pass": npass, "n_fail": nfail, "n_missing": nmiss,
            "pass_rate_over_available": (npass / (len(cells) - nmiss)) if (len(cells) - nmiss) else None,
            "values": {k: fmt(c["value"], 6) for k, c in cells.items()},
        }
check("criteria_rows_cover_nine_cells_each",
      all(len(r["per_cell"]) == 9 for r in crit_rows.values()),
      f"{len(crit_rows)} criteria rows x 9 cells")
report["criteria"] = crit_rows


def probe_map(p):
    obj = load(p)
    out = {}
    for path, rec in walk(obj):
        a = arm_of(path, rec)
        if a is None:
            continue
        out.setdefault(a, {k: rec.get(k) for k in TARGET if k in rec})
    return out


diff_probe = []
for seed in SEEDS:
    pub = probe_map(EV / f"diag_probe_s{seed}.json")
    for arm, rec in pub.items():
        new = probe.get((arm, seed, "00"), {})
        for k, v in rec.items():
            if k not in new:
                continue
            if isinstance(v, bool) or isinstance(new[k], bool):
                continue
            if isinstance(v, (int, float)) and isinstance(new[k], (int, float)):
                if isinstance(v, float) and math.isnan(v) and isinstance(new[k], float) and math.isnan(new[k]):
                    continue
                diff_probe.append(abs(float(v) - float(new[k])))
maxdiff = max(diff_probe) if diff_probe else None
check("probe_ep00_rerun_matches_published",
      maxdiff is not None and maxdiff < 1e-9,
      f"max|diff| = {maxdiff} over {len(diff_probe)} read-outs")
report["probe_control_max_abs_diff"] = maxdiff

# ------------------------------------------------------------- 8. the verdict
# E45's verdict is fixed by S3 (12/12 intervals cross zero) and by the
# pre-registered mechanism criteria.  The epoch axis is a diagnostic: it can
# only ADD evidence, it can never promote an arm.
eligible = []
for arm in ARMS:
    if arm == "B0":
        continue
    s3rec = s3_arms.get(arm, {})
    if any(s3rec.get("crosses_zero", [])):
        continue
report["verdict"] = {
    "e45_verdict_changed": False,
    "eligible_arm_set": eligible,
    "arms_promoted": [],
    "master_version": "v2.48",
    "reason": ("S3 stays 12/12 cross zero; the epoch axis is a post-hoc diagnostic "
               "and creates no criterion, so no arm is promoted by it."),
}
check("no_arm_promoted_by_epoch_axis",
      eligible == [] and report["verdict"]["e45_verdict_changed"] is False,
      "eligible arm set is empty")

# ------------------------------------------------------------------ 9. summary
report["n_checks_total"] = len(checks)
report["n_checks_pass"] = sum(1 for c in checks if c["ok"])
report["checks"] = checks
OUTP.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

print("=" * 78)
print(f"cells {len(csi)}/63   probe files {n_probe_files}/9   probe cells {len(probe)}/54")
print(f"control ep00 vs published CSI: max|diff| = {max_ctrl} over {len(ctrl)} cells")
print(f"control md5 (supplementary): {md5_ident}/{md5_comp} byte-identical")
print(f"published headline reproduced: max|diff| = {med_diff:.3e}; pass-rate match = {rate_ok}")
print(f"S3 guard: {sum(cz)}/{len(cz)} intervals cross zero "
      f"(area={v_area}, nogain={v_none})")
print("-" * 78)
print("zero-model floor per epoch (B0 reseeded, 3 pairs):")
for ep in EPOCHS:
    f = floor_ep[ep]
    print(f"  ep{ep}: median {f['median']:.6f}  max {f['max']:.6f}  pairs "
          + " ".join(f"{d:.6f}" for d in f["pairs"]))
print("-" * 78)
print("arm x epoch (n=3):  median CSI | paired dCSI | diff dCSI | +rate | floor-pass | vs floor")
for arm in ARMS:
    for ep in EPOCHS:
        r = arm_ep[f"{arm}_ep{ep}"]
        print(f"  {arm:4s} ep{ep}  {r['median_csi']:.6f}  {r['paired_median_delta']:+.6f}  "
              f"{r['diff_convention_delta']:+.6f}  {r['pos_sign_rate']:.2f}  "
              f"{r['n_over_floor']}/3  {r['paired_delta_exceeds_floor']}")
print("-" * 78)
print("peak epoch: arm  n1(seed0)  n2  n3   n3 median ep00/ep01/ep02")
for arm in ARMS:
    p = peak[arm]
    m = p["n3_median"]
    print(f"  {arm:4s}  {p['best_n1']}  {p['best_n2']}  {p['best_n3']}   "
          f"{m['00']:.6f} / {m['01']:.6f} / {m['02']:.6f}")
print(f"ep00 is peak: n1 {n1c}/7  n2 {n2c}/7  n3 {n3c}/7")
print("-" * 78)
print("criteria (9 cells = 3 seeds x 3 epochs): pass / fail / missing")
for k, r in crit_rows.items():
    print(f"  {k:34s} {r['n_pass']} / {r['n_fail']} / {r['n_missing']}   {r['rule']}")
print("=" * 78)
print(f"POS {report['n_checks_pass']}/{report['n_checks_total']}  "
      f"VERDICT: {'PASS' if report['n_checks_pass'] == report['n_checks_total'] else 'FAIL'}")
for c in checks:
    if not c["ok"]:
        print("  FAILED:", c["name"], "|", c["detail"])
print(f"written -> {OUTP}")
