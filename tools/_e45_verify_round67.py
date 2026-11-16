#!/usr/bin/env python3
"""E45 round 67 -- epoch axis, TWO-seed (n=2) controlled read.  READ ONLY.

Inputs (all local):
  tools/_evals/e45/epoch_axis/evals/{ARM}_s{0,1}_ep{00,01,02}.eval.json   (42 cells)
  tools/_evals/e45/epoch_axis/probe/diag_probe_s{0,1}_ep{00,01,02}.json      (6 probes)
  tools/_evals/e45/epoch_axis/{scores_summary,s3_equalbias,diag_probe_s0,diag_probe_s1}.json
  tools/_evals/e45/epoch_axis/md5_{epochaxis_ep00,published}.txt

Discipline carried from rounds 61-66:
  * median + pass rate; a single cell never decides anything
  * paired convention and difference convention reported side by side
  * n=2 -> direction and magnitude only, NO arm-level conclusion
  * zero-model floor reported for every comparable axis
  * "criterion flipped" is split into "read-out out of bounds" vs "read-out missing"
  * post-hoc diagnostic: creates no criterion, changes no verdict
"""
import hashlib
import json
import math
import os
import re
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
AX = ROOT / "tools" / "_evals" / "e45" / "epoch_axis"
OUTP = ROOT / "tools" / "_evals" / "e45" / "round67_epoch_axis_n2.json"
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
EPOCHS = ["00", "01", "02"]
SEEDS = [0, 1]
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


def read_md5(p):
    d = {}
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, f = line.split(None, 1)
        d[os.path.basename(f.strip())] = h
    return d


# ---------------------------------------------------------------- 1. control
mx = read_md5(AX / "md5_epochaxis_ep00.txt")
pub = read_md5(AX / "md5_published.txt")
ctrl = {k: (mx[k] == pub[k]) for k in mx if k in pub}
n_same = sum(ctrl.values())
check("control_ep00_rerun_md5_identical_to_published",
      len(ctrl) > 0 and all(ctrl.values()),
      f"{n_same}/{len(ctrl)} cells byte-identical")
report["control_md5"] = {"n_compared": len(ctrl), "n_identical": n_same}

# ------------------------------------------------------- 2. published guards
ss_md5 = md5_file(AX / "scores_summary.json")
s3_md5 = md5_file(AX / "s3_equalbias.json")
check("published_scores_summary_md5_unchanged",
      ss_md5 == "57bcd511a537ab85494ee41b029731a8", ss_md5)
check("published_s3_equalbias_md5_unchanged",
      s3_md5 == "3aba999d5ef6dcc4a775999be6585eb2", s3_md5)
report["published_md5"] = {"scores_summary": ss_md5, "s3_equalbias": s3_md5}

s3 = load(AX / "s3_equalbias.json")
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
      len(cz) > 0 and all(cz),
      f"{sum(cz)}/{len(cz)} cross zero; verdicts area={v_area} nogain={v_none}")
report["s3_guard"] = {"n_intervals": len(cz), "n_cross_zero": sum(cz),
                      "verdict_area": v_area, "verdict_nogain": v_none,
                      "per_arm": s3_arms}

# published per-arm seed-0 CSI replication (handover section 3)
EXPECT_S0 = {"A4": 0.175500, "A3b": 0.169661, "A3a": 0.149470, "A4b": 0.137567,
             "B0": 0.121817, "A1": 0.118397, "A2": 0.112003}
ss = load(AX / "scores_summary.json")


def pub_csi(obj):
    if isinstance(obj, dict):
        if "csi_at_2.0" in obj:
            return obj["csi_at_2.0"]
        t = obj.get("thresholds")
        if isinstance(t, dict) and "2.0" in t:
            return t["2.0"].get("csi")
    return None


published = {}
for arm, v in ss.items():
    if not isinstance(v, dict):
        continue
    per = v.get("csi_at_2.0")
    if not isinstance(per, dict):
        continue
    for sk, sv in per.items():
        m = re.fullmatch(r"s(\d+)", str(sk))
        if m and isinstance(sv, (int, float)) and not isinstance(sv, bool):
            published[(arm, int(m.group(1)))] = float(sv)
repl = {a: (published.get((a, 0)), EXPECT_S0[a]) for a in EXPECT_S0}
worst = max([abs(v[0] - v[1]) for v in repl.values() if v[0] is not None] or [float("nan")])
check("published_seed0_csi_replicated",
      all(v[0] is not None for v in repl.values()) and worst < 1e-6,
      f"max|diff| = {worst:.3e} over 7 arms")
report["published_seed0_replication"] = {
    "max_abs_diff": worst,
    "per_arm": {a: {"published": v[0], "handover": v[1]} for a, v in repl.items()}}
head_med, head_rate = {}, {}
for arm, v in ss.items():
    per = v.get("csi_at_2.0", {})
    vals = [float(x) for x in per.values() if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if len(vals) == 3:
        head_med[arm] = median(vals)
        head_rate[arm] = (sum(1 for x in vals if x >= FLOOR), len(vals))
med_diff = max([abs(head_med[a] - ss[a]["median_csi_at_2.0"]) for a in head_med] or [float("nan")])
rate_ok = all(ss[a].get("pass_rate_over_floor") == f"{head_rate[a][0]}/{head_rate[a][1]}" for a in head_rate)
check("published_headline_median_and_passrate_reproduced",
      med_diff < 1e-12 and rate_ok,
      f"max|diff| median = {med_diff:.3e}; pass-rate strings match = {rate_ok}")
report["published_headline"] = {"median": head_med,
                                "pass_rate": {a: f"{v[0]}/{v[1]}" for a, v in head_rate.items()}}

# ------------------------------------------------------------- 3. CSI table
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
check("all_42_two_seed_cells_present", len(csi) == 42, f"{len(csi)}/42 cells")
report["n_cells"] = len(csi)

# zero-model floor per epoch: |B0_s0 - B0_s1| (control arm reseeded)
floor_ep = {ep: abs(csi[("B0", 0, ep)] - csi[("B0", 1, ep)]) for ep in EPOCHS}

arm_ep = {}
for arm in ARMS:
    for ep in EPOCHS:
        vs = [csi[(arm, s, ep)] for s in SEEDS if (arm, s, ep) in csi]
        bs = [csi[("B0", s, ep)] for s in SEEDS if ("B0", s, ep) in csi]
        if len(vs) < 2:
            continue
        dl = [csi[(arm, s, ep)] - csi[("B0", s, ep)] for s in SEEDS
              if (arm, s, ep) in csi and ("B0", s, ep) in csi]
        arm_ep[f"{arm}_ep{ep}"] = {
            "median_csi": median(vs),
            "paired_median_delta": median(dl),
            "paired_deltas": dl,
            "diff_convention_delta": median(vs) - median(bs),
            "pos_sign_rate": sum(1 for d in dl if d > 0) / len(dl),
            "seed_spread": abs(vs[0] - vs[1]),
            "exceeds_zero_model_floor": [abs(d) > floor_ep[ep] for d in dl],
            "median_bias": median([bias[(arm, s, ep)] for s in SEEDS]),
            "median_pod": median([pod[(arm, s, ep)] for s in SEEDS]),
            "per_seed_csi": vs,
        }

# peak epoch per arm: n=1 (seed 0) vs n=2 (median of 2 seeds)
peak = {}
for arm in ARMS:
    n1 = {ep: csi[(arm, 0, ep)] for ep in EPOCHS if (arm, 0, ep) in csi}
    n2 = {ep: median([csi[(arm, s, ep)] for s in SEEDS]) for ep in EPOCHS
          if all((arm, s, ep) in csi for s in SEEDS)}
    b1 = max(n1, key=lambda e: n1[e]) if n1 else None
    b2 = max(n2, key=lambda e: n2[e]) if n2 else None
    peak[arm] = {"n1_seed0": n1, "best_n1": b1, "n2_median": n2, "best_n2": b2,
                 "ep00_is_peak_n1": b1 == "00", "ep00_is_peak_n2": b2 == "00",
                 "argmax_changed_n1_to_n2": b1 != b2}
n_ep00_peak_n1 = sum(1 for a in ARMS if peak[a]["ep00_is_peak_n1"])
n_ep00_peak_n2 = sum(1 for a in ARMS if peak[a]["ep00_is_peak_n2"])
n_changed = sum(1 for a in ARMS if peak[a]["argmax_changed_n1_to_n2"])
check("n1_replication_matches_round66",
      abs(peak["B0"]["n1_seed0"]["01"] - 0.130836) < 1e-6
      and abs(peak["B0"]["n1_seed0"]["00"] - 0.121817) < 1e-6
      and peak["B0"]["best_n1"] == "01"
      and peak["A1"]["best_n1"] == "02" and peak["A4b"]["best_n1"] == "02",
      f"n1 peak arms ep00 = {n_ep00_peak_n1}/7 (round 66 reported 4/7)")

report["zero_model_floor_per_epoch"] = floor_ep
report["arm_epoch"] = arm_ep
report["peak_epoch"] = peak
report["ep00_peak_count"] = {"n1": n_ep00_peak_n1, "n2": n_ep00_peak_n2,
                             "argmax_changed": n_changed}

# ------------------------------------------------------- 4. mechanism probes
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
for s in SEEDS:
    for ep in EPOCHS:
        p = AX / "probe" / f"diag_probe_s{s}_ep{ep}.json"
        if not p.exists():
            continue
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
        crit_rows[f"{arm}:{cname}"] = {
            "rule": f"{cname} {op} {thr}",
            "per_cell": cells,
            "n_pass": npass, "n_fail": len(cells) - npass - nmiss, "n_missing": nmiss,
            "pass_rate_over_available": (npass / (len(cells) - nmiss)) if (len(cells) - nmiss) else None,
            "values": {k: fmt(c["value"], 6) for k, c in cells.items()},
        }


def probe_map(p):
    obj = load(p)
    out = {}
    for path, rec in walk(obj):
        a = arm_of(path, rec)
        if a is None:
            continue
        out.setdefault(a, {k: rec.get(k) for k in TARGET if k in rec})
    return out


pub_probe = probe_map(AX / "diag_probe_s0.json")
diff_probe = []
for arm, rec in pub_probe.items():
    new = probe.get((arm, 0, "00"), {})
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

report["mechanism_probe_cells"] = len(probe)
report["mechanism_probe_unresolved_records"] = unresolved
report["probe_control_max_abs_diff"] = maxdiff
report["criteria"] = crit_rows

extra = {}
for arm in ("A1", "A2", "A3a", "A3b", "A4", "A4b"):
    for ep in EPOCHS:
        for s in SEEDS:
            rec = probe.get((arm, s, ep))
            if not rec:
                continue
            extra[f"{arm}_s{s}_ep{ep}"] = {
                "mask_at_peak": fmt(rec.get("mask_at_peak"), 4),
                "corr_mask_amp": fmt(rec.get("corr_mask_amp"), 4),
                "grad_top1pct_share": fmt(rec.get("grad_top1pct_share"), 4),
                "field_max_mm": fmt(rec.get("field_max_mm"), 2),
                "tail_max_over_p99": fmt(rec.get("tail_max_over_p99"), 3),
                "scale_cv": fmt(rec.get("scale_cv"), 4),
                "scale_truth_corr": fmt(rec.get("scale_truth_corr"), 4),
                "scale_mean": fmt(rec.get("scale_mean"), 4),
                "mean_out_equals_T": rec.get("mean_out_equals_T"),
                "maxratio_as_trained": fmt(rec.get("maxratio_as_trained"), 4),
            }
report["mechanism_extra"] = extra

report["verdict"] = {
    "e45_verdict_changed": False,
    "eligible_arm_set": [],
    "note": ("post-hoc diagnostic on the epoch axis; creates no criterion; "
             "n=2 gives direction and magnitude only, no arm-level conclusion; "
             "E45 stays negative (six arms, every S3 interval crosses zero)"),
}
n_ok = sum(1 for c in checks if c["ok"])
report["checks"] = checks
report["n_checks_pass"] = n_ok
report["n_checks_total"] = len(checks)
report["verdict_line"] = f"VERDICT: {'PASS' if n_ok == len(checks) else 'PARTIAL'} ({n_ok}/{len(checks)} positive)"

with open(OUTP, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

# ------------------------------------------------------------------- print
print("=" * 78)
print("E45 ROUND 67 -- epoch axis, TWO-seed (n=2) controlled read")
print("=" * 78)
print(f"cells={report['n_cells']}/42   probe cells={report['mechanism_probe_cells']}"
      f"   unresolved probe records={unresolved}")
print(f"control md5 (ep00 re-run vs published): {n_same}/{len(ctrl)} identical")
print(f"probe control max|diff| = {maxdiff}")
print(f"published md5 unchanged: scores_summary={ss_md5[:16]} s3={s3_md5[:16]}")
print(f"S3 guard: {sum(cz)}/{len(cz)} intervals cross zero; verdicts area={v_area} nogain={v_none}")
print(f"published seed-0 CSI replication max|diff| = {worst:.3e}")
print(f"published headline (median over 3 seeds, pass-rate over floor {FLOOR}): "
      + ", ".join(f"{a} {fmt(head_med.get(a), 6)} [{head_rate.get(a, (0, 0))[0]}/3]" for a in ARMS))
print("S3 per-arm verdicts: " + "; ".join(f"{a}={v['verdict']}" for a, v in s3_arms.items()))
print()
print("-- zero-model floor per epoch (|B0_s0 - B0_s1| at CSI@2.0) --")
for ep in EPOCHS:
    print(f"   ep{ep}: {fmt(floor_ep[ep])}")
print()
print("-- per arm x epoch: median CSI, paired median dCSI, diff convention, sign rate --")
for arm in ARMS:
    for ep in EPOCHS:
        r = arm_ep.get(f"{arm}_ep{ep}")
        if not r:
            continue
        fl = "  <- both |d| beyond zero-model floor" if all(r["exceeds_zero_model_floor"]) else ""
        print(f"   {arm:4s} ep{ep}  medCSI={fmt(r['median_csi'])}"
              f"  pairedD={fmt(r['paired_median_delta']):>10s}"
              f"  diffD={fmt(r['diff_convention_delta']):>10s}"
              f"  pos={r['pos_sign_rate']:.2f}  spread={fmt(r['seed_spread'])}"
              f"  BIAS={fmt(r['median_bias'],4)}{fl}")
print()
print("-- peak epoch: n=1 (seed 0) vs n=2 (median of 2 seeds) --")
for arm in ARMS:
    p = peak[arm]
    meds = " ".join(f"{e}={fmt(v)}" for e, v in p["n2_median"].items())
    print(f"   {arm:4s} best_n1={p['best_n1']}  best_n2={p['best_n2']}"
          f"  changed={p['argmax_changed_n1_to_n2']}   n2 medians: {meds}")
print(f"   ep00-is-peak count: n1 {n_ep00_peak_n1}/7, n2 {n_ep00_peak_n2}/7;"
      f" argmax changed for {n_changed} arms")
print()
print("-- pre-registered mechanism criteria (per arm x epoch, n=2 cells) --")
for k, r in crit_rows.items():
    arm = k.split(":")[0]
    detail = "  ".join(
        f"{ep}:{r['values'][f'{arm}_s0_ep{ep}']}/{r['values'][f'{arm}_s1_ep{ep}']}"
        for ep in EPOCHS)
    print(f"   {arm:4s} {r['rule']:32s} pass={r['n_pass']} fail={r['n_fail']}"
          f" missing={r['n_missing']}  rate={fmt(r['pass_rate_over_available'],3)}   {detail}")
print()
print("-- extra read-outs --")
for k, v in extra.items():
    print(f"   {k:18s} grad1%={v['grad_top1pct_share']:>9s} fmax={v['field_max_mm']:>8s}"
          f" cv={v['scale_cv']:>8s} s_corr={v['scale_truth_corr']:>8s}"
          f" mAtPeak={v['mask_at_peak']:>8s} cMA={v['corr_mask_amp']:>8s}"
          f" eqT={str(v['mean_out_equals_T']):>5s} maxratio={v['maxratio_as_trained']}")
print()
for c in checks:
    print(f"   [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}  -- {c['detail']}")
print()
print(report["verdict_line"])
print(f"product -> {OUTP}")
