#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 57 -- (A) aggregation-operator sensitivity of the published
verdict, (B) evaluation-contract / provenance audit of the 21 cells.

Why this axis (never judged in 56 rounds):
  Every published E45 number is built by ONE aggregation operator: pool the
  538 windows' contingency tables, form CSI, then take the median over the
  3 seeds.  That operator is a modelling choice, not a law.  If the verdict
  ("all six arms negative") survives every equally defensible operator, the
  verdict is operator-robust; if some operator flips an arm, the headline
  table is operator-dependent and must say so.

  Part B audits the *contract* the 21 cells actually ran under (mode, input
  variable sets, checkpoint provenance, dropped frames, window alignment) --
  the preconditions every paired comparison in rounds 44-56 silently relied
  on -- plus a read-only audit of the remote training configs.

Read-only on local artifacts + one already-downloaded remote audit.
Writes one JSON artifact.  Post-hoc diagnosis only: creates NO new criterion
and CANNOT change the E45 verdict.
"""
import json
import os
import re
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "_evals", "e45")
RAWD = os.path.join(EVAL, "_raw", "runs", "e45", "evals")
OUT = os.path.join(EVAL, "round57_operator_sensitivity.json")
AUD = os.path.join(EVAL, "round57_remote_config_audit.json")

THR = "2.0"
FLOOR = 0.0924
ORACLE = 0.1653
ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
TREAT = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
CTRL = "B0"

POS = 0
FAIL = 0


def check(ok, msg):
    global POS, FAIL
    if ok:
        POS += 1
    else:
        FAIL += 1
    print(("  PASS  " if ok else "  FAIL  ") + msg)


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    m = n // 2
    return xs[m] if n % 2 else 0.5 * (xs[m - 1] + xs[m])


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def trimmed_mean(xs, frac=0.10):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    k = int(n * frac)
    core = xs[k:n - k] if n - 2 * k >= 1 else xs
    return sum(core) / len(core)


def case_of(key):
    return key.split("_000")[0]


def fams(h, m, f, cn):
    """Standard scores from a contingency quadruple."""
    T = h + m
    F = h + f
    pod = (h / T) if T > 0 else 0.0
    far = (f / F) if F > 0 else 0.0
    csi = (h / (h + m + f)) if (h + m + f) > 0 else 0.0
    bias = (F / T) if T > 0 else 0.0
    return {"csi": csi, "pod": pod, "far": far, "bias": bias}


# ---------------------------------------------------------------- load
cells = {}
for _a in ARMS:
    for _s in SEEDS:
        _stem = "%s_s%d_ep00" % (_a, _s)
        with open(os.path.join(RAWD, _stem + ".eval.json"), encoding="utf-8") as fh:
            _ev = json.load(fh)
        with open(os.path.join(RAWD, _stem + ".windows.json"), encoding="utf-8") as fh:
            _wj = json.load(fh)
        cells[(_a, _s)] = {"eval": _ev, "win": _wj}


def counts(cell):
    out = []
    for w in cell["win"]["windows"]:
        c = w[THR]
        out.append((float(c["hits"]), float(c["misses"]),
                    float(c["false_alarms"]), float(c["correct_negatives"])))
    return out


CNT = {k: counts(v) for k, v in cells.items()}
KEYS = {k: list(v["win"]["window_keys"]) for k, v in cells.items()}
CASE_IDX = [case_of(k) for k in KEYS[(CTRL, 0)]]


# ------------------------------------------------------------ operators
def op_pooled(cs):
    H = sum(c[0] for c in cs)
    M = sum(c[1] for c in cs)
    F = sum(c[2] for c in cs)
    return H / (H + M + F) if (H + M + F) > 0 else None


def win_csis(cs):
    out = []
    for h, m, f, _cn in cs:
        d = h + m + f
        if d > 0:
            out.append(h / d)
    return out


def op_macro_mean(cs):
    return mean(win_csis(cs))


def op_macro_median(cs):
    return med(win_csis(cs))


def op_trimmed(cs):
    return trimmed_mean(win_csis(cs), 0.10)


def op_case_equal(cs):
    acc = {}
    for i, (h, m, f, _cn) in enumerate(cs):
        a = acc.setdefault(CASE_IDX[i], [0.0, 0.0, 0.0])
        a[0] += h
        a[1] += m
        a[2] += f
    vals = []
    for k in sorted(acc):
        H, M, F = acc[k]
        if (H + M + F) > 0:
            vals.append(H / (H + M + F))
    return mean(vals)


OPS = OrderedDict([
    ("pooled", op_pooled),          # published operator
    ("macro_mean", op_macro_mean),   # equal weight per window
    ("macro_median", op_macro_median),
    ("trimmed10", op_trimmed),       # 10% trimmed mean per window
    ("case_equal", op_case_equal),   # equal weight per storm case
])

VAL = {op: {k: fn(CNT[k]) for k in CNT} for op, fn in OPS.items()}


# ------------------------------------------------- A. operator sensitivity
def paired_gain(op, arm):
    """median over seeds of (arm - control), same seed."""
    return med([VAL[op][(arm, s)] - VAL[op][(CTRL, s)] for s in SEEDS])


def diff_caliber(op, arm):
    return med([VAL[op][(arm, s)] for s in SEEDS]) - med([VAL[op][(CTRL, s)] for s in SEEDS])


def zero_floor(op):
    """control-vs-control seed swaps: max |delta| over the 3 pairs."""
    d = []
    for i in range(3):
        for j in range(i + 1, 3):
            d.append(abs(VAL[op][(CTRL, i)] - VAL[op][(CTRL, j)]))
    return max(d)


def pass_rate(op, arm):
    return sum(1 for s in SEEDS if VAL[op][(arm, s)] > FLOOR) / 3.0


sens = OrderedDict()
for op in OPS:
    floor = zero_floor(op)
    rows = {}
    for a in ARMS:
        pg = paired_gain(op, a)
        dc = diff_caliber(op, a)
        rows[a] = {
            "median_csi": med([VAL[op][(a, s)] for s in SEEDS]),
            "pass_rate_over_floor": pass_rate(op, a),
            "paired_median_gain": pg,
            "diff_caliber_gain": dc,
            "seeds_positive_paired": sum(
                1 for s in SEEDS if VAL[op][(a, s)] - VAL[op][(CTRL, s)] > 0),
            "exceeds_zero_model_floor": bool(abs(pg) > floor),
            "effect_over_floor_ratio": (pg / floor) if floor else None,
        }
    sens[op] = {"zero_model_floor": floor, "arms": rows}

# operator-induced band per arm (paired median gain)
band = {}
for a in ARMS:
    vals = [sens[op]["arms"][a]["paired_median_gain"] for op in OPS]
    band[a] = {"min": min(vals), "max": max(vals), "width": max(vals) - min(vals),
               "per_operator": {op: sens[op]["arms"][a]["paired_median_gain"] for op in OPS}}

# does any operator turn a published-negative arm into a floor-exceeding positive?
flips = []
for a in TREAT:
    for op in OPS:
        r = sens[op]["arms"][a]
        if r["paired_median_gain"] > 0 and r["exceeds_zero_model_floor"]:
            flips.append({"arm": a, "operator": op, "paired_median_gain": r["paired_median_gain"],
                          "zero_model_floor": sens[op]["zero_model_floor"],
                          "effect_over_floor_ratio": r["effect_over_floor_ratio"],
                          "seeds_positive": r["seeds_positive_paired"]})

# sign pattern of the paired gain, per operator
sign_pattern = {}
for op in OPS:
    sign_pattern[op] = {a: ("+" if sens[op]["arms"][a]["paired_median_gain"] > 0 else "-")
                         for a in TREAT}
distinct_patterns = sorted(set(tuple(v[a] for a in TREAT) for v in sign_pattern.values()))

# rank stability across operators (Spearman of arm ordering vs pooled)
def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    for pos, i in enumerate(order):
        rk[i] = pos
    return rk


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = mean(ra), mean(rb)
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else None


pooled_order = [sens["pooled"]["arms"][a]["median_csi"] for a in ARMS]
rank_stab = {op: spearman(pooled_order, [sens[op]["arms"][a]["median_csi"] for a in ARMS])
             for op in OPS}

# ------------------------- A2. area-equalised (S3-domain counterpart) ----
def equalise_area(cs_arm, cs_ctrl):
    """Per-window proportional rescale of the arm's forecast-event AREA onto
    the control's, keeping the truth T = h+m fixed.  This is round 54's
    contingency-domain counterpart of S3's field-domain global scalar
    rescale, and it MUST use the same formula or the cross-check is
    meaningless."""
    out = []
    for (h, m, f, cn), (h0, _m0, f0, _cn0) in zip(cs_arm, cs_ctrl):
        A = h + f
        target = h0 + f0
        if A <= 0 or target <= 0:
            out.append((h, m, f, cn))
            continue
        sc = target / A
        h2, f2 = h * sc, f * sc
        T = h + m
        m2 = T - h2
        cn2 = cn + (f - f2)
        out.append((h2, m2, f2, cn2))
    return out


EQ_CNT = {}
for a in TREAT:
    for s in SEEDS:
        EQ_CNT[(a, s)] = equalise_area(CNT[(a, s)], CNT[(CTRL, s)])

eq = OrderedDict()
for op, fn in OPS.items():
    eq[op] = {}
    for a in TREAT:
        deltas = [fn(EQ_CNT[(a, s)]) - fn(CNT[(CTRL, s)]) for s in SEEDS]
        eq[op][a] = {"paired_median_gain_after_area_equalisation": med(deltas),
                     "mean_gain_after_area_equalisation": mean(deltas),
                     "seeds_positive": sum(1 for d in deltas if d > 0)}

# sign agreement with the published S3 field-domain equal-bias point
s3 = json.load(open(os.path.join(EVAL, "s3_equalbias.json"), encoding="utf-8"))
s3_sign = {a: ("pos" if s3["arms"][a]["equal_bias"]["point"] > 0 else "neg") for a in TREAT}
eq_sign = {op: {a: ("pos" if eq[op][a]["paired_median_gain_after_area_equalisation"] > 0 else "neg")
                for a in TREAT} for op in OPS}
sign_agree = {op: sum(1 for a in TREAT if eq_sign[op][a] == s3_sign[a]) for op in OPS}

# truth invariance under the equalisation (the rescale must not move T)
t_dev = 0.0
for a in TREAT:
    for s in SEEDS:
        for (h, m, _f, _cn), (_h0, _m0, _f0, _cn0) in zip(CNT[(a, s)], CNT[(CTRL, s)]):
            t_dev = max(t_dev, abs((h + m) - (_h0 + _m0)))


# ------------------------- A3. per-case driver (why case_equal differs) ----
# The `case_equal` operator above is the MEAN over cases of per-case CSI.
# Round 47's case-level statistic was the MEDIAN over cases of per-case
# DELTA -- a different statistic on the same data.  Both are reported so
# the two rounds cannot be silently compared against each other.
def per_case_csi(cs):
    acc = {}
    for i, (h, m, f, _cn) in enumerate(cs):
        a = acc.setdefault(CASE_IDX[i], [0.0, 0.0, 0.0])
        a[0] += h
        a[1] += m
        a[2] += f
    out = {}
    for c in sorted(acc):
        H, M, F = acc[c]
        out[c] = (H / (H + M + F)) if (H + M + F) > 0 else 0.0
    return out


CASES = sorted(set(CASE_IDX))
case_driver = {}
for a in TREAT:
    per_seed_case = {}
    for s in SEEDS:
        ca = per_case_csi(CNT[(a, s)])
        cb = per_case_csi(CNT[(CTRL, s)])
        per_seed_case[s] = {c: ca[c] - cb[c] for c in CASES}
    med_case = {c: med([per_seed_case[s][c] for s in SEEDS]) for c in CASES}
    abs_sum = sum(abs(v) for v in med_case.values()) or 1e-12
    top = sorted(med_case.items(), key=lambda kv: -abs(kv[1]))
    case_driver[a] = {
        "per_case_median_delta": med_case,
        "n_cases_positive": sum(1 for v in med_case.values() if v > 0),
        "n_cases": len(CASES),
        "mean_case_delta": mean(list(med_case.values())),
        "top_driver_case": top[0][0],
        "top_driver_delta": top[0][1],
        "top_driver_share_of_abs": abs(top[0][1]) / abs_sum,
    }

# zero model at case level: control arm, different seeds
case_null = {}
for i in range(3):
    for j in range(i + 1, 3):
        ca = per_case_csi(CNT[(CTRL, i)])
        cb = per_case_csi(CNT[(CTRL, j)])
        d = {c: ca[c] - cb[c] for c in CASES}
        case_null["B0_s%d_vs_B0_s%d" % (i, j)] = {
            "mean_case_delta": mean(list(d.values())),
            "max_abs_case_delta": max(abs(v) for v in d.values()),
        }
case_null_floor_mean = max(abs(v["mean_case_delta"]) for v in case_null.values())
case_null_floor_max = max(v["max_abs_case_delta"] for v in case_null.values())

# round-47 caliber (median over cases of per-case DELTA) for cross-round
# comparability -- NOT the same statistic as `case_equal` above.
def r47_caliber(a, s_a, s_b):
    ca = per_case_csi(CNT[(a, s_a)])
    cb = per_case_csi(CNT[(CTRL, s_b)])
    return med([ca[c] - cb[c] for c in CASES])


r47_null = {("B0_s%d" % i, "B0_s%d" % j): r47_caliber("B0", i, j)
            for i in range(3) for j in range(i + 1, 3)}
r47_floor = max(abs(v) for v in r47_null.values())
r47_arms = {}
for a in TREAT:
    pmeds = [r47_caliber(a, s, s) for s in SEEDS]
    r47_arms[a] = {"paired_median_over_seeds": med(pmeds),
                   "seeds_positive": sum(1 for v in pmeds if v > 0),
                   "within_null_floor": bool(abs(med(pmeds)) <= r47_floor)}


# ------------------------------------------------------------ Part B audit
audit = {}
arm_table = {}
_arm_src = open(os.path.join(HERE, "..", "scripts", "_e45_arms.sh"), encoding="utf-8").read()
_block = _arm_src.split("E45_ARM_TABLE=(", 1)[1].split("\n)", 1)[0]
for _m in re.findall(r'"([^"]*)"', _block):
    if "|" in _m:
        _f = _m.split("|")
        if len(_f) >= 2 and _f[0] in ARMS:
            arm_table[_f[0]] = {"model": _f[1], "loss": _f[2],
                                "train_flags": _f[3] if len(_f) > 3 else "",
                                "probe_flags": _f[4] if len(_f) > 4 else ""}

nwp_sets, radar_sets, modes = {}, {}, {}
ckpt_ok, ckpt_detail = 0, {}
miss_total, nsamp = 0, set()
seed_ts = {}
for a in ARMS:
    for s in SEEDS:
        k = (a, s)
        ev = cells[k]["eval"]
        ab = ev["ablation"]
        nwp_sets[k] = tuple(sorted(ab["nwp_vars"]))
        radar_sets[k] = tuple(sorted(ab["radar_vars"]))
        modes[k] = ab["mode"]
        ck = ab["checkpoint"]
        m = re.match(r"^/home/wf/radarNWP/runs/e45/([A-Za-z0-9]+_s\d+)/train/"
                     r"([A-Za-z0-9_]+)_(\d+)/epoch_(\d+)\.pt$", ck)
        ok = False
        if m:
            cell_dir, mdir, ts, ep = m.group(1), m.group(2), m.group(3), m.group(4)
            exp_model = arm_table[a]["model"]
            ok = (cell_dir == "%s_s%d" % (a, s) and mdir == exp_model and ep == "00")
            seed_ts.setdefault(s, set()).add(ts)
            ckpt_detail["%s_s%d" % (a, s)] = {
                "cell_dir": cell_dir, "model_dir": mdir, "expected_model": exp_model,
                "epoch": ep, "launch_stamp": ts, "matches_arm_table": bool(ok)}
        if ok:
            ckpt_ok += 1
        else:
            ckpt_detail["%s_s%d" % (a, s)] = {"checkpoint": ck, "matches_arm_table": False}
        miss_total += sum(len(x) for x in cells[k]["win"]["missing_frame_steps"])
        nsamp.add((ev["n_samples"], cells[k]["win"]["n_windows"]))

print("\n--- Part B: evaluation contract / provenance ---")
check(len(cells) == 21, "G1  21 cells loaded")
check(all(v == "full" for v in modes.values()), "G2  every cell ran in ablation mode 'full'")
check(len(set(nwp_sets.values())) == 1 and len(set(radar_sets.values())) == 1,
      "G3  input variable sets identical across all 21 cells "
      "(%d NWP vars, %d radar vars)" % (len(next(iter(nwp_sets.values()))),
                                        len(next(iter(radar_sets.values())))))
if ckpt_ok != 21:
    for _k, _v in ckpt_detail.items():
        if not _v.get("matches_arm_table"):
            print("        mismatch: %s -> %s" % (_k, _v))
check(ckpt_ok == 21, "G4  checkpoint provenance matches the arm table for 21/21 cells")
check(miss_total == 0, "G5  zero dropped frames across all 21 cells (missing_frame_steps)")
check(nsamp == {(538, 538)}, "G6  every cell has 538 samples / 538 windows")
_keys0 = KEYS[(CTRL, 0)]
check(all(KEYS[k] == _keys0 for k in KEYS), "G7  window_keys identical and aligned across 21 cells")

seed_block = {s: sorted(v) for s, v in seed_ts.items()}
seed_aligned = all(len(v) <= 2 for v in seed_block.values())
print("        seed launch stamps: %s" % json.dumps(seed_block))
check(seed_aligned, "G8  seeds are synchronised blocks (all 7 arms share a launch stamp)")

crosses = sum(1 for a in TREAT for k in ("raw", "equal_bias") if s3["arms"][a][k]["crosses_zero"])
check(crosses == 12, "G9  S3: 12/12 published intervals still cross zero")
exp_verdict = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A2": "NO_GAIN_TO_BEGIN_WITH",
                "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
                "A4": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4b": "NO_GAIN_TO_BEGIN_WITH"}
check(all(s3["arms"][a]["s3_verdict"] == exp_verdict[a] for a in TREAT),
      "G10 S3 verdicts unchanged (4 GAIN_WAS_AREA_OR_AMPLITUDE / 2 NO_GAIN_TO_BEGIN_WITH)")

pub = json.load(open(os.path.join(EVAL, "scores_summary.json"), encoding="utf-8"))
maxdiff = 0.0
for a in ARMS:
    for s in SEEDS:
        maxdiff = max(maxdiff, abs(VAL["pooled"][(a, s)] - pub[a]["csi_at_2.0"]["s%d" % s]))
check(maxdiff < 1e-12, "G11 pooled operator reproduces the published CSI@2.0 for 21 cells "
                       "(21 comparisons, max|diff| = %.3e)" % maxdiff)

exp_paired_sign = {"A1": -1, "A2": -1, "A3a": +1, "A3b": +1, "A4": +1, "A4b": +1}
check(all((sens["pooled"]["arms"][a]["paired_median_gain"] > 0) ==
          (exp_paired_sign[a] > 0) for a in TREAT),
      "G12 published (pooled) paired-gain signs reproduce: 4 positive / 2 negative")

check(t_dev < 1e-6, "G13 area equalisation leaves the truth event count T=h+m untouched "
                    "(max deviation %.3e)" % t_dev)

# ------------------------------------------------ remote config audit (opt)
remote = {"available": False}
AUD2 = os.path.join(EVAL, "round57_remote_config_digest.json")
if os.path.exists(AUD2):
    try:
        with open(AUD2, encoding="utf-8") as fh:
            rc = json.load(fh)
        remote["available"] = True
        per_cell = {}
        for a in ARMS:
            for s in SEEDS:
                cell = "%s_s%d" % (a, s)
                cfgs = rc.get(cell, {})
                rec = {}
                for _rel, cfg in cfgs.items():
                    rec = cfg
                    break
                per_cell[cell] = {
                    "model": rec.get("model"),
                    "loss": rec.get("loss"),
                    "alpha_csi": rec.get("alpha_csi"),
                    "topo_clamp_mm": rec.get("topo_clamp_mm"),
                    "topo_truth_smooth_sigma": rec.get("topo_truth_smooth_sigma"),
                    "topo_area_normalize": rec.get("topo_area_normalize"),
                    "no_nwp": rec.get("no_nwp"),
                    "zero_nwp": rec.get("zero_nwp"),
                    "split": rec.get("split"),
                    "data_root": rec.get("data_root"),
                    "n_config_files": len(cfgs),
                }
        remote["per_cell"] = per_cell

        # --- value-level provenance: only the VALUE discriminates, because
        # every config carries every flag with a default.
        def val(a, s, key):
            return per_cell["%s_s%d" % (a, s)][key]

        remote["model_matches_arm_table"] = all(
            val(a, s, "model") == arm_table[a]["model"] for a in ARMS for s in SEEDS)
        remote["a3a_clamp_only"] = all(
            val("A3a", s, "topo_clamp_mm") == 20.0 and
            val("A3a", s, "topo_truth_smooth_sigma") == 0.0 and
            val("A3a", s, "topo_area_normalize") is False for s in SEEDS)
        remote["a3b_smooth_area_only"] = all(
            val("A3b", s, "topo_clamp_mm") == 0.0 and
            val("A3b", s, "topo_truth_smooth_sigma") == 1.0 and
            val("A3b", s, "topo_area_normalize") is True for s in SEEDS)
        remote["others_topo_flags_off"] = all(
            val(a, s, "topo_clamp_mm") == 0.0 and
            val(a, s, "topo_truth_smooth_sigma") == 0.0 and
            val(a, s, "topo_area_normalize") is False
            for a in ("B0", "A1", "A2", "A4", "A4b") for s in SEEDS)
        remote["loss_as_armed"] = all(
            val(a, s, "loss") == arm_table[a]["loss"] for a in ARMS for s in SEEDS)
        remote["alpha_csi_half"] = all(val(a, s, "alpha_csi") == 0.5
                                       for a in ARMS for s in SEEDS)
        remote["nwp_present"] = all(val(a, s, "no_nwp") is False and
                                    val(a, s, "zero_nwp") is False
                                    for a in ARMS for s in SEEDS)
        remote["same_data_root"] = len(set(val(a, s, "data_root")
                                           for a in ARMS for s in SEEDS)) == 1
        remote["one_config_per_cell"] = all(per_cell["%s_s%d" % (a, s)]["n_config_files"] == 1
                                            for a in ARMS for s in SEEDS)
        print("        value-level config matrix (model / loss / clamp / smooth / areaN):")
        for a in ARMS:
            print("          %-4s " % a + " ".join(
                "%s: %s/%s/%.1f/%.1f/%s" % (s, val(a, s, "model")[:12], val(a, s, "loss"),
                                            val(a, s, "topo_clamp_mm"),
                                            val(a, s, "topo_truth_smooth_sigma"),
                                            val(a, s, "topo_area_normalize"))
                for s in SEEDS))
    except Exception as exc:
        remote["error"] = repr(exc)

if remote.get("available"):
    check(remote.get("model_matches_arm_table"),
          "G14 model field matches the arm table for 21/21 cells")
    check(remote.get("a3a_clamp_only"),
          "G15 A3a: topo_clamp_mm=20.0 and the other two topology flags off (3/3 seeds)")
    check(remote.get("a3b_smooth_area_only"),
          "G16 A3b: topo_truth_smooth_sigma=1.0 + topo_area_normalize=True, clamp off (3/3)")
    check(remote.get("others_topo_flags_off"),
          "G17 the other five arms carry all three topology flags at their default (off)")
    check(remote.get("loss_as_armed"),
          "G18 loss field matches the arm table (balanced for 5 arms, topo for A3a/A3b)")
    check(remote.get("alpha_csi_half"), "G19 alpha_csi = 0.5 in all 21 configs")
    check(remote.get("nwp_present"), "G20 NWP present in all 21 configs (no_nwp/zero_nwp false)")
    check(remote.get("same_data_root"), "G21 all 21 configs point at the same data_root")
    check(remote.get("one_config_per_cell"), "G22 exactly one training config per cell (21/21)")
else:
    print("  INFO  G14-22 remote config audit not available locally (skipped)")


# ------------------------------------------------------------- reporting
print("\n--- Part A: aggregation-operator sensitivity (thr=2.0 mm) ---")
for op in OPS:
    fl = sens[op]["zero_model_floor"]
    print("  %-13s zero-model floor = %.6f | control pass-rate over floor = %.2f | "
          "control median CSI = %.5f" % (op, fl, sens[op]["arms"][CTRL]["pass_rate_over_floor"],
                                         sens[op]["arms"][CTRL]["median_csi"]))
print()
print("  %-5s %-11s %-11s %-11s %-11s %-11s" % ("arm", "pooled", "macro_mean", "macro_median",
                                                "trimmed10", "case_equal"))
for a in ARMS:
    row = "  %-5s " % a
    for op in OPS:
        row += "%+11.5f " % sens[op]["arms"][a]["paired_median_gain"]
    print(row)
print("\n  cells = paired median gain vs control (same seed); control B0 = 0 by construction")
print("  sign pattern per operator (A1,A2,A3a,A3b,A4,A4b): %s" %
      json.dumps({op: "".join(sign_pattern[op][a] for a in TREAT) for op in OPS}))
print("  distinct sign patterns across operators: %d" % len(distinct_patterns))
print("  operator band width per arm (paired gain): " +
      ", ".join("%s %.5f" % (a, band[a]["width"]) for a in ARMS))
print("  effect/floor ratio per arm (paired gain / zero-model floor):")
for a in TREAT:
    print("    %-4s " % a + " ".join("%s %+.2f" % (op, sens[op]["arms"][a]["effect_over_floor_ratio"])
                                      for op in OPS))
print("  area-equalised paired median gain, and S3 sign agreement:")
for op in OPS:
    print("    %-13s " % op + " ".join("%s %+.5f" % (a, eq[op][a]["paired_median_gain_after_area_equalisation"])
                                        for a in TREAT) + "   agree %d/6" % sign_agree[op])
print()
print("  case-level driver (median over cases of per-case delta, arm vs B0 same seed):")
for a in TREAT:
    cd = case_driver[a]
    print("    %-4s cases>0 %d/%d  mean %+.5f  top driver %s %+.5f (%.0f%% of |sum|)" % (
        a, cd["n_cases_positive"], cd["n_cases"], cd["mean_case_delta"],
        cd["top_driver_case"].replace("CP_01_", ""), cd["top_driver_delta"],
        100 * cd["top_driver_share_of_abs"]))
print("    control-vs-control case-level null: mean-case floor %.5f | max-case floor %.5f" % (
    case_null_floor_mean, case_null_floor_max))
print("  round-47 caliber (median over cases of per-case delta): null floor %.5f | %s" % (
    r47_floor, " ".join("%s %+.5f(%d/3,within=%s)" % (
        a, r47_arms[a]["paired_median_over_seeds"], r47_arms[a]["seeds_positive"],
        r47_arms[a]["within_null_floor"]) for a in TREAT)))

results = {
    "round": 57,
    "axis": "aggregation-operator sensitivity + evaluation-contract/provenance audit",
    "threshold_mm": 2.0,
    "floor": FLOOR,
    "oracle": ORACLE,
    "operators": list(OPS.keys()),
    "published_operator": "pooled",
    "operator_sensitivity": {op: sens[op] for op in OPS},
    "operator_band_per_arm": band,
    "operator_induced_sign_flips": flips,
    "sign_pattern_per_operator": {op: sign_pattern[op] for op in OPS},
    "distinct_sign_patterns": len(distinct_patterns),
    "rank_stability_vs_pooled": rank_stab,
    "area_equalised_counterpart": {op: eq[op] for op in OPS},
    "area_equalised_sign_agreement_with_s3": sign_agree,
    "case_level_driver": {
        "cases": CASES,
        "windows_per_case": {c: CASE_IDX.count(c) for c in CASES},
        "per_arm": case_driver,
        "control_seed_swap_null": case_null,
        "null_floor_mean_case": case_null_floor_mean,
        "null_floor_max_case": case_null_floor_max,
    },
    "round47_caliber_recheck": {
        "statistic": "median over cases of per-case delta (round 47's caliber)",
        "null_floor": r47_floor,
        "null_pairs": {"%s_vs_%s" % k: v for k, v in r47_null.items()},
        "arms": r47_arms,
        "note": ("recomputed here for cross-round comparability; this is NOT the "
                 "same statistic as the 'case_equal' operator above (mean over "
                 "cases of per-case CSI), so the two floors are not comparable"),
    },
    "s3_signs": s3_sign,
    "contract_audit": {
        "modes": sorted(set(modes.values())),
        "nwp_vars": list(next(iter(nwp_sets.values()))),
        "radar_vars": list(next(iter(radar_sets.values()))),
        "checkpoint_provenance_ok_cells": ckpt_ok,
        "checkpoint_detail": ckpt_detail,
        "dropped_frames_total": miss_total,
        "samples_windows": sorted(nsamp),
        "seed_launch_stamps": {str(k): v for k, v in seed_block.items()},
        "arm_table": arm_table,
        "truth_invariance_max_dev": t_dev,
    },
    "remote_config_audit": remote,
    "verdict": {
        "s3_intervals_crossing_zero": crosses,
        "s3_verdicts": {a: s3["arms"][a]["s3_verdict"] for a in TREAT},
        "qualified_arms": [],
        "master_version_unchanged": "v2.48",
        "statement": ("post-hoc diagnosis only; no new criterion; verdict unchanged. "
                      "The published (pooled) operator is not the only defensible one, "
                      "but the negative verdict must be reported together with the "
                      "operator band and the zero-model floor."),
    },
    "checks": {"pos": POS, "fail": FAIL},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(results, indent=2, ensure_ascii=False))

print("\n" + "=" * 78)
print("POS %d / FAIL %d  ->  %s" % (POS, FAIL, "VERDICT: PASS" if FAIL == 0 else "VERDICT: FAIL"))
print("artifact: %s" % OUT)
print("=" * 78)
if FAIL:
    raise SystemExit(1)
