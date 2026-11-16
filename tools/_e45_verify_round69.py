#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E45 round 69 -- epoch axis x S3 EQUAL-BIAS GUARD (first time).  READ ONLY.

Why this round exists
---------------------
Rounds 65-68 brought the epoch axis (ep00/ep01/ep02 x 7 arms x 3 seeds) into
existence and judged it on RAW CSI/BIAS plus the mechanism probe.  The verdict
that actually decided E45 -- the S3 "equal bias" guard, i.e. "is the gain an
area/amplitude artifact?" -- was NEVER applied to ep01/ep02.  The open question
left by round 68 is:

    does the six-arm negative verdict survive a change of epoch, and if the
    REASON for the verdict changes, how?

Instrument
----------
The published S3 is FIELD-domain: it rescales the forecast FIELD by one global
scalar so that its bias matches the control's, then recomputes CSI.  It needs
the raw forecast fields, which are not shipped with the epoch-axis artefacts.
Therefore this round uses the CONTINGENCY-DOMAIN counterpart built and
validated in round 54 (`tools/_e45_verify_round54.py`): rescale the arm's
forecast-event AREA to the control's area (truth event count T=h+m held
invariant), recompute CSI.  Round 54 validated it on ep00: 6/6 sign agreement
with the published field-domain S3 point estimates.

The uncertainty model is the SAME two-level structure as the published guard
(`tools/_e41_s3_equalbias.py::two_level_bootstrap`): resample seeds, then
resample windows inside the drawn seed; replication unit is the seed (n=3),
window resampling only describes within-seed sampling consistency (round 46/48
discipline: window-level intervals are pseudo-replicates).

The verdict logic is copied VERBATIM from the published guard:
    gained_raw = raw["point"] > 0
    survived   = (not equal["crosses_zero"]) and equal["point"] > 0
    verdict    = GAIN_SURVIVES_EQUALISATION if survived else
                 (GAIN_WAS_AREA_OR_AMPLITUDE if gained_raw else
                  NO_GAIN_TO_BEGIN_WITH)

Fidelity is re-verified at ep00 BEFORE the instrument is used at ep01/ep02:
  * raw arm-level points must reproduce the published S3 raw points EXACTLY
  * raw intervals must track the published ones (width + crossing decision)
  * the instrument's verdict must agree with the published verdict per arm
Any disagreement is documented as an instrument limitation, not as a finding.

Discipline carried from rounds 46-68:
  * median + pass rate; a single cell never decides anything
  * paired convention and difference convention side by side
  * zero-model floor reported for every comparable axis
  * post-hoc diagnostic: creates no criterion, changes no verdict
  * assertions never assume the data will agree with the conclusion
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"
AX = EV / "epoch_axis"
OUTP = EV / "round69_epoch_guard_axis.json"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
CTRL = "B0"
SIX = [a for a in ARMS if a != CTRL]
SEEDS = [0, 1, 2]
EPOCHS = ["00", "01", "02"]
THR = "2.0"
FLOOR = 0.0924
N_BOOT = 4000
RNG_SEED = 20261115

# round 68 published zero-model floors (HANDOVER E45 section 30) -- used only
# to cross-validate this round's independently computed null floors.
R68_FLOORS = {"00": 0.030118, "01": 0.060224, "02": 0.058907}

report = {}
checks = []


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    return bool(ok)


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def median(xs):
    v = sorted(xs)
    n = len(v)
    if n == 0:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def csi(h, m, f):
    d = h + m + f
    return (h / d) if d > 0 else float("nan")


def csi_v(h, m, f):
    d = h + m + f
    return np.where(d > 0, h / np.maximum(d, 1e-12), np.nan)


# --------------------------------------------------------------------------
# 0. guard: published artefacts must not have moved
# --------------------------------------------------------------------------
ss_md5 = md5_file(EV / "scores_summary.json")
s3_md5 = md5_file(EV / "s3_equalbias.json")
check("published_scores_summary_md5_unchanged",
      ss_md5 == "57bcd511a537ab85494ee41b029731a8", ss_md5)
check("published_s3_equalbias_md5_unchanged",
      s3_md5 == "3aba999d5ef6dcc4a775999be6585eb2", s3_md5)
report["published_md5"] = {"scores_summary": ss_md5, "s3_equalbias": s3_md5}

s3 = load(EV / "s3_equalbias.json")
cz = []
for arm in SIX:
    for label in ("raw", "equal_bias"):
        cz.append(bool(s3["arms"][arm][label]["crosses_zero"]))
check("s3_all_12_intervals_cross_zero", all(cz), f"{sum(cz)}/12 cross zero")
v_area = sum(1 for a in SIX
             if s3["arms"][a]["s3_verdict"] == "GAIN_WAS_AREA_OR_AMPLITUDE")
v_none = sum(1 for a in SIX
             if s3["arms"][a]["s3_verdict"] == "NO_GAIN_TO_BEGIN_WITH")
check("s3_verdicts_unchanged_4_area_2_nogain",
      (v_area, v_none) == (4, 2), f"area={v_area} nogain={v_none}")
check("published_qualified_set_still_empty",
      all(s3["arms"][a]["s3_verdict"] != "GAIN_SURVIVES_EQUALISATION"
          for a in SIX), "0/6 arms survive equalisation in the published guard")
report["s3_guard"] = {
    "n_intervals": len(cz), "n_cross_zero": sum(cz),
    "verdicts": {a: s3["arms"][a]["s3_verdict"] for a in SIX},
}

summ = {s: load(EV / f"scores_summary_s{s}.json") for s in SEEDS}
pub_head = load(EV / "scores_summary.json")


def pub_pass_rate(v):
    """Published pass rate may be a string like '2/3' or a float."""
    if isinstance(v, str):
        if "/" in v:
            a, b = v.split("/")
            return float(a) / float(b)
        return float(v)
    return float(v)


head = {}
for a in ARMS:
    vals = [float(summ[s][f"{a}_s{s}_ep00"]["csi_at_2.0"]) for s in SEEDS]
    head[a] = {"median_csi_at_2.0": median(vals), "per_seed": vals,
               "pass_rate_over_floor": sum(1 for v in vals if v >= FLOOR) / 3.0}
hdev = 0.0
for a in ARMS:
    hdev = max(hdev, abs(head[a]["median_csi_at_2.0"]
                         - float(pub_head[a]["median_csi_at_2.0"])))
    hdev = max(hdev, abs(head[a]["pass_rate_over_floor"]
                         - pub_pass_rate(pub_head[a]["pass_rate_over_floor"])))
check("published_headline_median_passrate_reproduced", hdev < 1e-9,
      f"max|diff|={hdev:.3e} over 7 arms x 2 quantities")
report["headline_published"] = head

# --------------------------------------------------------------------------
# 1. load the 63 window tables (7 arms x 3 seeds x 3 epochs)
# --------------------------------------------------------------------------
def load_cell(arm, seed, ep):
    d = load(AX / "windows" / f"{arm}_s{seed}_ep{ep}.windows.json")
    arr = np.zeros((d["n_windows"], 4), dtype=np.float64)
    for i, w in enumerate(d["windows"]):
        c = w[THR]
        arr[i, 0] = c["hits"]
        arr[i, 1] = c["misses"]
        arr[i, 2] = c["false_alarms"]
        arr[i, 3] = c["correct_negatives"]
    return d, arr


cells = {}
n_win_seen = set()
keys_ref = None
keys_same = True
for ep in EPOCHS:
    for arm in ARMS:
        for s in SEEDS:
            d, arr = load_cell(arm, s, ep)
            cells[(arm, s, ep)] = arr
            n_win_seen.add(d["n_windows"])
            if keys_ref is None:
                keys_ref = list(d["window_keys"])
            elif list(d["window_keys"]) != keys_ref:
                keys_same = False
check("epoch_axis_cells_63_present", len(cells) == 63, f"{len(cells)}/63")
check("window_tables_538_windows_each",
      n_win_seen == {538}, f"n_windows values={sorted(n_win_seen)}")
check("window_keys_identical_across_63_cells", keys_same, "ordered window_keys")

# --------------------------------------------------------------------------
# 2. per-cell pooled counts / CSI, ep00 rerun-vs-published check
# --------------------------------------------------------------------------
pooled = {}
for k, arr in cells.items():
    h, m, f, cn = arr.sum(axis=0)
    pooled[k] = {"h": h, "m": m, "f": f, "cn": cn, "csi": csi(h, m, f)}

ep00_diff = 0.0
for arm in ARMS:
    for s in SEEDS:
        pub = float(summ[s][f"{arm}_s{s}_ep00"]["csi_at_2.0"])
        ep00_diff = max(ep00_diff, abs(pooled[(arm, s, "00")]["csi"] - pub))
check("ep00_rerun_reproduces_published_csi", ep00_diff < 1e-12,
      f"21 cells, max|diff|={ep00_diff:.3e}")


# --------------------------------------------------------------------------
# 3. contingency-domain area equalisation (round-54 counterpart)
# --------------------------------------------------------------------------
def equalised_arrays(arm_arr, ctrl_arr):
    """Rescale the arm's forecast-event AREA to the control's (T=h+m invariant)."""
    a_h = arm_arr.sum(axis=0)
    c_h = ctrl_arr.sum(axis=0)
    A = a_h[0] + a_h[2]
    target = c_h[0] + c_h[2]
    if A <= 0 or target <= 0:
        return None, None
    sc = target / A
    h2 = arm_arr[:, 0] * sc
    f2 = arm_arr[:, 2] * sc
    m2 = (arm_arr[:, 0] + arm_arr[:, 1]) - h2
    cn2 = arm_arr[:, 3] + (arm_arr[:, 2] - f2)
    return np.column_stack([h2, m2, f2, cn2]), sc


def two_level_boot(arm_by_seed, ctrl_by_seed, rng, n_boot=N_BOOT):
    """Same structure as the published guard: resample seeds, then windows."""
    n_seeds = len(SEEDS)
    n_win = arm_by_seed[0].shape[0]
    si = rng.integers(0, n_seeds, (n_boot, n_seeds))
    diffs = np.zeros((n_boot, n_seeds))
    for k in range(n_seeds):
        for v in range(n_seeds):
            sel = np.nonzero(si[:, k] == v)[0]
            if sel.size == 0:
                continue
            ii = rng.integers(0, n_win, (sel.size, n_win))
            a = arm_by_seed[v][ii].sum(axis=1)
            c = ctrl_by_seed[v][ii].sum(axis=1)
            diffs[sel, k] = csi_v(a[:, 0], a[:, 1], a[:, 2]) - \
                csi_v(c[:, 0], c[:, 1], c[:, 2])
    obs = diffs.mean(axis=1)
    lo = float(np.percentile(obs, 2.5))
    hi = float(np.percentile(obs, 97.5))
    return {"ci_lo": lo, "ci_hi": hi,
            "crosses_zero": bool(lo <= 0.0 <= hi),
            "n_boot": n_boot, "n_seeds": n_seeds, "n_windows": int(n_win),
            "ci_half_width": 0.5 * (hi - lo)}


# per-seed gains
gains = {}
for ep in EPOCHS:
    for arm in SIX:
        raw_ps, eq_ps = {}, {}
        for s in SEEDS:
            ctrl_csi = pooled[(CTRL, s, ep)]["csi"]
            raw_ps[s] = pooled[(arm, s, ep)]["csi"] - ctrl_csi
            eq_arr, _sc = equalised_arrays(cells[(arm, s, ep)],
                                          cells[(CTRL, s, ep)])
            eq_ps[s] = None if eq_arr is None else \
                csi(*eq_arr.sum(axis=0)[:3]) - ctrl_csi
        gains[(arm, ep)] = {"raw_per_seed": raw_ps, "eq_per_seed": eq_ps}

# ---- instrument fidelity at ep00 against the published field-domain S3 ----
raw_dev, eq_sign, eq_dev = 0.0, 0, []
for arm in SIX:
    raw_pt = mean([gains[(arm, "00")]["raw_per_seed"][s] for s in SEEDS])
    eq_vals = [gains[(arm, "00")]["eq_per_seed"][s] for s in SEEDS]
    if any(v is None for v in eq_vals):
        continue
    eq_pt = mean(eq_vals)
    raw_dev = max(raw_dev, abs(raw_pt - float(s3["arms"][arm]["raw"]["point"])))
    pub_eq = float(s3["arms"][arm]["equal_bias"]["point"])
    if (eq_pt > 0) == (pub_eq > 0):
        eq_sign += 1
    eq_dev.append(abs(eq_pt - pub_eq))
check("raw_points_reproduce_published_s3_at_ep00", raw_dev < 1e-9,
      f"max|diff|={raw_dev:.3e} over 6 arms")
check("eq_instrument_sign_agreement_with_published_s3_at_ep00",
      eq_sign == len(SIX),
      f"{eq_sign}/{len(SIX)} signs agree; median |diff|={median(eq_dev):.6f}")
report["instrument_fidelity_ep00"] = {
    "raw_point_max_abs_diff_vs_published": raw_dev,
    "eq_sign_agreement": f"{eq_sign}/{len(SIX)}",
    "eq_median_abs_diff": median(eq_dev),
}

# --------------------------------------------------------------------------
# 4. per-epoch guard: arm-level two-level bootstrap + published verdict logic
# --------------------------------------------------------------------------
def verdict_of(raw, eq):
    """Copied verbatim from tools/_e41_s3_equalbias.py (main())."""
    gained_raw = raw["point"] > 0
    survived = (not eq["crosses_zero"]) and eq["point"] > 0
    if survived:
        return "GAIN_SURVIVES_EQUALISATION"
    if gained_raw:
        return "GAIN_WAS_AREA_OR_AMPLITUDE"
    return "NO_GAIN_TO_BEGIN_WITH"


rng = np.random.default_rng(RNG_SEED)
epoch_rows = {}
for ep in EPOCHS:
    rows = {}
    for arm in SIX:
        arm_raw = [cells[(arm, s, ep)] for s in SEEDS]
        ctl_raw = [cells[(CTRL, s, ep)] for s in SEEDS]
        arm_eq = []
        for s in SEEDS:
            eq_arr, _sc = equalised_arrays(cells[(arm, s, ep)],
                                           cells[(CTRL, s, ep)])
            arm_eq.append(eq_arr)
        raw_boot = two_level_boot(arm_raw, ctl_raw, rng)
        eq_boot = two_level_boot(arm_eq, ctl_raw, rng)
        raw_ps = gains[(arm, ep)]["raw_per_seed"]
        eq_ps = gains[(arm, ep)]["eq_per_seed"]
        raw_row = dict(raw_boot)
        raw_row["point"] = mean([raw_ps[s] for s in SEEDS])
        eq_row = dict(eq_boot)
        eq_row["point"] = (mean([eq_ps[s] for s in SEEDS])
                           if all(eq_ps[s] is not None for s in SEEDS) else None)
        rows[arm] = {
            "raw": raw_row,
            "eq": eq_row,
            "raw_per_seed": raw_ps,
            "eq_per_seed": eq_ps,
            "raw_median_per_seed": median([raw_ps[s] for s in SEEDS]),
            "eq_median_per_seed": (median([eq_ps[s] for s in SEEDS])
                                   if eq_row["point"] is not None else None),
            "raw_same_sign_seeds": sum(
                1 for s in SEEDS if (raw_ps[s] > 0) == (raw_row["point"] > 0)),
            "eq_same_sign_seeds": (sum(1 for s in SEEDS
                                       if (eq_ps[s] > 0) == (eq_row["point"] > 0))
                                   if eq_row["point"] is not None else None),
            "verdict": verdict_of(raw_row, eq_row),
        }
    epoch_rows[ep] = rows
report["epoch_guard"] = epoch_rows

# ---- instrument fidelity: verdict agreement + interval width at ep00 ----
verdict_agree, conflicts = 0, []
raw_width_ratio, raw_cz_agree = [], 0
for arm in SIX:
    mine = epoch_rows["00"][arm]["verdict"]
    theirs = s3["arms"][arm]["s3_verdict"]
    if mine == theirs:
        verdict_agree += 1
    else:
        conflicts.append({"arm": arm, "instrument": mine, "published": theirs,
                          "instrument_eq_point": epoch_rows["00"][arm]["eq"]["point"],
                          "instrument_eq_ci": [epoch_rows["00"][arm]["eq"]["ci_lo"],
                                               epoch_rows["00"][arm]["eq"]["ci_hi"]],
                          "published_eq_point": float(s3["arms"][arm]["equal_bias"]["point"]),
                          "published_eq_ci": [float(s3["arms"][arm]["equal_bias"]["ci_lo"]),
                                              float(s3["arms"][arm]["equal_bias"]["ci_hi"])]})
    pw = float(s3["arms"][arm]["raw"]["ci_hi"]) - float(s3["arms"][arm]["raw"]["ci_lo"])
    mw = epoch_rows["00"][arm]["raw"]["ci_hi"] - epoch_rows["00"][arm]["raw"]["ci_lo"]
    raw_width_ratio.append(mw / pw if pw > 0 else float("nan"))
    mine_cross = bool(epoch_rows["00"][arm]["raw"]["ci_lo"] <= 0.0
                       <= epoch_rows["00"][arm]["raw"]["ci_hi"])
    if mine_cross == bool(s3["arms"][arm]["raw"]["crosses_zero"]):
        raw_cz_agree += 1
check("raw_interval_width_tracks_published_at_ep00",
      all(0.9 <= r <= 1.1 for r in raw_width_ratio),
      "width ratios vs published: " +
      ", ".join(f"{a}={r:.3f}" for a, r in zip(SIX, raw_width_ratio)))
check("raw_crossing_zero_decision_matches_published_at_ep00",
      raw_cz_agree == len(SIX), f"{raw_cz_agree}/{len(SIX)} decisions agree")

# the equalised side of the counterpart is compared, not trusted: measure how
# far it sits from the published field-domain equalisation
eq_cmp = {}
for arm in SIX:
    ieq = epoch_rows["00"][arm]["eq"]
    peq = s3["arms"][arm]["equal_bias"]
    iw = ieq["ci_hi"] - ieq["ci_lo"]
    pw2 = float(peq["ci_hi"]) - float(peq["ci_lo"])
    eq_cmp[arm] = {
        "instrument_point": ieq["point"], "published_point": float(peq["point"]),
        "point_abs_ratio": abs(ieq["point"]) / abs(float(peq["point"]))
        if abs(float(peq["point"])) > 0 else None,
        "instrument_width": iw, "published_width": pw2,
        "width_ratio": iw / pw2 if pw2 > 0 else None,
        "instrument_crosses_zero": ieq["crosses_zero"],
        "published_crosses_zero": bool(peq["crosses_zero"]),
    }
report["instrument_equalised_side_comparison_ep00"] = eq_cmp
mags_larger = sum(1 for a in SIX
                  if eq_cmp[a]["point_abs_ratio"] is not None
                  and eq_cmp[a]["point_abs_ratio"] > 1.0)
check("instrument_equalised_magnitudes_systematically_larger",
      mags_larger >= len(SIX) - 1,
      f"{mags_larger}/{len(SIX)} arms have |eq point| larger than the published "
      f"field-domain equalisation (ratios: "
      + ", ".join(f"{a}={eq_cmp[a]['point_abs_ratio']:.2f}" for a in SIX) + ")")

# divergences must be confined to the equalised side and be of the known type
conflict_type_ok = True
for c in conflicts:
    ieq = epoch_rows["00"][c["arm"]]["eq"]
    peq = s3["arms"][c["arm"]]["equal_bias"]
    if not (ieq["crosses_zero"] is False and bool(peq["crosses_zero"]) is True):
        conflict_type_ok = False
    if c["instrument"] != "GAIN_SURVIVES_EQUALISATION":
        conflict_type_ok = False
check("instrument_divergence_confined_to_equalised_side_and_documented",
      conflict_type_ok,
      f"{len(conflicts)} divergence(s) at ep00, all of the known type "
      f"(instrument equalised interval excludes zero where the published one "
      f"crosses zero): {[c['arm'] for c in conflicts]}")
report["instrument_verdict_conflicts_ep00"] = conflicts
report["instrument_raw_width_ratio_ep00"] = dict(zip(SIX, raw_width_ratio))

# --------------------------------------------------------------------------
# 5. null model: the control arm with permuted seeds (seed noise only)
# --------------------------------------------------------------------------
nulls = {}
for ep in EPOCHS:
    perms = [(1, 0, 2), (0, 2, 1), (2, 1, 0)]
    hw, czc, pts = [], [], []
    for pi in perms:
        arm_side = [cells[(CTRL, pi[s], ep)] for s in SEEDS]
        ctl_side = [cells[(CTRL, s, ep)] for s in SEEDS]
        b = two_level_boot(arm_side, ctl_side, rng)
        hw.append(b["ci_half_width"])
        czc.append(b["crosses_zero"])
        per_seed = [pooled[(CTRL, pi[s], ep)]["csi"]
                    - pooled[(CTRL, s, ep)]["csi"] for s in SEEDS]
        pts.append(mean(per_seed))
    nulls[ep] = {"per_permutation_ci_half_width": hw,
                 "per_permutation_crosses_zero": czc,
                 "median_ci_half_width": median(hw),
                 "n_perms_ci_excluding_zero": sum(1 for x in czc if not x),
                 "per_permutation_point": pts,
                 "round68_published_floor": R68_FLOORS[ep],
                 "ratio_to_round68_floor": median(hw) / R68_FLOORS[ep]}
report["null_model_seed_permutation"] = nulls
check("null_model_produces_no_false_positive_at_any_epoch",
      all(nulls[ep]["n_perms_ci_excluding_zero"] == 0 for ep in EPOCHS),
      "; ".join(f"ep{ep}:{nulls[ep]['n_perms_ci_excluding_zero']}/3" for ep in EPOCHS))
check("null_floor_cross_validates_round68",
      all(0.95 <= nulls[ep]["ratio_to_round68_floor"] <= 1.05 for ep in EPOCHS),
      "; ".join(f"ep{ep}: {nulls[ep]['median_ci_half_width']:.6f} vs "
                f"{R68_FLOORS[ep]:.6f}" for ep in EPOCHS))

# --------------------------------------------------------------------------
# 6. headline measurements (not criteria)
# --------------------------------------------------------------------------
survivors = [(ep, arm) for ep in EPOCHS for arm in SIX
             if epoch_rows[ep][arm]["verdict"] == "GAIN_SURVIVES_EQUALISATION"]
raw_sig = [(ep, arm) for ep in EPOCHS for arm in SIX
           if not epoch_rows[ep][arm]["raw"]["crosses_zero"]]
report["measurement_arm_surviving_equalisation"] = survivors
report["measurement_raw_interval_excluding_zero"] = raw_sig
report["measurement_note"] = (
    "The two lists above are MEASUREMENTS of this round's instrument, not "
    "criteria.  The published guard is authoritative and is unchanged: 12/12 "
    "intervals cross zero, verdicts 4 x GAIN_WAS_AREA_OR_AMPLITUDE + 2 x "
    "NO_GAIN_TO_BEGIN_WITH, qualified set empty.")

reason_rows = {ep: {arm: epoch_rows[ep][arm]["verdict"] for arm in SIX}
               for ep in EPOCHS}
report["verdict_reason_per_epoch"] = reason_rows
report["verdict_reason_moves_vs_ep00"] = [
    (ep, a) for ep in EPOCHS for a in SIX
    if epoch_rows[ep][a]["verdict"] != epoch_rows["00"][a]["verdict"]]

per_epoch_head = {}
for ep in EPOCHS:
    per_epoch_head[ep] = {}
    for arm in ARMS:
        vals = [pooled[(arm, s, ep)]["csi"] for s in SEEDS]
        bias = [(pooled[(arm, s, ep)]["h"] + pooled[(arm, s, ep)]["f"])
                / max(pooled[(arm, s, ep)]["h"] + pooled[(arm, s, ep)]["m"], 1e-12)
                for s in SEEDS]
        per_epoch_head[ep][arm] = {
            "median_csi_at_2.0": median(vals),
            "per_seed": vals,
            "pass_rate_over_floor": sum(1 for v in vals if v >= FLOOR) / 3.0,
            "median_bias_at_2.0": median(bias),
        }
report["per_epoch_headline"] = per_epoch_head

conv = {}
for ep in EPOCHS:
    c_med = per_epoch_head[ep][CTRL]["median_csi_at_2.0"]
    conv[ep] = {}
    for arm in SIX:
        r = epoch_rows[ep][arm]
        conv[ep][arm] = {
            "paired_raw_median_of_per_seed": r["raw_median_per_seed"],
            "paired_raw_mean_of_per_seed": r["raw"]["point"],
            "difference_raw_median_minus_control_median":
                per_epoch_head[ep][arm]["median_csi_at_2.0"] - c_med,
            "paired_eq_median_of_per_seed": r["eq_median_per_seed"],
            "raw_same_sign_seeds": r["raw_same_sign_seeds"],
            "eq_same_sign_seeds": r["eq_same_sign_seeds"],
        }
report["conventions"] = conv

# control collapse context
collapse = {}
for ep in EPOCHS:
    b0 = per_epoch_head[ep][CTRL]["median_csi_at_2.0"]
    collapse[ep] = {
        "control_median_csi": b0,
        "control_median_bias": per_epoch_head[ep][CTRL]["median_bias_at_2.0"],
        "control_collapse_ratio_vs_ep00": b0 / per_epoch_head["00"][CTRL]["median_csi_at_2.0"],
        "arms_above_control_median": [a for a in SIX
                                      if per_epoch_head[ep][a]["median_csi_at_2.0"] > b0],
    }
report["control_collapse_context"] = collapse

# --------------------------------------------------------------------------
# 7. write report
# --------------------------------------------------------------------------
n_pass = sum(1 for c in checks if c["ok"])
report["checks"] = checks
report["n_checks"] = len(checks)
report["n_pass"] = n_pass
report["verdict"] = "PASS" if n_pass == len(checks) else "FAIL"
report["note"] = ("Post-hoc diagnostic on the epoch axis. Creates no criterion, "
                  "changes no verdict. E45's published verdict (six arms negative, "
                  "empty qualified set, no ten-seed spend) is untouched; "
                  "docs/MASTER.md is not bumped.")

# read-only proof: published artefacts untouched by this analysis
check("analysis_left_published_artefacts_untouched",
      md5_file(EV / "scores_summary.json") == ss_md5
      and md5_file(EV / "s3_equalbias.json") == s3_md5,
      "md5 of scores_summary.json and s3_equalbias.json re-verified after run")
n_pass = sum(1 for c in checks if c["ok"])
report["n_checks"] = len(checks)
report["n_pass"] = n_pass
report["verdict"] = "PASS" if n_pass == len(checks) else "FAIL"

with open(OUTP, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)

print("=" * 78)
print(f"E45 ROUND 69 -- epoch axis x S3 equal-bias guard   VERDICT: {report['verdict']}"
      f"  ({n_pass}/{len(checks)})")
print("=" * 78)
for c in checks:
    print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")

print("\n-- instrument fidelity at ep00 (vs published field-domain S3) --")
print(f"  raw point max|diff| = {raw_dev:.3e} | raw width ratios: "
      + ", ".join(f"{a}={r:.3f}" for a, r in zip(SIX, raw_width_ratio)))
print(f"  eq sign agreement {eq_sign}/6  eq median |diff| = {median(eq_dev):.6f}")
print(f"  verdict agreement {verdict_agree}/6; conflicts: {conflicts}")

print("\n-- per-epoch median CSI@2.0 (pass rate over floor 0.0924) --")
for ep in EPOCHS:
    print(f"  ep{ep}: " + "  ".join(
        f"{a}={per_epoch_head[ep][a]['median_csi_at_2.0']:.6f}"
        f"({per_epoch_head[ep][a]['pass_rate_over_floor']:.2f})" for a in ARMS))

print("\n-- verdict reason per epoch (this instrument, published logic) --")
for ep in EPOCHS:
    print(f"  ep{ep}: " + "  ".join(
        f"{a}={epoch_rows[ep][a]['verdict'][:24]}" for a in SIX))

print("\n-- arm-level raw / equalised points and intervals (thr=2.0) --")
for ep in EPOCHS:
    print(f"  == ep{ep} ==  null CI half-width {nulls[ep]['median_ci_half_width']:.6f}"
          f" (round68 floor {R68_FLOORS[ep]:.6f}, ratio "
          f"{nulls[ep]['ratio_to_round68_floor']:.3f})  "
          f"null CIs excluding zero {nulls[ep]['n_perms_ci_excluding_zero']}/3")
    for a in SIX:
        r = epoch_rows[ep][a]
        eqp = r["eq"]["point"]
        eqp = float("nan") if eqp is None else eqp
        print(f"     {a:4s} raw {r['raw']['point']:+.6f} "
              f"[{r['raw']['ci_lo']:+.6f},{r['raw']['ci_hi']:+.6f}] "
              f"cross={str(r['raw']['crosses_zero']):5s} "
              f"same-sign {r['raw_same_sign_seeds']}/3 | "
              f"eq {eqp:+.6f} [{r['eq']['ci_lo']:+.6f},{r['eq']['ci_hi']:+.6f}] "
              f"cross={str(r['eq']['crosses_zero']):5s} "
              f"same-sign {r['eq_same_sign_seeds']}/3")

print("\n-- paired vs difference convention (raw) --")
for ep in EPOCHS:
    print(f"  ep{ep}: " + "  ".join(
        f"{a}:{conv[ep][a]['paired_raw_median_of_per_seed']:+.6f}/"
        f"{conv[ep][a]['difference_raw_median_minus_control_median']:+.6f}"
        for a in SIX))

print("\n-- control collapse context --")
for ep in EPOCHS:
    c = collapse[ep]
    print(f"  ep{ep}: control median CSI {c['control_median_csi']:.6f} "
          f"(x{c['control_collapse_ratio_vs_ep00']:.3f} of ep00), "
          f"median BIAS {c['control_median_bias']:.4f}, "
          f"arms above control: {c['arms_above_control_median']}")

print(f"\nmeasurements: survivors={survivors}  raw-interval-excluding-zero={raw_sig}")
print(f"verdict-reason moves vs ep00: {report['verdict_reason_moves_vs_ep00']}")
print(f"\nreport -> {OUTP}")
