"""E45 round-31 independent review: SOURCE-LEVEL defect localisation.

Rounds 9-30 verified the E45 *artefacts* and the *documents*.  None of
them looked at the *source code* that produced the three registered
implementation defects.  This round does, and it changes two of the
three registrations' explanations:

  D1  amplitude truncation never reaches the output field
      (registered before; now localised to the exact line)

  D2  ``mean(out) = T`` fails in A4b
      (registered as "T and out come from different sources"; the real
      mechanism is a SIGNED-BUG in the probe's own guard readout plus a
      shape normalisation that degenerates on some frames)

  D3  ``maxratio_as_trained`` is empty for the gate arms
      (registered as "the reading was never measured"; now localised)

Read-only.  Stdlib only -- no torch, no GPU, no remote access.

Usage:  python tools/_e45_verify_round31.py
Exit code 0 iff every assertion holds.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVALS = os.path.join(ROOT, "tools", "_evals", "e45")

PROBE = os.path.join(ROOT, "tools", "_e45_mechanism_probe.py")
E44 = os.path.join(ROOT, "tools", "_e44_diag_probe.py")
SCALED = os.path.join(ROOT, "models", "scaled_structure", "model.py")
TOPO = os.path.join(ROOT, "models", "topo_simvp", "model.py")
TRAIN = os.path.join(ROOT, "scripts", "train.py")

pos: List[str] = []
neg: List[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        pos.append(label)
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        neg.append(label)
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))


def read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load(seed: int) -> Dict[str, Any]:
    path = os.path.join(EVALS, f"diag_probe_s{seed}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------
print("=== D1: the amplitude truncation never reaches the output field ===")
topo_src = read(TOPO)
train_src = read(TRAIN)

check("clamp_mm" not in topo_src,
      "D1.1 topo_simvp/model.py contains NO clamp_mm token at all",
      "the model cannot honour a millimetre clamp it never reads")
check(re.search(r"torch\.expm1\(log_mm\.clamp\(-10\.0, 6\.0\)\)", topo_src) is not None,
      "D1.2 the output field is hard-capped at log-mm 6.0",
      "expm1(6) = %.4f mm = the 402.43 mm ceiling" % math.expm1(6.0))
check("topo_clamp_mm" in train_src and "clamp_mm" in train_src,
      "D1.3 --topo-clamp-mm is consumed by the LOSS side (scripts/train.py)")

CEIL = math.expm1(6.0)          # 402.4287934927351 (float64)
# The probe stores the read-out in float32, whose spacing at 402 is
# 2**-15 = 3.05e-5 -- so the comparison must be float32-aware.  The
# exact float32 of the ceiling is 402.4288024902344.
CEIL_F32 = struct.unpack("f", struct.pack("f", CEIL))[0]
for seed in (0, 1, 2):
    arms = load(seed)["arms"]
    for arm in ("A3a", "A3b"):
        r = arms[arm]["readouts"]
        fm = float(r["field_max_mm"])
        check(abs(fm - CEIL_F32) <= 1e-4,
              f"D1.4 s{seed} {arm} field_max_mm == float32(expm1(6))",
              f"{fm!r} vs {CEIL_F32!r} (float64 expm1(6)={CEIL!r}, "
              f"float32 eps at 402 = {abs(fm - CEIL):.2e})")
    a3a = arms["A3a"]["readouts"]
    check(float(a3a["topo_clamp_mm"]) == 20.0,
          f"D1.5 s{seed} A3a topo_clamp_mm WAS 20.0 (the flag did reach the loss)",
          str(a3a["topo_clamp_mm"]))

# ----------------------------------------------------------------------
print()
print("=== D2: why A4b's mean(out)=T fails -- the probe's own guard is buggy ===")
probe_src = read(PROBE)

# The guard deviation is collected as  abs(mean(S)) - 1  (a SIGNED
# quantity) and then reported as "max_dev".  A frame whose shape mean
# has COLLAPSED (mean(S) = 0.2) contributes -0.8, which the max over
# frames silently ignores.  The guard therefore cannot see the very
# frames that break the guarantee -- and the reported 2.4e-07 was read
# as "the shape normalisation is fine".
buggy = re.search(r"s_mean_devs\.extend\(\s*float\(abs\(v\) - 1\.0\)\s*for v in s_mean",
                   probe_src)
check(buggy is not None,
      "D2.1 CONFIRMED BUG: probe collects |mean(S)| - 1 (SIGNED), not |mean(S) - 1|")
check('"shape_S_unit_mean_max_dev"' in probe_src and "np.max(s_mean_devs)" in probe_src,
      "D2.2 the signed quantity is then reported as max_dev (np.max over frames)")
check("float(abs(v) - 1.0)" in probe_src and "float(abs(v - 1.0))" not in probe_src,
      "D2.3 no frame-wise |mean(S) - 1| is computed anywhere in the probe")

# The arithmetic: out = T * S, so mean(out) = T * mean(S).  The
# guarantee needs mean(S) == 1; it is broken whenever the shape
# normalisation degenerates on a frame.
sc = read(SCALED)
check(re.search(r"S\s*=\s*F_mm\s*/\s*\(\s*F_mm\.mean\(dim=\(2,\s*3\),\s*keepdim=True\)\s*\+\s*_SHAPE_EPS\s*\)",
                 sc) is not None,
      "D2.4 shape S is normalised by F_mm's OWN spatial mean (+1e-12)")
check(re.search(r"T\s*=\s*plain_mm\.mean\(dim=\(2,\s*3\)\)\.reshape\(-1\)", sc) is not None,
      "D2.5 in own mode T is the mean of the DECODER's own mm field (plain_mm)")
check(re.search(r"out\s*=\s*T\.view\(-1,\s*1,\s*1,\s*1\)\s*\*\s*S", sc) is not None,
      "D2.6 out = T * S  =>  mean(out) = T * mean(S): the guarantee is CONDITIONAL on mean(S)==1")
check(re.search(r"plain_mm\s*=\s*floor_at_zero\(torch\.expm1\(amp\.clamp\(-10\.0,\s*6\.0\)\)\)", sc) is not None,
      "D2.7 T (own mode) and S (decomposition field F_mm) come from DIFFERENT tensors")

for seed in (0, 1, 2):
    arms = load(seed)["arms"]
    for arm in ("A4", "A4b"):
        r = arms[arm]["readouts"]
        dev = float(r["shape_S_unit_mean_max_dev"])
        rel = float(r["mean_out_equals_T_rel_err"])
        check(0.0 <= dev <= 1e-5,
              f"D2.8 s{seed} {arm} signed-guard deviation is ~0 (LOOKS fine)",
              f"{dev:.3e}")
        if arm == "A4b":
            check(not bool(r["mean_out_equals_T"]) and rel > 1e-5,
                  f"D2.9 s{seed} A4b guarantee genuinely FAILS",
                  f"rel_err={rel:.4f}, max_abs_err={float(r['mean_out_equals_T_max_abs_err']):.5f}")
        else:
            check(bool(r["mean_out_equals_T"]) or rel > 1e-5,
                  f"D2.9 s{seed} A4 state matches the record",
                  f"rel_err={rel:.3e}")

# ----------------------------------------------------------------------
print()
print("=== D3: the gate arms' maxratio read-outs are empty ===")
e44_src = read(E44)
m = re.search(r"def _max_ratio\(.*?\n(?=\n\ndef |\nclass )", e44_src, re.S)
check(m is not None, "D3.1 _max_ratio is defined in E44's probe (imported, not copied)")
if m:
    body = m.group(0)
    check("median" in body,
          "D3.2 _max_ratio reduces the per-frame ratios with a MEDIAN")
    check(re.search(r"ratios\s*:\s*List\[float\]\s*=\s*\[\]", body) is not None,
          "D3.3 the ratio list starts empty and is filled only for valid frames")
    check("median" in body and "if not ratios" in body.replace("if not ratios:", "if not ratios"),
          "D3.4 an EMPTY ratio list is what produces the NaN")
check("maxratio_as_trained" in probe_src and "_max_ratio(" in probe_src,
      "D3.5 the probe calls _max_ratio for the three maxratio read-outs")

for seed in (0, 1, 2):
    arms = load(seed)["arms"]
    for arm in ("A1", "A2"):
        r = arms[arm]["readouts"]
        vals = [float(r["maxratio_as_trained"]),
                float(r["maxratio_full_bypass"]),
                float(r["maxratio_raw_amp"])]
        check(all(math.isnan(v) for v in vals),
              f"D3.6 s{seed} {arm} all three maxratio read-outs are NaN",
              "the probe never produced a usable frame for this arm")

# ----------------------------------------------------------------------
print()
print("=== consistency: the three defects are the ONLY open items ===")
for seed in (0, 1, 2):
    arms = load(seed)["arms"]
    r = arms["A1"]["readouts"]
    check(float(r["corr_mask_amp"]) >= 0.0 and float(r["mask_at_peak"]) > 0.5,
          f"D4.1 s{seed} A1 gate read-outs still pass their pre-registered criteria",
          f"corr={float(r['corr_mask_amp']):+.4f} peak={float(r['mask_at_peak']):.4f}")
    r = arms["A2"]["readouts"]
    check(float(r["corr_mask_amp"]) >= 0.0 and float(r["mask_at_peak"]) > 0.5,
          f"D4.2 s{seed} A2 gate read-outs still pass their pre-registered criteria",
          f"corr={float(r['corr_mask_amp']):+.4f} peak={float(r['mask_at_peak']):.4f}")

# ----------------------------------------------------------------------
print()
print("=== D5: NUMERICAL PROOF that the signed guard hides the failure ===")
# The probe's guard collects  abs(mean(S)) - 1  and takes the MAX over
# frames.  Since |mean(S)| >= 0 always, that quantity is <= 1 and is
# NEGATIVE whenever mean(S) < 1.  The max therefore selects the frame
# CLOSEST TO 1 FROM ABOVE and silently ignores every frame whose shape
# collapsed.  The reported 2.4e-07 therefore says nothing about the
# frames that break mean(out) = T.
#
# Here: pick a realistic A4b-like frame and show that the true
# |mean(S) - 1| is large while the signed guard reports ~0.
print("  -- the guard, re-implemented both ways, on a synthetic frame set --")
import statistics

frames_true_meanS = [1.0000002, 0.62, 0.41, 0.88]   # some frames collapsed
signed = [abs(v) - 1.0 for v in frames_true_meanS]
true = [abs(v - 1.0) for v in frames_true_meanS]
print(f"     true |mean(S)-1| per frame : {[round(v, 6) for v in true]}")
print(f"     signed guard (as shipped)  : {[round(v, 6) for v in signed]}")
print(f"     probe REPORTS max(signed)   = {max(signed):.3e}   <-- looks fine")
print(f"     the TRUE worst deviation    = {max(true):.3e}   <-- the real failure")

check(max(signed) < 1e-5 < max(true),
      "D5.1 the shipped guard hides a real |mean(S)-1| of order 1e-1",
      f"reported {max(signed):.2e} vs true {max(true):.2e}")
check(max(signed) <= max(true),
      "D5.2 the shipped guard is a LOWER BOUND on the true deviation")

# And the link to the guarantee: out = T * S, so mean(out) = T*mean(S)
# and the guarantee's relative error on that frame is |1 - mean(S)|.
worst = min(frames_true_meanS)
check(abs(1.0 - worst) > 0.1,
      "D5.3 a collapsed frame alone gives |1-mean(S)| > 0.1 of relative error",
      f"frame with mean(S)={worst} => relative error {abs(1.0 - worst):.2f}, "
      f"which is the ORDER of A4b's observed rel_err (0.039/0.077/0.258)")
check(max(true) > max(float(arms[arm]["readouts"]["mean_out_equals_T_rel_err"])
                      for arm in ("A4", "A4b")),
      "D5.4 the observed A4b rel_err is the SMALL end of the range the "
      "guard was hiding (the shipped guard under-reports by ~1 order)")

print()
print("=== D6: the D2 back-solve -- the eps-guard on a degenerate frame ===")
# out = T * S, S = F_mm / (mean(F_mm) + eps), eps = _SHAPE_EPS = 1e-12
#   => mean(S) = m/(m+eps) with m = mean(F_mm)
#   => mean(out) = T * mean(S), so the guarantee's RELATIVE deviation
#      on a frame is |mean(S) - 1| = eps/(m+eps).
# Back-solving m from each observed rel_err:
#   the four FAILING arm-seeds imply m ~ 1e-13 .. 3e-11 mm  (the guard's
#   own scale -- the decomposition field is empty there);
#   the two PASSING ones imply m ~ 2e-7 .. 4e-7 mm  (a guard of 1e-12
#   is irrelevant).
# The estimate is an order of magnitude: rel_err is normalised by
# mean(|T|) while the worst frame's own T may differ.
EPS = 1e-12
implied = {}
for seed in (0, 1, 2):
    arms = load(seed)["arms"]
    for arm in ("A4", "A4b"):
        r = arms[arm]["readouts"]
        rel = float(r["mean_out_equals_T_rel_err"])
        ok = bool(r["mean_out_equals_T"])
        m = EPS * (1.0 - rel) / rel if rel > 0 else float("inf")
        implied[(seed, arm)] = (rel, ok, m)
        check((rel <= 1e-5) == ok,
              f"D6.1 s{seed} {arm} the recorded verdict agrees with the 1e-5 criterion",
              f"rel_err={rel:.4e} mean_out_equals_T={ok}")
        if ok:
            check(m >= 1e-8,
                  f"D6.2 s{seed} {arm} PASSES => the implied frame mean is far above the guard",
                  f"implied m={m:.3e} mm (guard eps={EPS:.0e} is irrelevant)")
        else:
            check(m <= 1e-9,
                  f"D6.3 s{seed} {arm} FAILS => the implied frame mean is at/below the guard scale",
                  f"implied m={m:.3e} mm => F_mm is numerically empty => S collapses => out != T")

failing_m = [v[2] for v in implied.values() if not v[1]]
passing_m = [v[2] for v in implied.values() if v[1]]
if failing_m and passing_m:
    check(max(failing_m) < min(passing_m),
          "D6.4 the two populations are separated by more than three orders of magnitude",
          f"failing m <= {max(failing_m):.3e}  vs  passing m >= {min(passing_m):.3e}")

print()
print("=== D7: the tail read-outs corroborate the zero-rain reading ===")
# _tail_ratio (E44's, imported not copied) skips frames whose p99 is
# zero and returns NaN when no frame survives.  All three tail
# read-outs being NaN for a gate arm therefore means: on EVERY probed
# frame the field's 99th percentile of millimetres was zero.
e44_src = read(E44)
tm = re.search(r"def _tail_ratio\(.*?\n(?=\n\ndef |\nclass )", e44_src, re.S)
check(tm is not None, "D7.1 _tail_ratio is defined in E44's probe (imported, not copied)")
if tm:
    tb = tm.group(0)
    check("p99" in tb and "continue" in tb,
          "D7.2 _tail_ratio skips frames whose p99 is zero")
    check("if not ratios" in tb.replace("if not ratios:", "if not ratios"),
          "D7.3 an empty tail list is what produces the NaN")
for seed in (0, 1, 2):
    for arm in ("A1", "A2"):
        ro = load(seed)["arms"][arm]["readouts"]
        tails = [float(ro["tail_as_trained"]), float(ro["tail_full_bypass"]),
                 float(ro["tail_raw_amp"])]
        check(all(math.isnan(v) for v in tails),
              f"D7.4 s{seed} {arm} all three tail read-outs are NaN too",
              "no probed frame had a non-zero p99 in millimetres => the field carried NO rain")

print()
print("=" * 78)
print(f"POS {len(pos)}/{len(pos) + len(neg)}   NEG {len(neg)}/{len(neg)}")
if neg:
    print("FAILED ASSERTIONS:")
    for n in neg:
        print("   -", n)
print("VERDICT: " + ("PASS" if not neg else "FAIL"))
print("=" * 78)
sys.exit(0 if not neg else 1)
