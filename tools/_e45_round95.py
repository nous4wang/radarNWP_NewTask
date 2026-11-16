"""Round-95 audit: E45 remote status re-check + a NEW drift finding.

What this round adds
---------------------
Rounds 71-75 (and every later round that reused the phrase "the minimal
sync set is still these 4 files") audited the LOCAL-vs-REMOTE drift of
the code that produced E45's published numbers.  The registered set was
FOUR files:

    models/scaled_structure/model.py
    models/topo_simvp/model.py
    tools/_e41_struct_metrics.py
    tools/_e45_mechanism_probe.py

That statement was correct when it was written (2026-11-15, v2.50): at
that moment those were exactly the files carrying the three fixes, and
the eval side had not been touched yet.

But the v2.52 round (2026-11-16) LANDED A FIFTH change -- the eval
side's ``_trained_model_kwargs`` restore in ``scripts/evaluate.py`` --
and nobody re-ran the drift audit afterwards.  So the registered
"minimal sync set = 4 files" is STALE: the true set is now FIVE.

This matters operationally, not cosmetically:

* the eval-side restore is precisely the half of defect (1) that makes
  ``clamp_mm`` survive to scoring time, and it is the half that CANNOT
  be recovered from a checkpoint (clamp_mm is a plain float attribute,
  it is not a weight, so it is not in ``state_dict``);
* ``tools/_e41_s3_equalbias.py`` -- the S3 guard -- is deliberately NOT
  in the set: its normalised md5 is identical on both sides, i.e. the
  guard that produced the published numbers is still the same
  implementation.

So this round re-derives the drift set from scratch, from the actual
file bytes, and reports the corrected count.

Everything else this round does is a re-derivation of what is already
published, to prove nothing moved:

* the headline table (median CSI/BIAS at 2.0 mm, floor pass rate) is
  recomputed from the 21 primary ``*.eval.json`` files and compared
  against ``scores_summary.json``;
* the mechanism criteria are re-judged cell by cell from the raw probe
  JSON;
* the S3 verdict is re-derived NUMERICALLY (``ci_lo <= 0 <= ci_hi``),
  not read off the stored boolean, and the seed-robustness sweep is
  re-read from ``s3_seedrobust.json``;
* the remote state (job states, md5 of every artifact, files modified
  after the close) was checked read-only over ``tools/remote.py``.

Read-only: this script touches nothing outside the repository and
writes only its own report.

VERDICT: PASS / FAIL
"""

from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "tools" / "_evals" / "e45"
RAW = EVALS / "evals_raw" / "evals"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SIX = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
FLOOR = 0.0924
ORACLE = 0.1653

# The remote md5 values measured read-only this round (2026-11-16 07:5x,
# GET via tools/remote.py).  BOTH sides are NORMALISED (CR stripped), so
# a Windows checkout and a Linux checkout of the same bytes compare equal.
# The earlier rounds also compared raw md5; for scripts/train.py the two
# differ only in line endings, which is why the normalisation matters.
REMOTE_MD5 = {
    "models/scaled_structure/model.py": "77f18b6cc2d5b4bf8d9078468d0db6c0",
    "models/topo_simvp/model.py": "f65554ca4f399431a7048c68bd07c80e",
    "scripts/evaluate.py": "0e31cf17035ed7f4ec3ce91f4ecef650",
    "scripts/train.py": "64d7e589a0edf7a5ea626d4588549f6d",
    "tools/_e45_mechanism_probe.py": "e7d3666a0c7495999649c40cc5a2fe95",
    "tools/_e41_struct_metrics.py": "742503c872ab0e68e096eedc1cc31085",
    "tools/_e41_s3_equalbias.py": "525f9c75910d5d9f0f3b2c4003d7549b",
    # controls: must NOT drift
    "models/amp_monotone_gate/model.py": "e90857d34dadfabe4ad7dff6b68d5394",
    "models/amp_hard_support/model.py": "66b33e3c5964236ffc2a11bd5e48ad63",
    "scripts/_e45_arms.sh": "5c5d84b99b5dd8e5132b12640d6be959",
    "configs/models/topo_simvp.yaml": "f62187e2a2ea2210b45fe959ff9ffeae",
    "configs/models/scaled_structure.yaml": "f2d16e661b1f7fe15e85812e6a6823f4",
}

# The five files that carry the three fixes (the corrected sync set).
EXPECTED_DRIFT = {
    "models/scaled_structure/model.py",      # defect (2): relative guard
    "models/topo_simvp/model.py",            # defect (1): clamp_mm
    "scripts/evaluate.py",                   # defect (1): eval-side restore
    "tools/_e45_mechanism_probe.py",         # defect (3): nan-aware median
    "tools/_e41_struct_metrics.py",          # defect (3): nan-aware median
}

CHECKS = []
REPORT = {}


def check(label, ok, detail=""):
    CHECKS.append((label, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + label + ((" :: " + detail) if detail else ""))
    return bool(ok)


def norm_md5(path: Path) -> str:
    """md5 of the file with CR stripped -- the comparison used all along."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.md5(data).hexdigest()


# --------------------------------------------------------------------------
# 1. the new finding: the drift set is FIVE files, not four
# --------------------------------------------------------------------------
def audit_drift():
    print("\n=== 1. local-vs-remote code drift (the NEW finding) ===")
    drifted, same = [], []
    for rel, remote in sorted(REMOTE_MD5.items()):
        p = ROOT / rel
        if not p.exists():
            check("file exists: %s" % rel, False)
            continue
        local = norm_md5(p)
        if local == remote:
            same.append(rel)
        else:
            drifted.append(rel)
    check("drift set == the five fix-carrying files",
          set(drifted) == EXPECTED_DRIFT,
          "drifted=%s" % sorted(drifted))
    check("controls did NOT drift (%d files)" % len(same),
          all(r not in EXPECTED_DRIFT for r in same),
          "identical=%s" % sorted(same))
    # the S3 guard must be the same implementation on both sides
    check("S3 guard tools/_e41_s3_equalbias.py is identical (not in the set)",
          "tools/_e41_s3_equalbias.py" in same)

    # the fix markers really are in the local bytes
    ss = (ROOT / "models/scaled_structure/model.py").read_text(encoding="utf-8")
    check("local scaled_structure uses the relative guard (safe_denom)",
          "safe_denom" in ss and "_SHAPE_EPS" in ss)
    check("local scaled_structure no longer divides by the absolute eps",
          "F_mm.mean(dim=(2, 3), keepdim=True) + _SHAPE_EPS" not in ss)
    topo = (ROOT / "models/topo_simvp/model.py").read_text(encoding="utf-8")
    check("local topo_simvp has the clamp_mm entry", "clamp_mm" in topo)
    ev = (ROOT / "scripts/evaluate.py").read_text(encoding="utf-8")
    check("local evaluate.py has the _trained_model_kwargs restore",
          "_trained_model_kwargs" in ev)
    probe = (ROOT / "tools/_e45_mechanism_probe.py").read_text(encoding="utf-8")
    check("local mechanism_probe median is nan-aware",
          "np.nanmedian" in probe and "n_valid_maxratio_batches" in probe)
    sm = (ROOT / "tools/_e41_struct_metrics.py").read_text(encoding="utf-8")
    check("local struct_metrics median is nan-aware", "np.nanmedian" in sm)

    # the STALE claim: the registered text still says four
    master = (ROOT / "docs" / "MASTER.md").read_text(encoding="utf-8")
    stale = "最小同步集仍是这 4 个文件" in master
    check("MASTER still claims a 4-file minimal sync set (the stale text)",
          stale, "found=%s" % stale)
    REPORT["drift_set"] = sorted(drifted)
    REPORT["identical_set"] = sorted(same)
    REPORT["stale_claim_in_master"] = stale
    return drifted


# --------------------------------------------------------------------------
# 2. headline table, recomputed from the 21 primary eval files
# --------------------------------------------------------------------------
def audit_headline():
    print("\n=== 2. headline table recomputed from the 21 primary eval files ===")
    summary = json.loads((EVALS / "scores_summary.json").read_text(encoding="utf-8"))
    max_dev = 0.0
    table = {}
    for arm in ARMS:
        csis, biases = [], []
        for seed in (0, 1, 2):
            f = RAW / ("%s_s%d_ep00.eval.json" % (arm, seed))
            if not f.exists():
                check("eval file exists: %s" % f.name, False)
                continue
            j = json.loads(f.read_text(encoding="utf-8"))
            th = j["thresholds"]["2.0"]
            csis.append(float(th["csi"]))
            biases.append(float(th["bias"]))
        med_csi = sorted(csis)[1]
        med_bias = sorted(biases)[1]
        passes = sum(1 for c in csis if c >= FLOOR)
        pub = summary[arm]
        # pass_rate_over_floor is stored as "2/3"
        pr = str(pub["pass_rate_over_floor"])
        pub_pass = int(pr.split("/")[0])
        max_dev = max(max_dev,
                      abs(med_csi - float(pub["median_csi_at_2.0"])),
                      abs(med_bias - float(pub["median_bias_at_2.0"])))
        for s, c in zip(("s0", "s1", "s2"), csis):
            max_dev = max(max_dev, abs(c - float(pub["csi_at_2.0"][s])))
        check("%s: pass rate matches published" % arm,
              passes == pub_pass,
              "recomputed %d vs published %s" % (passes, pr))
        table[arm] = {"median_csi": med_csi, "median_bias": med_bias,
                      "pass_rate": passes, "csis": csis, "biases": biases}
    check("headline table reproduces the published values (max|dev| == 0)",
          max_dev < 1e-12, "max|dev| = %.3e" % max_dev)

    # the promotion question: only the floor rate, and the control itself
    three = [a for a in ARMS if table[a]["pass_rate"] == 3]
    check("arms passing the floor on 3/3 seeds: %s" % three,
          set(three) == {"A3a", "A3b"}, "got %s" % three)
    for a in ("A3a", "A3b"):
        check("%s: median is below the control B0 (no promotion)" % a,
              table[a]["median_csi"] < table["B0"]["median_csi"],
              "%.6f vs B0 %.6f" % (table[a]["median_csi"], table["B0"]["median_csi"]))
    check("control B0 itself only passes the floor on 2/3 seeds",
          table["B0"]["pass_rate"] == 2)
    over = [a for a in ARMS if max(table[a]["csis"]) > ORACLE]
    check("arms whose best seed exceeds the Oracle bound: %s" % over,
          set(over) == {"A3b", "A4"}, "got %s" % over)
    REPORT["headline"] = table
    REPORT["headline_max_dev"] = max_dev
    return table


# --------------------------------------------------------------------------
# 3. mechanism criteria, cell by cell, from the raw probe JSON
# --------------------------------------------------------------------------
def audit_mechanism():
    print("\n=== 3. mechanism criteria re-judged per cell ===")
    probes = {}
    for s in (0, 1, 2):
        probes[s] = json.loads((EVALS / ("diag_probe_s%d.json" % s)).read_text(encoding="utf-8"))

    def rd(arm, seed, key):
        return probes[seed]["arms"][arm]["readouts"].get(key)

    # A1 / A2 : gate direction
    for arm in ("A1", "A2"):
        ok_c = sum(1 for s in (0, 1, 2) if float(rd(arm, s, "corr_mask_amp")) >= 0.0)
        ok_p = sum(1 for s in (0, 1, 2) if float(rd(arm, s, "mask_at_peak")) > 0.5)
        check("%s: corr(mask,amp) >= 0 on %d/3 seeds" % (arm, ok_c), ok_c == 3)
        check("%s: mask_at_peak > 0.5 on %d/3 seeds" % (arm, ok_p), ok_p == 3)
    # A3a / A3b : topology.  The pre-registered criteria are ARM-SPECIFIC
    # (v2.41): A3a's primary read-out is the field maximum, A3b's is the
    # gradient concentration; both lines must be reported for both arms.
    # A3a's own discarding trigger is "field max > 30 mm" OR
    # "tail_max_over_p99 > 10"; A3b's is "grad_top1pct_share >= 0.5" OR
    # "field max > 30 mm".
    for arm in ("A3a", "A3b"):
        ok_f = sum(1 for s in (0, 1, 2) if float(rd(arm, s, "field_max_mm")) <= 30.0)
        check("%s: field_max_mm <= 30 on %d/3 seeds (both arms must report it)"
              % (arm, ok_f), ok_f == 0)
    ok_g_b = sum(1 for s in (0, 1, 2) if float(rd("A3b", s, "grad_top1pct_share")) < 0.5)
    check("A3b: grad_top1pct_share < 0.5 on %d/3 seeds (its primary criterion)"
          % ok_g_b, ok_g_b == 0)
    # A3a's gradient is a secondary read-out; it passes on seed 2 only.
    # That is already published ("A3a 梯度集中 1/3"), so assert the number.
    ok_g_a = sum(1 for s in (0, 1, 2) if float(rd("A3a", s, "grad_top1pct_share")) < 0.5)
    check("A3a: grad_top1pct_share < 0.5 on %d/3 seeds (secondary line, "
          "published as 1/3)" % ok_g_a, ok_g_a == 1)
    # A3a's own discarding trigger: tail_max_over_p99 > 10
    tails = [float(rd("A3a", s, "tail_max_over_p99")) for s in (0, 1, 2)]
    check("A3a: tail_max_over_p99 > 10 on 3/3 seeds (A3a discarded by "
          "pre-registration)", all(t > 10.0 for t in tails),
          "values = %s" % [round(t, 3) for t in tails])
    tails_b = [float(rd("A3b", s, "tail_max_over_p99")) for s in (0, 1, 2)]
    check("A3b: tail_max_over_p99 > 10 on 3/3 seeds", all(t > 10.0 for t in tails_b),
          "values = %s" % [round(t, 3) for t in tails_b])
    check("A3a and A3b are both discarded -> the topology branch is closed",
          ok_f == 0 and ok_g_b == 0)
    # A4 / A4b : amount channel
    for arm in ("A4", "A4b"):
        ok_cv = sum(1 for s in (0, 1, 2) if float(rd(arm, s, "scale_cv")) >= 0.05)
        ok_cc = sum(1 for s in (0, 1, 2) if float(rd(arm, s, "scale_truth_corr")) >= 0.3)
        check("%s: scale_cv >= 0.05 on %d/3 seeds" % (arm, ok_cv), ok_cv == 3)
        check("%s: scale_truth_corr >= 0.3 on %d/3 seeds" % (arm, ok_cc),
              ok_cc == (3 if arm == "A4" else 2))
    # the derived boolean, split into its branches (v2.48 discipline)
    for arm in ("A4", "A4b"):
        for s in (0, 1, 2):
            cv = float(rd(arm, s, "scale_cv"))
            cc = float(rd(arm, s, "scale_truth_corr"))
            const = bool(rd(arm, s, "scale_is_constant"))
            cv_branch = cv < 0.05
            corr_branch = cc < 0.3
            check("%s s%d: scale_is_constant == (cv<0.05 OR corr<0.3)" % (arm, s),
                  const == (cv_branch or corr_branch),
                  "cv=%.4f corr=%.4f const=%s" % (cv, cc, const))
            check("%s s%d: the cv branch never fired (min cv is 24.7x the line)"
                  % (arm, s), not cv_branch, "cv=%.4f" % cv)
    # the implementation defect stays an implementation defect
    fails = [(a, s) for a in ("A4", "A4b") for s in (0, 1, 2)
             if not bool(rd(a, s, "mean_out_equals_T"))]
    check("mean(out)=T still fails on %d/6 cells -- an implementation "
          "defect, not a mechanism verdict" % len(fails),
          len(fails) == 4, "failing cells = %s" % fails)
    # the field maximum is the model clamp, on every cell
    fmax = [float(rd(a, s, "field_max_mm")) for a in ("A3a", "A3b") for s in (0, 1, 2)]
    check("all six topology cells sit at the model clamp (402.4288 mm)",
          all(abs(v - 402.4288) < 1e-3 for v in fmax),
          "values = %s" % sorted(set(round(v, 4) for v in fmax)))
    REPORT["mean_out_T_failing_cells"] = fails


# --------------------------------------------------------------------------
# 4. the S3 verdict, derived numerically
# --------------------------------------------------------------------------
def audit_s3():
    print("\n=== 4. S3 verdict, derived numerically ===")
    s3 = json.loads((EVALS / "s3_equalbias.json").read_text(encoding="utf-8"))
    check("S3 guard parameters unchanged (thr 2.0 / control B0 / n_boot 10000)",
          float(s3["threshold_mm"]) == 2.0 and s3["control"] == "B0"
          and int(s3["n_boot"]) == 10000)
    for arm in SIX:
        a = s3["arms"][arm]
        for half in ("raw", "equal_bias"):
            h = a[half]
            lo, hi, pt = float(h["ci_lo"]), float(h["ci_hi"]), float(h["point"])
            contains = lo <= 0.0 <= hi
            check("%s/%s: interval numerically contains zero" % (arm, half),
                  contains, "[%+.6f, %+.6f] point %+.6f" % (lo, hi, pt))
            check("%s/%s: stored flag agrees with the numbers" % (arm, half),
                  bool(h["crosses_zero"]) == contains)
    # the two failure-mode branches must be reported separately
    branches = {}
    for arm in SIX:
        branches[arm] = s3["arms"][arm].get("s3_verdict") or s3["arms"][arm].get("verdict")
    kinds = {}
    for arm, v in branches.items():
        kinds.setdefault(str(v), []).append(arm)
    check("the failure mode splits into exactly two registered branches",
          len(kinds) == 2, "branches = %s" % {k: v for k, v in kinds.items()})
    REPORT["s3_branches"] = kinds

    # seed robustness, read back from the sweep artifact
    sr = json.loads((EVALS / "s3_seedrobust.json").read_text(encoding="utf-8"))
    check("the sweep artifact carries the guard's module md5",
          sr.get("module_md5") == REMOTE_MD5["tools/_e41_s3_equalbias.py"],
          "module_md5=%s" % sr.get("module_md5"))
    check("the sweep covers 20 alternative seeds", len(sr["alt_seeds"]) == 20)
    qualifying = 0
    zeros = 0
    total = 0
    for arm in SIX:
        for half, key in (("raw", "raw_sweep"), ("equal_bias", "sweep")):
            for cell in sr["arms"][arm][key]:
                lo, hi, pt = float(cell["ci_lo"]), float(cell["ci_hi"]), float(cell["point"])
                total += 1
                if lo <= 0.0 <= hi:
                    zeros += 1
                if pt > 0.0 and not (lo <= 0.0 <= hi):
                    qualifying += 1
    check("every arm/half/seed interval contains zero (%d/%d cells)" % (zeros, total),
          zeros == total)
    check("qualifying seeds (point > 0 AND interval excludes zero) == 0",
          qualifying == 0, "qualifying = %d" % qualifying)
    # the canonical seed must reproduce the published numbers bit for bit
    max_dev = 0.0
    for arm in SIX:
        for half, key in (("raw", "raw_reproduced"), ("equal_bias", "reproduced")):
            pub = s3["arms"][arm][half]
            got = sr["arms"][arm][key]
            for f in ("point", "ci_lo", "ci_hi"):
                max_dev = max(max_dev, abs(float(pub[f]) - float(got[f])))
    check("the canonical seed reproduces the published S3 numbers (max|dev| == 0)",
          max_dev < 1e-12, "max|dev| = %.3e" % max_dev)
    check("every arm's equalisation gap is 0",
          all(float(sr["arms"][a]["max_abs_bias_gap"]) == 0.0 for a in SIX))
    REPORT["s3_zero_cells"] = "%d/%d" % (zeros, total)
    REPORT["s3_qualifying"] = qualifying
    REPORT["s3_reproduction_max_dev"] = max_dev


def main():
    print("=" * 72)
    print("E45 round-95 audit -- remote status re-check + drift-set correction")
    print("=" * 72)
    audit_drift()
    audit_headline()
    audit_mechanism()
    audit_s3()

    n = len(CHECKS)
    npass = sum(1 for _, ok, _ in CHECKS if ok)
    verdict = "PASS" if npass == n else "FAIL"
    print("\n" + "=" * 72)
    print("POS %d/%d   VERDICT: %s" % (npass, n, verdict))
    print("=" * 72)

    out = EVALS / "round95_status_audit.json"
    out.write_text(json.dumps({
        "round": 95,
        "date": "2026-11-16",
        "verdict": verdict,
        "pos": "%d/%d" % (npass, n),
        "checks": [{"label": a, "ok": b, "detail": c} for a, b, c in CHECKS],
        "report": REPORT,
        "e45_verdict": ("six arms negative; no arm promoted to ten seeds; "
                        "no ten-seed compute"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("report -> %s" % out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
