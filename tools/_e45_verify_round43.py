"""E45 round-43 verification (read-only, local only).

What rounds 9-42 did NOT check
------------------------------
They re-read the artefacts, re-derived the S3 verdict, checked the
numbers printed in the documents and checked the source-level status of
the three registered implementation defects.  What none of them ever did
is decompose a DERIVED BOOLEAN READ-OUT into the branches that produced
it.

``scale_is_constant`` is such a derived boolean.  The probe's own
docstring defines it (tools/_e45_mechanism_probe.py, l.63-64 and
l.679-681):

    scale_is_constant is a DEGENERATION FLOOR, not a target:
    scale_cv < 0.05 OR scale_truth_corr < 0.3 means the amount channel
    has stopped carrying information about the rain.

It is therefore a DISJUNCTION of two independent branches:

    CV_BRANCH    : scale_cv < 0.05        -> the amount stopped varying
                   (literally "degenerated into a constant")
    CORR_BRANCH  : scale_truth_corr < 0.3 -> the amount still varies but
                   no longer tracks the rain

The E45 conclusion prose reported the A4b seed-2 cell as
"退化成常数" (degenerated into a constant) -- which asserts the CV
branch fired.  This round re-derives, from the raw probe artefacts,
WHICH branch actually fired in every cell of the scale family.

The findings
------------
1.  The flag is reproducible from the two floor values in all 6 cells.
2.  The CV branch NEVER fires in any cell of E45 (min scale_cv over the
    six cells is 1.2358, i.e. 24.7x the 0.05 floor).  The only branch
    that ever fires is the CORR branch, and only in A4b seed 2.
3.  Hence "量通道退化成常数" is NOT observed anywhere in E45.  In
    A4b seed 2 the amount channel is in fact the MOST variable of the
    six cells (scale_cv 3.0188, the maximum) while its mean has
    collapsed to ~1/10 of the other two seeds.
4.  The pre-registered consequence for A4b (E45 doc §3.1b/§3.2) is the
    correlation-flavoured one ("既有解码头的平均雨强不足以承担量
    通道") -- so the conclusion prose contradicts the very rule it
    cites.
5.  The verdict is UNCHANGED: A4b's flag is true at seed 2, so the
    pre-registered discard fires regardless of which branch produced
    it; six arms stay negative and no arm enters the 10-seed line.

Read-only.  Writes nothing, submits nothing, touches no remote state.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EV = ROOT / "tools" / "_evals" / "e45"

FAILS = []
WARNS = []
CHECKS = {"POS": 0, "NEG": 0}


def ok(msg):
    print(f"  PASS  {msg}")
    CHECKS["POS"] += 1


def bad(msg):
    FAILS.append(msg)
    print(f"  FAIL  {msg}")
    CHECKS["NEG"] += 1


def info(msg):
    print(f"  INFO  {msg}")


def load(name):
    p = EV / name
    if not p.exists():
        bad(f"missing artefact: {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# A. the flag is a disjunction -- and which branch fires where
# --------------------------------------------------------------------------
def check_scale_branches():
    print("\n=== A. scale_is_constant: which branch fires in each cell? ===")
    cells = []
    for s in range(3):
        d = load(f"diag_probe_s{s}.json")
        if d is None:
            return
        for arm in ("A4", "A4b"):
            r = d["arms"][arm]["readouts"]
            cells.append((arm, s, r))

    if len(cells) != 6:
        bad(f"expected 6 scale cells, got {len(cells)}")
        return

    print(f"\n  {'cell':7s}{'cv':>9s}{'corr':>9s}{'flag':>7s}"
          f"{'CVbranch':>10s}{'CORRbranch':>12s}   scale_mean")
    fired_cv, fired_corr = [], []
    for arm, s, r in cells:
        cv = r["scale_cv"]
        corr = r["scale_truth_corr"]
        flag = r["scale_is_constant"]
        b_cv = cv < r["scale_cv_floor"]
        b_corr = corr < r["scale_truth_corr_floor"]
        if b_cv:
            fired_cv.append(f"{arm}_s{s}")
        if b_corr:
            fired_corr.append(f"{arm}_s{s}")
        print(f"  {arm + '_s' + str(s):7s}{cv:9.4f}{corr:9.4f}"
              f"{str(flag):>7s}{str(b_cv):>10s}{str(b_corr):>12s}"
              f"      {r['scale_mean']:.5f}")
        # the flag must equal the disjunction of the two floors
        if bool(flag) != (b_cv or b_corr):
            bad(f"{arm}_s{s}: flag={flag} but cv-branch={b_cv} / "
                f"corr-branch={b_corr}")
        else:
            ok(f"{arm}_s{s}: flag == (cv<{r['scale_cv_floor']} OR "
               f"corr<{r['scale_truth_corr_floor']})")

    if fired_cv:
        bad(f"CV branch (literal 'degenerated into a constant') fired in "
            f"{fired_cv} -- the conclusion prose would be right")
    else:
        min_cv = min(r["scale_cv"] for _, _, r in cells)
        ok(f"CV branch fires in NO cell of E45 (min scale_cv = {min_cv:.4f}, "
           f"i.e. {min_cv / 0.05:.1f}x the 0.05 floor)")

    if fired_corr != ["A4b_s2"]:
        bad(f"CORR branch fired in {fired_corr}, expected exactly "
            f"['A4b_s2']")
    else:
        ok("CORR branch fires exactly once, in A4b seed 2")

    # A4b_s2 is the MOST variable cell, not the least
    a4b2 = [r for a, s, r in cells if a == "A4b" and s == 2][0]
    if a4b2["scale_cv"] == max(r["scale_cv"] for _, _, r in cells):
        ok(f"A4b_s2 has the HIGHEST scale_cv of the six cells "
           f"({a4b2['scale_cv']:.4f}) -- it is not a constant")
    else:
        bad("A4b_s2 does not have the highest scale_cv; re-check the "
            "'degenerated into a constant' prose")

    # and its amount has collapsed in magnitude
    means = {f"{a}_s{s}": r["scale_mean"] for a, s, r in cells}
    peers = [means["A4b_s0"], means["A4b_s1"]]
    ratio = means["A4b_s2"] / min(peers)
    if ratio < 0.15:
        ok(f"A4b_s2 scale_mean {means['A4b_s2']:.5f} is "
           f"{ratio:.3f}x the smaller A4b peer -- the amount COLLAPSED "
           f"in magnitude, it did not become constant")
    else:
        bad(f"A4b_s2 scale_mean ratio {ratio:.3f} is not < 0.15; the "
            f"'collapsed magnitude' reading needs re-checking")
    info("scale_max by cell: " + ", ".join(
        f"{a}_s{s}={r['scale_max']:.3f}" for a, s, r in cells))


# --------------------------------------------------------------------------
# B. the probe's own definition is information-based, not constancy-based
# --------------------------------------------------------------------------
def check_probe_definition():
    print("\n=== B. the probe's own definition of the flag ===")
    p = ROOT / "tools" / "_e45_mechanism_probe.py"
    if not p.exists():
        bad(f"probe source missing: {p}")
        return
    txt = p.read_text(encoding="utf-8")
    if "DEGENERATION FLOOR" in txt and "scale_cv < 0.05" in txt \
            and "scale_truth_corr < 0.3" in txt:
        ok("the probe defines the flag as a DEGENERATION FLOOR "
           "(cv<0.05 OR corr<0.3), i.e. 'stopped carrying information "
           "about the rain'")
    else:
        bad("the probe no longer carries the expected disjunction "
            "definition -- re-read it before trusting this round")
    if "stopped carrying information about the rain" in txt:
        ok("the probe's own wording is information-based "
           "('stopped carrying information about the rain')")


# --------------------------------------------------------------------------
# C. the pre-registered consequence for A4b is the correlation-flavoured one
# --------------------------------------------------------------------------
def check_prereg_consequence():
    print("\n=== C. the pre-registered consequence for A4b ===")
    p = (ROOT / "docs" / "experiments" /
         "E45-gate-direction-amount-separation-baseline.md")
    txt = p.read_text(encoding="utf-8")
    if "既有解码头的平均雨强" in txt and "不足以" in txt:
        ok("E45 §3.1b/§3.2 records the A4b consequence in "
           "correlation terms ('既有解码头的平均雨强不足以承担量通道')")
    else:
        bad("the E45 A4b consequence text changed -- re-read it")
    if "两条退化线之一被触发" in txt:
        ok("E45 §3.1b states the rule as 'one of the two degeneracy "
           "lines fired' -- the prose must not name a single line")


# --------------------------------------------------------------------------
# D. the verdict is untouched: S3 still says six arms negative
# --------------------------------------------------------------------------
def check_verdict_unchanged():
    print("\n=== D. the verdict is unaffected ===")
    s3 = load("s3_equalbias.json")
    if s3 is None:
        return
    arms = s3.get("arms", {})
    qualified = [n for n, a in arms.items()
                 if not (a.get("raw", {}).get("crosses_zero")
                         and a.get("equal_bias", {}).get("crosses_zero"))]
    if qualified:
        bad(f"arms whose S3 CI does not cross zero: {qualified}")
    else:
        ok("all 6 arms: raw AND equal-bias CI cross zero -> qualified "
           "set EMPTY, verdict unchanged")
    info("A4b_s2's scale_is_constant is true, so the pre-registered "
         "discard of A4b fires whichever branch produced it")


def main():
    print("=" * 78)
    print("E45 round-43 verification (read-only, local)")
    print("=" * 78)
    check_scale_branches()
    check_probe_definition()
    check_prereg_consequence()
    check_verdict_unchanged()
    print("\n" + "=" * 78)
    print(f"POS {CHECKS['POS']}   NEG {CHECKS['NEG']}")
    if FAILS:
        for f in FAILS:
            print(f"  - {f}")
        print("VERDICT: FAIL")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
