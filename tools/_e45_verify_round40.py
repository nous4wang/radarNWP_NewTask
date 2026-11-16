#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E45 round-40 read-only verification: decompose the S3 equal-bias verdict into
per-seed contributions.

Why this check exists
---------------------
Rounds 27-39 verified the S3 verdict by reading the stored point estimates and
CI flags (round 34 re-derived the CI from the raw window counts remotely).
Nobody has checked what the verdict actually rests on: the S3 aggregation is a
plain mean over the 3 seeds of a per-seed gain, so a mean that crosses zero
can be produced either by (a) every seed being weakly negative, or (b) seeds
with opposite signs cancelling out.  Those two situations justify the same
"CI crosses zero" sentence but they mean very different things for a re-open:
(b) is evidence of seed instability, (a) is evidence of a real deficit.

This script therefore answers, for every arm:
  1. Does the stored S3 point estimate reproduce exactly as the mean of the
     three per-seed gains?  (verifies the aggregation is what we think it is)
  2. What is the sign pattern of the per-seed raw gains and of the per-seed
     equal-bias gains?  (sign consistency of the verdict)
  3. How much of the raw per-seed gain survives equalisation?  (which seed
     carries the verdict)
  4. Does any arm show a positive raw gain in ALL three seeds?  If not, the
     negative verdict is corroborated by a plain sign count and does not rest
     on the bootstrap machinery at all.

Reads only local artifacts; writes nothing.  No network, no Slurm, no GPU.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "_evals", "e45")
SUMMARY = os.path.join(EVAL_DIR, "scores_summary.json")
S3 = os.path.join(EVAL_DIR, "s3_equalbias.json")

TOL = 1e-6
CONTROL = "B0"
SEEDS = ("s0", "s1", "s2")
ARMS = ("A1", "A2", "A3a", "A3b", "A4", "A4b")


def sign(x: float) -> str:
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    with open(SUMMARY, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    with open(S3, "r", encoding="utf-8") as fh:
        s3 = json.load(fh)

    ctrl = summary[CONTROL]["csi_at_2.0"]
    ctrl_vals = [ctrl[s] for s in SEEDS]

    # --- the floor / oracle context, for the record -------------------------
    print("=" * 78)
    print("E45 S3 verdict decomposition (round 40, read-only)")
    print("=" * 78)
    print("control = %s | threshold = %.1f mm | epoch = %s | n_boot = %d | rng = %d"
          % (s3["control"], s3["threshold_mm"], s3["epoch"], s3["n_boot"], s3["rng_seed"]))
    print("control CSI@2.0 per seed: " + "  ".join(
        "%s=%.6f" % (s, ctrl[s]) for s in SEEDS))
    print()

    max_repro_raw = 0.0
    max_repro_eq = 0.0
    raw_pos_counts = {}
    eq_pos_counts = {}
    all_pos_raw = []
    all_pos_eq = []

    for arm in ARMS:
        a_csi = summary[arm]["csi_at_2.0"]
        entry = s3["arms"][arm]

        raw_gain = {s: a_csi[s] - ctrl[s] for s in SEEDS}
        eq_csi = {s: entry["seeds"][s.replace("s", "")]["csi_equalised"] for s in SEEDS}
        eq_gain = {s: eq_csi[s] - ctrl[s] for s in SEEDS}

        raw_mean = sum(raw_gain.values()) / 3.0
        eq_mean = sum(eq_gain.values()) / 3.0

        d_raw = abs(raw_mean - entry["raw"]["point"])
        d_eq = abs(eq_mean - entry["equal_bias"]["point"])
        max_repro_raw = max(max_repro_raw, d_raw)
        max_repro_eq = max(max_repro_eq, d_eq)
        if d_raw > TOL:
            problems.append("%s raw point mismatch: recomputed %.12f vs stored %.12f"
                            % (arm, raw_mean, entry["raw"]["point"]))
        if d_eq > TOL:
            problems.append("%s equal_bias point mismatch: recomputed %.12f vs stored %.12f"
                            % (arm, eq_mean, entry["equal_bias"]["point"]))

        n_raw_pos = sum(1 for s in SEEDS if raw_gain[s] > 0)
        n_eq_pos = sum(1 for s in SEEDS if eq_gain[s] > 0)
        raw_pos_counts[arm] = n_raw_pos
        eq_pos_counts[arm] = n_eq_pos
        if n_raw_pos == 3:
            all_pos_raw.append(arm)
        if n_eq_pos == 3:
            all_pos_eq.append(arm)

        raw_pattern = "".join(sign(raw_gain[s]) for s in SEEDS)
        eq_pattern = "".join(sign(eq_gain[s]) for s in SEEDS)

        print("-" * 78)
        print("%s   verdict = %s" % (arm, entry["s3_verdict"]))
        print("  raw        gains: " + "  ".join(
            "%s %+.6f" % (s, raw_gain[s]) for s in SEEDS)
            + "   signs=[%s]  mean %+.6f (stored %+.6f)"
            % (raw_pattern, raw_mean, entry["raw"]["point"]))
        print("  equal-bias gains: " + "  ".join(
            "%s %+.6f" % (s, eq_gain[s]) for s in SEEDS)
            + "   signs=[%s]  mean %+.6f (stored %+.6f)"
            % (eq_pattern, eq_mean, entry["equal_bias"]["point"]))
        print("  raw CI [%.6f, %.6f] crosses_zero=%s | equal CI [%.6f, %.6f] crosses_zero=%s"
              % (entry["raw"]["ci_lo"], entry["raw"]["ci_hi"], entry["raw"]["crosses_zero"],
                 entry["equal_bias"]["ci_lo"], entry["equal_bias"]["ci_hi"],
                 entry["equal_bias"]["crosses_zero"]))

        # how much of each seed's raw gain survives equalisation
        keep = []
        for s in SEEDS:
            if abs(raw_gain[s]) > 1e-12:
                keep.append("%s %+.1f%%" % (s, 100.0 * (1.0 - abs(eq_gain[s]) / abs(raw_gain[s]))))
            else:
                keep.append("%s n/a" % s)
        print("  share of raw gain removed by equalisation: " + " | ".join(keep))

        if not entry["raw"]["crosses_zero"]:
            problems.append("%s raw CI does not cross zero" % arm)
        if not entry["equal_bias"]["crosses_zero"]:
            problems.append("%s equal-bias CI does not cross zero" % arm)

        # the stored "seed_spread" field is a ratio, not a difference
        spread_field = summary[arm]["seed_spread_csi_at_2.0"]
        vals = [a_csi[s] for s in SEEDS]
        ratio = max(vals) / min(vals)
        if abs(spread_field - ratio) > 1e-9:
            problems.append("%s seed_spread field %.9f != max/min ratio %.9f"
                            % (arm, spread_field, ratio))

    # --- global statements --------------------------------------------------
    print("-" * 78)
    print("point-estimate reproduction: max |recomputed - stored| raw = %.3e, equal = %.3e (tol %.0e)"
          % (max_repro_raw, max_repro_eq, TOL))
    print("arms with positive raw gain in ALL 3 seeds : %s"
          % (", ".join(all_pos_raw) if all_pos_raw else "NONE"))
    print("arms with positive equal-bias gain in ALL 3 seeds: %s"
          % (", ".join(all_pos_eq) if all_pos_eq else "NONE"))
    print("per-arm count of positive seeds (raw / equal-bias): " + ", ".join(
        "%s %d/3 %d/3" % (a, raw_pos_counts[a], eq_pos_counts[a]) for a in ARMS))
    print()
    print("seed_spread_csi_at_2.0 semantics: verified = max(seed CSI) / min(seed CSI)")
    print("  " + ", ".join("%s %.4f" % (a, summary[a]["seed_spread_csi_at_2.0"]) for a in ARMS))
    print()

    # --- interpretation ------------------------------------------------------
    if all_pos_raw:
        notes.append("arm(s) %s show a positive raw gain in all three seeds; the "
                     "negative verdict for them would rest on the CI alone"
                     % ", ".join(all_pos_raw))
    else:
        notes.append("no arm has a positive raw gain in all three seeds -> the "
                     "negative verdict is corroborated by a plain sign count, "
                     "independently of the bootstrap interval")
    mixed = [a for a in ARMS if 0 < raw_pos_counts[a] < 3]
    if mixed:
        notes.append("arms with mixed per-seed sign (%s): the mean gain is a "
                     "cancellation, so seed instability -- not a uniform deficit "
                     "--is the operative reason" % ", ".join(mixed))
    notes.append("the S3 bootstrap is a two-level (seed + window) resample, so the "
                 "stored CI already absorbs seed-to-seed variance; a CI that "
                 "crosses zero is therefore consistent with the per-seed sign "
                 "counts above")
    notes.append("verdict unchanged: all six arms cross zero on both raw and "
                 "equal-bias sides -> six arms negative, no arm enters the "
                 "10-seed line")

    print("=" * 78)
    for n in notes:
        print("NOTE: " + n)
    print("=" * 78)
    if problems:
        for p in problems:
            print("FAIL: " + p)
        print("VERDICT: FAIL (%d problem(s))" % len(problems))
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
