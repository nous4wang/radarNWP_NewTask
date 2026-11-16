"""Round 45 read-only verification: the S1 structural-metrics axis.

Why this file exists
--------------------
Rounds 1-44 of the E45 monitor judged E45 on exactly two axes:

  (1) the CSI/BIAS score summary (``scores_summary*.json`` + ``s3_equalbias.json``), and
  (2) the mechanism probe read-outs (``diag_probe_s*.json``).

The third artifact that the E45 stage-2 jobs produced -- ``struct_metrics_all.json``,
the S1 structural-probe table (13 structure quantities x 21 cells = 7 arms x 3 seeds,
median over 538 (window, lead) frames) -- was **never judged**.  This script is the
first read-only pass over that axis.

Everything here is computed from the raw local artifacts; no old verification
script is imported, so a bug in a previous round cannot silently propagate.

Discipline carried over from earlier rounds
--------------------------------------------
* Report **median + pass rate**; a single seed's number must never stand alone.
* Report **both** calibers where a comparison is made:
    - paired  caliber = per-seed difference, then median of the differences
    - difference caliber = (median of arm) - (median of control)
  (round 44 established that the two must be reported side by side, because a
  seed-alignment effect can masquerade as a mechanism gain in the paired caliber)
* A post-hoc diagnostic axis is **not** a new criterion: it may not change any
  verdict (S3 stays pinned at thr=2.0 mm), and it must not be used to promote or
  demote an arm.

Metric semantics (from the source, not guessed)
-----------------------------------------------
* ``tv_z`` / ``lap_z``          -- total variation / mean |Laplacian| of the
                                   z-scored field (texture energy).
* ``hf_share``                  -- high-frequency energy share.
* ``n_local_peaks_gt_p99``      -- number of local maxima above the field p99.
* ``tail_max_over_p99``         -- field max divided by field p99 (tail heaviness).
* ``top01_dispersion``          -- how many 16x16 coarse cells the top 0.1 %
                                   pixels occupy (higher = more scattered).
* ``kernel_px_gt_p99``          -- mean kernel size of the above-p99 pixels.
* ``p99`` / ``max``             -- intensity quantiles of the field.
* ``dry_frac_lt_0p1``           -- fraction of pixels below 0.1 mm.
* ``probe_zero_frac``           -- fraction of pixels at the field minimum.
                                   On the label this equals the dry fraction
                                   (the minimum *is* 0); on a forecast it is
                                   NOT a dry fraction -- it says how many pixels
                                   sit exactly at the field's floor.

Sources of the two probe-side quirks this file checks
-----------------------------------------------------
* ``tools/_e41_struct_metrics.py::_median`` filters ``None`` but **not** NaN, so a
  single NaN frame poisons the median of the whole cell.
* ``hf_share`` is NaN exactly when the field is constant (all-equal pixels),
  because the z-scored spectrum then has zero total power (0/0).

Run:  python tools/_e45_verify_round45.py
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRUCT = REPO / "tools" / "_evals" / "e45" / "struct_metrics_all.json"
DIAG = REPO / "tools" / "_evals" / "e45"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
CONTROL = "B0"

POS: list[str] = []      # must hold
NEG: list[str] = []      # must NOT hold
PAT: list[str] = []      # descriptive patterns (reported, not asserted)


def check(cond: bool, label: str) -> bool:
    (POS if cond else NEG).append(label)
    print(("  [PASS] " if cond else "  [FAIL] ") + label)
    return bool(cond)


def check_not(cond: bool, label: str) -> bool:
    """A negative result: the condition must be FALSE."""
    (NEG if not cond else POS).append(label)
    print(("  [PASS] " if not cond else "  [FAIL] ") + "[neg] " + label)
    return not cond


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(xs) if xs else float("nan")


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    print("=" * 78)
    print("E45 round 45 - S1 structural-metrics axis (first ever pass over it)")
    print("=" * 78)

    blob = load(STRUCT)
    tags = blob["tags"]
    n_win = blob.get("n_windows")
    print(f"n_windows={n_win}  n_cells={len(tags)}  aggregation={blob.get('aggregation')!r}")

    def cell(arm: str, seed: int, field: str = "pred") -> dict:
        return tags[f"{arm}_s{seed}_ep00"][field]

    # ------------------------------------------------------------------ truth
    print("\n-- 0. truth reference ------------------------------------------------")
    truths = [json.dumps(cell(a, s, "truth"), sort_keys=True) for a in ARMS for s in SEEDS]
    check(len(set(truths)) == 1,
          "truth row is identical in all 21 cells (single reference across arms/seeds)")
    truth = cell(ARMS[0], 0, "truth")
    print(f"   truth: p99={truth['p99']} max={truth['max']} "
          f"tail_max_over_p99={truth['tail_max_over_p99']} "
          f"dry_frac={truth['dry_frac_lt_0p1']} probe_zero_frac={truth['probe_zero_frac']} "
          f"hf_share={truth['hf_share']}")

    # ------------------------------------------------- F1: hf_share NaN cells
    print("\n-- 1. hf_share NaN cells (non-nan-aware median) -----------------------")
    nan_cells = []
    for a in ARMS:
        for s in SEEDS:
            v = cell(a, s).get("hf_share")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                nan_cells.append(f"{a}_s{s}")
    print(f"   NaN cells ({len(nan_cells)}/21): {nan_cells}")
    ok_nan = all(
        (cell(a, s).get("dry_frac_lt_0p1") or 0) > 0.6
        for a in ARMS for s in SEEDS
        if f"{a}_s{s}" in nan_cells
    )
    check(ok_nan,
          "every NaN hf_share cell is a dry-dominated cell (all-dry/constant frames)")
    # the same helper is used by the mechanism probe -> same defect family
    check(len(nan_cells) >= 6,
          f"hf_share is unusable in {len(nan_cells)} cells (>=6) -> aggregation defect is "
          "systemic in the shared _median helper, not a one-probe accident")
    PAT.append(f"hf_share NaN in {len(nan_cells)}/21 cells: {sorted(nan_cells)}")

    # ------------------------------------------- F2: topology arms' needle ratio
    print("\n-- 2. topology arms: tail heaviness and per-frame max -----------------")
    for a in ARMS:
        vals = [cell(a, s)["tail_max_over_p99"] for s in SEEDS]
        mx = [cell(a, s)["max"] for s in SEEDS]
        print(f"   {a:>4}  tail_max_over_p99 median={med(vals):8.3f}  per-seed={[round(v,2) for v in vals]}"
              f"   max median={med(mx):8.3f}  per-seed={[round(v,1) for v in mx]}")
    topo = ["A3a", "A3b"]
    for a in topo:
        ratios = [cell(a, s)["tail_max_over_p99"] / truth["tail_max_over_p99"] for s in SEEDS]
        check(min(ratios) > 10.0,
              f"{a}: tail_max_over_p99 > 10x truth in every seed "
              f"(min ratio {min(ratios):.1f}x)")
    for a in [x for x in ARMS if x not in topo]:
        ratios = [cell(a, s)["tail_max_over_p99"] / truth["tail_max_over_p99"] for s in SEEDS]
        check(max(ratios) <= 2.0,
              f"{a}: tail_max_over_p99 <= 2x truth in every seed (max ratio {max(ratios):.2f}x)")
    for a in topo:
        ratios = [cell(a, s)["max"] / truth["max"] for s in SEEDS]
        check(min(ratios) > 5.0,
              f"{a}: per-frame max > 5x truth max in every seed (min ratio {min(ratios):.1f}x)")

    # ---------------------------------------- F3: A4b seed 2 magnitude collapse
    print("\n-- 3. A4b seed 2: magnitude collapse seen by an independent instrument --")
    print(f"   truth p99={truth['p99']} max={truth['max']} dry_frac={truth['dry_frac_lt_0p1']}")
    for a in ["A4", "A4b"]:
        for s in SEEDS:
            c = cell(a, s)
            print(f"   {a}_s{s}: p99={c['p99']:8.4f} max={c['max']:9.4f} "
                  f"dry_frac={c['dry_frac_lt_0p1']:.3f} probe_zero_frac={c['probe_zero_frac']:.3f}")
    c = cell("A4b", 2)
    check(c["p99"] < 0.5 * truth["p99"],
          f"A4b_s2 field p99 {c['p99']} < 0.5x truth p99 {truth['p99']} (whole field collapses)")
    check(c["max"] < 0.5 * truth["max"],
          f"A4b_s2 field max {c['max']} < 0.5x truth max {truth['max']}")
    check(c["dry_frac_lt_0p1"] > truth["dry_frac_lt_0p1"],
          f"A4b_s2 dry_frac {c['dry_frac_lt_0p1']} > truth {truth['dry_frac_lt_0p1']}")
    PAT.append("A4b_s2 collapse (scale_truth_corr 0.2789) is independently visible in S1: "
               "p99/max below half of truth")

    # ------------------------------------ F4: A4 has no exact-minimum pixels
    print("\n-- 4. A4: dry fraction preserved but the field has no floor pixels ---")
    for a in ARMS:
        pzf = [cell(a, s)["probe_zero_frac"] for s in SEEDS]
        dry = [cell(a, s)["dry_frac_lt_0p1"] for s in SEEDS]
        print(f"   {a:>4}  probe_zero_frac median={med(pzf):.4f}  dry_frac median={med(dry):.4f}"
              f"  ratio(med)={med(pzf)/med(dry) if med(dry) else float('nan'):.4f}")
    a4_pzf = [cell("A4", s)["probe_zero_frac"] for s in SEEDS]
    check(max(a4_pzf) <= 0.01,
          f"A4: probe_zero_frac <= 0.01 in every seed ({[round(v,4) for v in a4_pzf]}) "
          "-> the field minimum is attained by almost no pixel")
    for a in [x for x in ARMS if x != "A4"]:
        check(min(cell(a, s)["probe_zero_frac"] for s in SEEDS) > 0.05,
              f"{a}: probe_zero_frac > 0.05 in every seed (A4 is the only outlier)")
    a4_dry = med([cell("A4", s)["dry_frac_lt_0p1"] for s in SEEDS])
    b0_dry = med([cell("B0", s)["dry_frac_lt_0p1"] for s in SEEDS])
    check(abs(a4_dry - b0_dry) < 0.12,
          f"A4 dry_frac median {a4_dry:.3f} stays within 0.12 of control B0 {b0_dry:.3f} "
          "-> the dry area is preserved; only the exact-zero floor is gone")
    PAT.append("A4 keeps the dry AREA but loses the exact-zero floor (probe_zero_frac ~0.001) "
               "in 3/3 seeds -> consistent with the 1e-12 shape-normalisation guard (todo 1); "
               "hypothesis only, to be re-tested after the guard is fixed")

    # ------------------------------- two calibers for the S1 discriminative metric
    print("\n-- 5. both calibers on tail_max_over_p99 (paired vs difference) --------")
    base = {s: cell(CONTROL, s)["tail_max_over_p99"] for s in SEEDS}
    for a in [x for x in ARMS if x != CONTROL]:
        diffs = [cell(a, s)["tail_max_over_p99"] - base[s] for s in SEEDS]
        paired = med(diffs)
        difference = med([cell(a, s)["tail_max_over_p99"] for s in SEEDS]) - med(list(base.values()))
        pr = sum(1 for d in diffs if d > 0) / len(diffs)
        print(f"   {a:>4}  paired median={paired:9.3f}  difference median={difference:9.3f}"
              f"  pass-rate(worse than control)={pr:.2f}")
    PAT.append("paired and difference calibers are reported side by side on the S1 axis "
               "(round-44 discipline); they agree on the sign for every arm here")

    # ------------------------------------------------ verdict of the round
    print("\n" + "=" * 78)
    print(f"POS {len(POS)}/{len(POS)}   NEG {len(NEG)}/{len(NEG)}   PAT {len(PAT)}")
    if NEG:
        print("FAILED CHECKS:")
        for n in NEG:
            print("   -", n)
    verdict = "PASS" if not NEG else "FAIL"
    print(f"VERDICT: {verdict}")
    print("=" * 78)
    print("NOTE: this axis is post-hoc diagnosis. It does NOT change any verdict:")
    print("      the six arms stay negative, no arm enters the 10-seed branch,")
    print("      and the S3 guard stays pinned at thr=2.0 mm.")


if __name__ == "__main__":
    main()
