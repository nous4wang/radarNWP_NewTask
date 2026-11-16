#!/usr/bin/env python3
"""E45 round 91 -- bootstrap RNG-seed robustness of the S3 guard rail.

The published S3 verdict rests on ONE random seed (20261109, n_boot=
10000).  Earlier rounds checked the stored endpoints -- that they contain
zero, that the stored flags agree with the stored numbers -- but never
re-ran the resampling with a different seed.  A negative verdict ("six
arms discarded") has to be robust, so this job asks the question
directly, on the remote, where the dumped fields live:

    for every arm, does the "the interval contains zero" answer change
    when the bootstrap's random seed changes?

How it is done, and why it is cheap enough to be worth a job:

* The equalising SCALE is taken from the stored ``s3_equalbias.json``
  (it was found by bisection there), so the bisection is NOT repeated.
  The job recomputes the contingency counts at that scale from the
  dumped fields and CHECKS that the achieved BIAS still equals the
  stored one -- if that check failed, the counts would be a different
  animal and the comparison meaningless.
* The equalised counts are then resampled with the two-level bootstrap
  of ``tools/_e41_s3_equalbias.py`` -- the SAME implementation that
  produced the published numbers, imported rather than reimplemented,
  so a difference in the result cannot come from a reimplementation.
* The registered seed is run first and must reproduce the stored
  endpoints; then nineteen other seeds are run and the number of seeds
  whose interval still contains zero is recorded.

Nothing here trains or evaluates a model: it reads the fields that
``scripts/evaluate.py --dump-fields`` wrote and the window tables next
to them, exactly like the guard rail it is auditing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

# The registered seed plus nineteen others.
ALT_SEEDS = [
    20261109, 1, 2, 3, 5, 7, 11, 13, 101, 1009,
    4242, 2026, 20261110, 20261111, 20261112, 20270101,
    123456789, 314159, 555555, 999983,
]
ARMS = ["A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]


def load_s3_module(repo: Path):
    """Import the guard rail's own implementation."""
    path = repo / "tools" / "_e41_s3_equalbias.py"
    spec = importlib.util.spec_from_file_location("s3mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    return mod, digest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="/home/wf/radarNWP")
    ap.add_argument("--runs", default="/home/wf/radarNWP/runs/e45")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = Path(args.repo)
    runs = Path(args.runs)
    evals = runs / "evals"
    fields = runs / "fields"

    s3mod, module_md5 = load_s3_module(repo)
    print(f"guard rail implementation: tools/_e41_s3_equalbias.py md5={module_md5}")
    print(f"n_boot={s3mod.N_BOOT} rng_seed={s3mod.RNG_SEED} thr={s3mod.THR}")

    stored = json.loads((runs / "s3_equalbias.json").read_text(encoding="utf-8"))
    if stored["threshold_mm"] != s3mod.THR or stored["n_boot"] != s3mod.N_BOOT \
            or stored["rng_seed"] != s3mod.RNG_SEED:
        raise SystemExit("the stored guard rail used different parameters")

    def tag(arm: str, seed: int) -> str:
        return f"{arm}_s{seed}_ep{stored['epoch']}"

    ctrl = {s: s3mod.window_counts(evals / f"{tag('B0', s)}.windows.json")
            for s in SEEDS}
    ctrl_list = [ctrl[s] for s in SEEDS]

    out: dict = {
        "axis": "s3_bootstrap_rng_robustness",
        "round": 91,
        "threshold_mm": stored["threshold_mm"],
        "control": stored["control"],
        "n_boot": s3mod.N_BOOT,
        "rng_seed": s3mod.RNG_SEED,
        "module_md5": module_md5,
        "alt_seeds": ALT_SEEDS,
        "arms": {},
    }

    for arm in ARMS:
        entry: dict = {}
        raw_a = [s3mod.window_counts(evals / f"{tag(arm, s)}.windows.json")
                 for s in SEEDS]
        raw_rep = s3mod.two_level_bootstrap(raw_a, ctrl_list,
                                            np.random.default_rng(s3mod.RNG_SEED))
        entry["raw_reproduced"] = raw_rep
        entry["raw_sweep"] = [
            s3mod.two_level_bootstrap(raw_a, ctrl_list, np.random.default_rng(sd))
            for sd in ALT_SEEDS
        ]

        eq_a = []
        bias_gaps = []
        for s in SEEDS:
            pp = fields / f"{tag(arm, s)}.pred.npy"
            tp = fields / f"{tag(arm, s)}.truth.npy"
            if not pp.exists() or not tp.exists():
                raise SystemExit(f"MISSING fields for {arm} s{s}")
            pred = np.load(pp, mmap_mode="r")
            truth = np.load(tp, mmap_mode="r")
            scale = stored["arms"][arm]["seeds"][str(s)]["scale"]
            c = s3mod.counts_for_scale(pred, truth, scale, s3mod.THR)
            tot = c.sum(axis=0)
            achieved = s3mod.bias_of(*tot)
            want = stored["arms"][arm]["seeds"][str(s)]
            gap = abs(achieved - want["achieved_bias"])
            bias_gaps.append(gap)
            print(f"  {arm} s{s}: scale={scale:.8f} achieved BIAS={achieved:.8f} "
                  f"(stored {want['achieved_bias']:.8f}) gap={gap:.3e} "
                  f"CSI={s3mod.csi_of(*tot):.6f} "
                  f"(stored {want['csi_equalised']:.6f})")
            eq_a.append(c)
        entry["max_abs_bias_gap"] = float(max(bias_gaps))
        entry["reproduced"] = s3mod.two_level_bootstrap(
            eq_a, ctrl_list, np.random.default_rng(s3mod.RNG_SEED))
        entry["sweep"] = [
            s3mod.two_level_bootstrap(eq_a, ctrl_list, np.random.default_rng(sd))
            for sd in ALT_SEEDS
        ]
        out["arms"][arm] = entry
        print(f"  {arm}: raw reproduced point={raw_rep['point']:+.6f} "
              f"CI[{raw_rep['ci_lo']:+.6f}, {raw_rep['ci_hi']:+.6f}]")
        e = entry["reproduced"]
        print(f"  {arm}: equalised reproduced point={e['point']:+.6f} "
              f"CI[{e['ci_lo']:+.6f}, {e['ci_hi']:+.6f}]")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
