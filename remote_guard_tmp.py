#!/usr/bin/env python3
"""E41 / MASTER 4.43 -- S3 equal-bias pure-amplitude control (run on the remote).

E41 section 5, frozen 2026-11-09 09:29:

    S3 -- the guard rail.  Multiply the A-arm forecast field by a GLOBAL
    SCALAR so that its ``BIAS@2.0`` equals the B0 arm's ``BIAS@2.0``.  If
    the A arm's ``CSI@2.0`` gain over B0 disappears, or its interval
    crosses zero, after that equalisation, the verdict is "still an area /
    amplitude effect" and the arm is discarded under section 11.8.  This
    is the landing of the E18 / E25 / E33 lessons: **the gain has to be
    shown not to be "reporting more rain"**.

Why this runs on the remote: the window tables are ~320 kB each and the
SSH channel truncates around 256 kB, so a table that is fetched comes back
with its tail missing.  The resampling has to happen where the data is.

Two levels are resampled, because there are two independent units: the
TRAINING SEED (a different seed is a different model) and the EVALUATION
WINDOW (the pixels inside one window are not independent of each other).
The seed level comes first; within the drawn seed the windows are
resampled with replacement.

The equalisation itself is exact and not a search: ``BIAS@2.0`` is
``(hits + false_alarms) / (hits + misses)``.  For a threshold ``t`` and a
scaling ``s`` applied to the forecast, the contingency counts are
recomputed from the dumped field, and the scalar is found by bisection on
``s`` so that the resulting ``BIAS@2.0`` matches the B0 value.  This is
the same "one global scalar" the criterion asks for.

Nothing here trains or evaluates a model.  It reads the fields that
``scripts/evaluate.py --dump-fields`` wrote and the window tables it wrote
next to them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_BOOT = 10000
RNG_SEED = 20261109
THR = 2.0
THR_KEY = "2.0"


def csi_of(h: float, m: float, f: float) -> float:
    d = h + m + f
    return h / d if d > 0 else float("nan")


def bias_of(h: float, m: float, f: float) -> float:
    d = h + m
    return (h + f) / d if d > 0 else float("nan")


def window_counts(path: Path, thr: str = THR_KEY) -> np.ndarray:
    """The per-window contingency counts of one evaluation."""
    rows = json.loads(path.read_text(encoding="utf-8"))["windows"]
    out = np.zeros((len(rows), 3), dtype=np.float64)
    for i, w in enumerate(rows):
        c = w[thr]
        out[i] = (c["hits"], c["misses"], c["false_alarms"])
    return out


def _sorted_masses(pred: np.ndarray, truth: np.ndarray, thr: float
                   ) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Split each window's forecast values by whether the truth is wet.

    For a scaling ``s`` the forecast crosses the threshold when
    ``pred >= thr / s``, so the contingency counts are a function of the
    ORDER STATISTICS of the forecast values, not of the values
    themselves.  Sorting once per window turns the bisection into a
    handful of binary searches instead of a pass over 2 GB of fields per
    step.

    Args:
        pred: ``(n_win, out_len, H, W)`` forecast field.
        truth: the matching truth field.
        thr: the threshold in mm.

    Returns:
        ``(wet_sorted, dry_sorted, n_wet)`` -- one array per window: the
        forecast values at the wet-truth pixels and at the dry-truth
        pixels, both ascending, plus the wet-pixel count.
    """
    wet_sorted: list[np.ndarray] = []
    dry_sorted: list[np.ndarray] = []
    n_wet: list[int] = []
    for w in range(pred.shape[0]):
        p = np.asarray(pred[w], dtype=np.float32).ravel()
        t = np.asarray(truth[w], dtype=np.float32).ravel() >= thr
        wet = np.sort(p[t])
        dry = np.sort(p[~t])
        wet_sorted.append(wet)
        dry_sorted.append(dry)
        n_wet.append(int(wet.size))
    return wet_sorted, dry_sorted, n_wet


def _counts_at(wet_sorted: list[np.ndarray], dry_sorted: list[np.ndarray],
               n_wet: list[int], scale: float, thr: float) -> np.ndarray:
    """The per-window contingency counts at one scaling."""
    thr_s = thr / scale
    out = np.zeros((len(wet_sorted), 3), dtype=np.float64)
    for w in range(len(wet_sorted)):
        wet = wet_sorted[w]
        dry = dry_sorted[w]
        h = wet.size - int(np.searchsorted(wet, thr_s, side="left"))
        fa = dry.size - int(np.searchsorted(dry, thr_s, side="left"))
        out[w] = (h, n_wet[w] - h, fa)
    return out


def counts_for_scale(pred: np.ndarray, truth: np.ndarray, scale: float,
                     thr: float) -> np.ndarray:
    """Recompute the contingency counts after scaling the forecast.

    Args:
        pred: ``(n_win, out_len, H, W)`` float32 forecast field.
        truth: the matching truth field.
        scale: the global scalar.
        thr: the threshold in mm.

    Returns:
        ``(n_win, 3)`` array of ``(hits, misses, false_alarms)``.
    """
    wet_sorted, dry_sorted, n_wet = _sorted_masses(pred, truth, thr)
    return _counts_at(wet_sorted, dry_sorted, n_wet, scale, thr)


def equalising_scale(pred: np.ndarray, truth: np.ndarray, target_bias: float,
                     thr: float = THR) -> tuple[float, float]:
    """The global scalar that makes ``BIAS@thr`` equal ``target_bias``.

    ``BIAS`` is monotone in the scale (a larger forecast crosses the
    threshold more often, so the numerator grows and the denominator does
    not), so a bisection finds the scalar.  The search is done on the
    AGGREGATE counts, which is what ``BIAS`` in the report is.

    Returns:
        ``(scale, achieved_bias)``.
    """
    wet_sorted, dry_sorted, n_wet = _sorted_masses(pred, truth, thr)
    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = (lo * hi) ** 0.5
        c = _counts_at(wet_sorted, dry_sorted, n_wet, mid, thr).sum(axis=0)
        b = bias_of(c[0], c[1], c[2])
        if not np.isfinite(b) or b > target_bias:
            hi = mid
        else:
            lo = mid
    scale = (lo * hi) ** 0.5
    c = _counts_at(wet_sorted, dry_sorted, n_wet, scale, thr).sum(axis=0)
    return scale, bias_of(c[0], c[1], c[2])

def two_level_bootstrap(a_counts: list[np.ndarray], b_counts: list[np.ndarray],
                        rng: np.random.Generator) -> dict:
    """The difference ``a - b`` with a two-level bootstrap.

    Args:
        a_counts: per-seed window counts of the first arm.
        b_counts: per-seed window counts of the second arm, seed-aligned.
        rng: the generator.

    Returns:
        A dict with the observed difference and its percentile interval.
    """
    n_seeds = len(a_counts)
    # The number of windows is len(c), NOT len(c[0]): each c is a
    # (n_win, 3) array of per-window (hits, misses, false_alarms), so
    # c[0] is ONE window's three counts and len(c[0]) is always 3.
    #
    # Reading it the other way made the resampling draw three windows
    # out of 538, so the interval described the sampling distribution of
    # a three-window draw instead of the whole test set.  The symptom was
    # visible in the output: "n_windows": 3, and a point estimate that
    # fell outside its own interval.  E41's and E42's S3 verdicts were
    # taken from intervals of that kind, so both are re-derived after
    # this fix.
    n_win = min(len(c) for c in a_counts)
    diffs = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        si = rng.integers(0, n_seeds, n_seeds)
        obs = []
        for k in si:
            idx = rng.integers(0, n_win, n_win)
            ca = a_counts[k][idx].sum(axis=0)
            cb = b_counts[k][idx].sum(axis=0)
            obs.append(csi_of(*ca) - csi_of(*cb))
        diffs[i] = np.mean(obs)
    point = float(np.mean([
        csi_of(*a_counts[k].sum(axis=0)) - csi_of(*b_counts[k].sum(axis=0))
        for k in range(n_seeds)
    ]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "n_boot": N_BOOT,
        "n_seeds": n_seeds,
        "n_windows": int(n_win),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/home/wf/radarNWP/runs/e41")
    ap.add_argument("--control", default="B0",
                    help="the arm the others are compared against")
    ap.add_argument("--arms", nargs="+", default=["A1", "A2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epoch", default="00")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = Path(args.runs)
    evals = runs / "evals"
    fields = runs / "fields"

    def tag(arm: str, seed: int) -> str:
        return f"{arm}_s{seed}_ep{args.epoch}"

    # The control arm's own numbers.
    ctrl_counts: dict[int, np.ndarray] = {}
    ctrl_bias: dict[int, float] = {}
    for s in args.seeds:
        p = evals / f"{tag(args.control, s)}.windows.json"
        if not p.exists():
            print(f"MISSING control window table: {p}")
            return 1
        c = window_counts(p)
        ctrl_counts[s] = c
        tot = c.sum(axis=0)
        ctrl_bias[s] = bias_of(*tot)
        print(f"control {args.control} s{s}: CSI@{THR_KEY}="
              f"{csi_of(*tot):.4f} BIAS@{THR_KEY}={ctrl_bias[s]:.4f}")

    out: dict = {
        "threshold_mm": THR,
        "epoch": args.epoch,
        "control": args.control,
        "n_boot": N_BOOT,
        "rng_seed": RNG_SEED,
        "arms": {},
    }

    for arm in args.arms:
        entry: dict = {"seeds": {}}
        # 1. the raw comparison
        raw_a, raw_b = [], []
        for s in args.seeds:
            p = evals / f"{tag(arm, s)}.windows.json"
            if not p.exists():
                print(f"MISSING {arm} s{s} window table")
                continue
            raw_a.append(window_counts(p))
            raw_b.append(ctrl_counts[s])
        if not raw_a:
            continue
        rng = np.random.default_rng(RNG_SEED)
        entry["raw"] = two_level_bootstrap(raw_a, raw_b, rng)
        print(f"{arm} raw vs {args.control}: "
              f"{entry['raw']['point']:+.4f} "
              f"CI [{entry['raw']['ci_lo']:+.4f}, {entry['raw']['ci_hi']:+.4f}]")

        # 2. the equal-bias control
        eq_a, eq_b = [], []
        for s in args.seeds:
            pp = fields / f"{tag(arm, s)}.pred.npy"
            tp = fields / f"{tag(arm, s)}.truth.npy"
            if not pp.exists() or not tp.exists():
                print(f"MISSING fields for {arm} s{s}")
                continue
            pred = np.load(pp, mmap_mode="r")
            truth = np.load(tp, mmap_mode="r")
            scale, achieved = equalising_scale(
                pred, truth, ctrl_bias[s], THR)
            c = counts_for_scale(pred, truth, scale, THR)
            tot = c.sum(axis=0)
            print(f"  {arm} s{s}: scale={scale:.6f} "
                  f"BIAS {achieved:.4f} (control {ctrl_bias[s]:.4f}) "
                  f"CSI {csi_of(*tot):.4f}")
            entry["seeds"][str(s)] = {
                "scale": scale,
                "achieved_bias": achieved,
                "control_bias": ctrl_bias[s],
                "csi_equalised": csi_of(*tot),
            }
            eq_a.append(c)
            eq_b.append(ctrl_counts[s])
        if eq_a:
            rng = np.random.default_rng(RNG_SEED)
            entry["equal_bias"] = two_level_bootstrap(eq_a, eq_b, rng)
            e = entry["equal_bias"]
            print(f"{arm} equal-bias vs {args.control}: {e['point']:+.4f} "
                  f"CI [{e['ci_lo']:+.4f}, {e['ci_hi']:+.4f}] "
                  f"crosses_zero={e['crosses_zero']}")
            # The verdict of S3, read as written.
            gained_raw = entry["raw"]["point"] > 0
            survived = (not e["crosses_zero"]) and e["point"] > 0
            entry["s3_verdict"] = (
                "GAIN_SURVIVES_EQUALISATION" if survived
                else ("GAIN_WAS_AREA_OR_AMPLITUDE" if gained_raw
                      else "NO_GAIN_TO_BEGIN_WITH")
            )
            print(f"{arm} S3: {entry['s3_verdict']}")
        out["arms"][arm] = entry

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nS3 result written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
