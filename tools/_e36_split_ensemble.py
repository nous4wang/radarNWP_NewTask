#!/usr/bin/env python3
"""E36-P: offline 5+5 split-ensemble proxy on the ten E25/E26 float32 fields.

NO model, NO training, NO re-inference. Reads the ten dumped whole-field
forecasts already on the remote host and answers one narrow question:

    If the ten seeds are randomly split into two groups of five and each
    group is arithmetically averaged, how far apart are the two synthetic
    models -- compared with the single-seed range 0.1671?

This is an EXPLORATORY PROXY ONLY. The preregistered A4-B criterion needs
>=20 independent seeds split into two groups of ten; ten seeds cannot pass
or fail it. See docs/experiments/E36-split-ensemble-proxy.md (criteria
frozen 2026-11-07 16:30, before any compute).

Only the 2.0 mm/6min tier is scored, using the exact same contingency
arithmetic as tools/_field_probe.py::counts(). No argsort, no tie-breaking,
no cross-device discrete ops -> pure numpy deterministic (MASTER 11.6 S3).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TIER = 2.0
FLOOR = 0.0924
E26_FULL_MEAN_CSI2 = 0.10678156330315557  # runs/field_probe.json A4.mean.2.0.csi
ANCHOR_MIN = 0.0194
ANCHOR_MAX = 0.1865
SINGLE_SEED_RANGE = 0.1865 - 0.0194  # 0.1671


class Acc:
    """Accumulate 2.0 mm contingency counts across windows."""

    def __init__(self) -> None:
        self.h = 0.0
        self.m = 0.0
        self.f = 0.0

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        p = pred >= TIER
        t = truth >= TIER
        self.h += float(np.count_nonzero(p & t))
        self.m += float(np.count_nonzero(~p & t))
        self.f += float(np.count_nonzero(p & ~t))

    def report(self) -> dict:
        den = self.h + self.m + self.f
        hm = self.h + self.m
        return {
            "csi": self.h / den if den else float("nan"),
            "bias": (self.h + self.f) / hm if hm else float("nan"),
            "hits": self.h,
            "misses": self.m,
            "false_alarms": self.f,
        }


def make_splits(n_seeds: int, n_splits: int) -> list[tuple[list[int], list[int]]]:
    """Fixed first split (front/back), then seeded random permutations."""
    out: list[tuple[list[int], list[int]]] = [
        (list(range(n_seeds // 2)), list(range(n_seeds // 2, n_seeds)))
    ]
    for k in range(n_splits - 1):
        order = np.random.default_rng(20261107 + k).permutation(n_seeds)
        out.append(
            (sorted(order[: n_seeds // 2].tolist()),
             sorted(order[n_seeds // 2 :].tolist()))
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields-dir", default="runs/fields")
    ap.add_argument("--prefix", default="base_concat_s")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="smoke-test only; full run must omit this")
    ap.add_argument("--out", default="runs/e36_split_ensemble.json")
    args = ap.parse_args()

    root = Path(args.fields_dir)
    pred_paths = [root / f"{args.prefix}{i}.pred.npy" for i in range(args.n_seeds)]
    truth_paths = [root / f"{args.prefix}{i}.truth.npy" for i in range(args.n_seeds)]
    for p in pred_paths + truth_paths:
        if not p.is_file():
            raise SystemExit(f"missing field: {p}")

    preds = [np.load(p, mmap_mode="r") for p in pred_paths]
    truths = [np.load(p, mmap_mode="r") for p in truth_paths]
    dtypes = {str(a.dtype) for a in preds + truths}
    if dtypes != {"float32"}:
        raise SystemExit(
            f"dumped fields must be float32, got {sorted(dtypes)}; float16 "
            "rounds 0.1 mm to 0.0999756 and breaks the per-window invariant "
            "(tools/_check_field_dtype.py)"
        )
    shapes = {a.shape for a in preds + truths}
    if len(shapes) != 1:
        raise SystemExit(f"field shape mismatch: {shapes}")
    n = preds[0].shape[0]
    if args.max_windows:
        n = min(n, args.max_windows)
    print(f"fields={len(preds)} windows={n} shape={preds[0].shape[1:]} "
          f"dtype=float32 splits={args.splits} smoke={bool(args.max_windows)}",
          flush=True)

    # All ten truth files must be the same ground truth; check a few windows
    # rather than hashing 19 GB.
    for w in (0, n // 2, n - 1):
        t0 = np.asarray(truths[0][w])
        for i in range(1, args.n_seeds):
            if not np.array_equal(t0, np.asarray(truths[i][w])):
                raise SystemExit(f"truth mismatch between seed 0 and {i} at window {w}")

    splits = make_splits(args.n_seeds, args.splits)
    seed_acc = [Acc() for _ in range(args.n_seeds)]
    full_acc = Acc()
    group_a = [Acc() for _ in splits]
    group_b = [Acc() for _ in splits]

    t_start = time.time()
    for w in range(n):
        stack = np.stack(
            [np.asarray(p[w], dtype=np.float32) for p in preds], axis=0
        )  # (n_seeds, T=10, 301, 301)
        truth_w = np.asarray(truths[0][w])

        for i in range(args.n_seeds):
            seed_acc[i].add(stack[i], truth_w)
        full_acc.add(stack.mean(axis=0), truth_w)

        for sidx, (g1, g2) in enumerate(splits):
            group_a[sidx].add(stack[g1].mean(axis=0), truth_w)
            group_b[sidx].add(stack[g2].mean(axis=0), truth_w)

        if (w + 1) % 50 == 0 or w + 1 == n:
            el = time.time() - t_start
            print(f"  window {w + 1}/{n}  elapsed {el:.1f}s", flush=True)

    per_seed = [a.report() for a in seed_acc]
    full = full_acc.report()
    csi2 = [r["csi"] for r in per_seed]
    n_over = sum(1 for c in csi2 if c > FLOOR)
    smin, smax = min(csi2), max(csi2)
    anchor_pass = (
        abs(full["csi"] - E26_FULL_MEAN_CSI2) <= 1e-6
        and n_over == 4
        and smin <= ANCHOR_MIN + 1e-3
        and smax >= ANCHOR_MAX - 1e-3
    )

    rows = []
    for sidx, (g1, g2) in enumerate(splits):
        ra, rb = group_a[sidx].report(), group_b[sidx].report()
        rows.append({
            "split": sidx,
            "group_a_seeds": g1,
            "group_b_seeds": g2,
            "group_a": ra,
            "group_b": rb,
            "abs_diff": abs(ra["csi"] - rb["csi"]),
            "both_over_floor": bool(ra["csi"] > FLOOR and rb["csi"] > FLOOR),
        })
    diffs = [r["abs_diff"] for r in rows]
    summary = {
        "median_abs_diff": float(np.median(diffs)),
        "max_abs_diff": float(max(diffs)),
        "min_abs_diff": float(min(diffs)),
        "n_both_over_floor": sum(1 for r in rows if r["both_over_floor"]),
        "n_splits": len(rows),
        "single_seed_range": SINGLE_SEED_RANGE,
        "median_diff_over_single_seed_range": float(np.median(diffs)) / SINGLE_SEED_RANGE,
        "max_diff_over_single_seed_range": float(max(diffs)) / SINGLE_SEED_RANGE,
        "usable_space_oracle_minus_floor": 0.1653 - FLOOR,
    }

    out = {
        "meta": {
            "experiment": "E36-P exploratory 5+5 split-ensemble proxy",
            "criteria_frozen_at": "2026-11-07 16:30",
            "fields_dir": str(root),
            "n_seeds": args.n_seeds,
            "n_windows": n,
            "smoke": bool(args.max_windows),
            "tier_mm_per_6min": TIER,
            "floor": FLOOR,
            "device_scale": "pure numpy CPU, no discrete cross-device ops (MASTER 11.6 S3)",
            "warning": "5+5 CANNOT pass or fail preregistered A4-B (>=20 seeds); "
                       "descriptive only, A4-B remains NOT TESTED",
        },
        "anchors": {
            "full_mean": full,
            "e26_full_mean_csi2": E26_FULL_MEAN_CSI2,
            "full_mean_abs_error": abs(full["csi"] - E26_FULL_MEAN_CSI2),
            "per_seed_csi2": csi2,
            "n_over_floor": n_over,
            "min": smin,
            "max": smax,
            "spread": smax / smin if smin > 0 else float("nan"),
            "anchor_pass": bool(anchor_pass),
        },
        "splits": rows,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ANCHOR full_mean csi2={full['csi']:.9f} "
          f"(E26 {E26_FULL_MEAN_CSI2:.9f}, err {out['anchors']['full_mean_abs_error']:.2e}) "
          f"per-seed over floor {n_over}/10 range [{smin:.6f},{smax:.6f}] "
          f"spread {out['anchors']['spread']:.2f}x PASS={anchor_pass}", flush=True)
    print(f"SPLITS median |diff|={summary['median_abs_diff']:.6f} "
          f"max={summary['max_abs_diff']:.6f} "
          f"both>floor {summary['n_both_over_floor']}/{summary['n_splits']} "
          f"median/range={summary['median_diff_over_single_seed_range']:.4f}", flush=True)
    print(f"WROTE {out_path}", flush=True)
    return 0 if anchor_pass else 2


if __name__ == "__main__":
    sys.exit(main())
