#!/usr/bin/env python3
"""Offline analysis of dumped whole-field forecasts (E25 A.2 + E26 A.4).

Reads the float16 memory-mapped fields written by
``scripts/evaluate.py --dump-fields`` and computes, WITHOUT any model or
training:

* A2 anchor: recomputed CSI@2.0 must match the eval.json of the same run.
* A2-A: multiscale fractions skill score at half-widths 2 / 4 / 8 px
  (5x5 / 9x9 / 17x17), next to pixel-wise CSI.
* A2-B (decisive): +/-3 px max-pool phase relaxation vs a PURE AMPLITUDE
  control that scales the original forecast by one constant until its
  BIAS@0.5 equals the max-pooled one.  If max-pool does not beat that
  control, the CSI gain cannot be attributed to phase correction
  (MASTER 4.18.5.4 / 4.20.1.1 / bearing conclusion 48).
* A4: arithmetic-mean and majority-vote ensembles over all supplied seeds,
  with per-window tables for paired bootstrap against the k=1 floor.

All arithmetic is on whole 301x301 fields, one window at a time, so peak RAM
stays small even with ten seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Use the benchmark's own implementation so A2's radius-3 anchor is
# bit-comparable with the existing eval.json `fss` field.
from radar_nwp_bench.metrics.precipitation import fractions_skill_score  # noqa: E402

TIERS = (0.1, 0.5, 2.0, 4.0)
# These are JSON KEYS, not display labels.  f"{2.0:g}" would produce "2",
# which does not match evaluate.py's "thresholds" dict, the reference
# floor table, or _bootstrap_qpe_vs_zr.py's TIERS.  Keep the exact strings.
TIER_KEYS = ("0.1", "0.5", "2.0", "4.0")
FSS_RADII = (2, 4, 8)  # 5x5 / 9x9 / 17x17
POOL_RADIUS = 3         # +/-3 px -> 7x7 window
FLOOR_K1_CSI2 = 0.0924188
VOTE_TIER = 2.0


def counts(pred: np.ndarray, truth: np.ndarray, thr: float) -> dict:
    p = pred >= thr
    t = truth >= thr
    return {
        "hits": float(np.count_nonzero(p & t)),
        "misses": float(np.count_nonzero(~p & t)),
        "false_alarms": float(np.count_nonzero(p & ~t)),
        "correct_negatives": float(np.count_nonzero(~p & ~t)),
    }


class Accumulator:
    """Sum contingency counts over windows; also keep per-window rows."""

    def __init__(self) -> None:
        self.total = {k: {c: 0.0 for c in
                          ("hits", "misses", "false_alarms", "correct_negatives")}
                      for k in TIER_KEYS}
        self.rows: list[dict] = []

    def add(self, pred: np.ndarray, truth: np.ndarray) -> None:
        row = {}
        for thr, key in zip(TIERS, TIER_KEYS):
            c = counts(pred, truth, thr)
            row[key] = c
            for name, v in c.items():
                self.total[key][name] += v
        self.rows.append(row)

    def report(self) -> dict:
        out = {}
        for key in TIER_KEYS:
            c = self.total[key]
            h, m, f = c["hits"], c["misses"], c["false_alarms"]
            out[key] = {
                "csi": h / (h + m + f) if (h + m + f) else float("nan"),
                "bias": (h + f) / (h + m) if (h + m) else float("nan"),
                "pod": h / (h + m) if (h + m) else float("nan"),
                "far": f / (h + f) if (h + f) else float("nan"),
                **c,
            }
        return out


def fss(pred: np.ndarray, truth: np.ndarray, thr: float, radius: int) -> float:
    """Delegate to the benchmark verifier's own FSS (identical arithmetic)."""
    return float(fractions_skill_score(pred, truth, thr, radius))


def maxpool(field: np.ndarray, device: torch.device) -> np.ndarray:
    """Dilate each lead frame by a 7x7 max filter (phase relaxation)."""
    x = torch.from_numpy(field.astype(np.float32)).to(device)
    # (T, H, W) -> (T, 1, H, W) so each lead is filtered independently
    y = F.max_pool2d(x[:, None], kernel_size=2 * POOL_RADIUS + 1,
                     stride=1, padding=POOL_RADIUS)
    return y[:, 0].cpu().numpy()


def device_consistency(device: torch.device) -> dict:
    """S2 check (MASTER §11.6): discrete ops must agree across CPU / CUDA.

    Max pooling and the majority-vote comparison are the only discrete
    operations in this probe. Unlike ``argsort`` (§4.24), neither depends on
    tie ordering, but §11.6 requires the check to be run and recorded rather
    than assumed. Synthetic fields with deliberate ties are used.
    """
    rng = np.random.default_rng(20261106)
    x = rng.integers(0, 5, size=(2, 9, 9)).astype(np.float32) / 2.0
    x[0, 4, 4] = 3.0
    a = maxpool(x, torch.device("cpu"))
    if device.type == "cuda":
        b = maxpool(x, device)
        pool_identical = bool(np.array_equal(a, b))
        pool_max_abs_diff = float(np.max(np.abs(a - b)))
    else:
        pool_identical, pool_max_abs_diff = None, None

    stack = rng.integers(0, 4, size=(5, 2, 9, 9)).astype(np.float32)
    need = (len(stack) + 1) // 2
    v_np = (stack >= VOTE_TIER).sum(axis=0) >= need
    v_cpu = ((torch.from_numpy(stack).to("cpu") >= VOTE_TIER)
             .sum(dim=0) >= need).numpy()
    vote_cpu_identical = bool(np.array_equal(v_np, v_cpu))
    if device.type == "cuda":
        v_cuda = ((torch.from_numpy(stack).to(device) >= VOTE_TIER)
                  .sum(dim=0) >= need).cpu().numpy()
        vote_cuda_identical = bool(np.array_equal(v_np, v_cuda))
    else:
        vote_cuda_identical = None

    admissible = (pool_identical is not False
                  and vote_cpu_identical
                  and vote_cuda_identical is not False)
    return {
        "maxpool_cpu_vs_cuda_identical": pool_identical,
        "maxpool_max_abs_diff": pool_max_abs_diff,
        "vote_numpy_vs_torch_cpu_identical": vote_cpu_identical,
        "vote_numpy_vs_torch_cuda_identical": vote_cuda_identical,
        "S2_admissible": admissible,
        "note": ("CPU-only run: cross-device comparison not available, values "
                 "are diagnostic only" if device.type != "cuda" else
                 "synthetic tied fields; max is order-independent and the vote "
                 "is a threshold comparison, so no tie-breaking hazard"),
    }


def write_windows(acc: Accumulator, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({
        "n_windows": len(acc.rows),
        "thresholds": list(TIER_KEYS),
        "windows": acc.rows,
    }, indent=2), encoding="utf-8")


def load_fields(paths: list[str], truth_path: str, max_windows: int | None):
    preds = [np.load(p, mmap_mode="r") for p in paths]
    truth = np.load(truth_path, mmap_mode="r")
    shapes = {a.shape for a in preds} | {truth.shape}
    if len(shapes) != 1:
        raise SystemExit(f"field shape mismatch: {shapes}")
    n = preds[0].shape[0]
    if max_windows:
        n = min(n, max_windows)
    return preds, truth, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", nargs="+", required=True,
                    help="one or more <prefix>.pred.npy (one per seed)")
    ap.add_argument("--truth", required=True, help="one <prefix>.truth.npy")
    ap.add_argument("--eval-json", default=None,
                    help="the eval.json of the FIRST field, for A2 anchor")
    ap.add_argument("--out-json", default="runs/field_probe.json")
    ap.add_argument("--windows-prefix", default="runs/field_probe")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--k-grid", nargs="*", type=float,
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, truth, n = load_fields(args.fields, args.truth, args.max_windows)
    dtypes = {str(a.dtype) for a in preds} | {str(truth.dtype)}
    if dtypes != {"float32"}:
        raise SystemExit(
            f"dumped fields must be float32, got {sorted(dtypes)}; float16 "
            f"rounds 0.1 mm to 0.0999756 and breaks the per-window invariant "
            f"(tools/_check_field_dtype.py). Re-run evaluate.py --dump-fields.")
    s2 = device_consistency(device)
    print(f"fields={len(preds)} windows={n} shape={preds[0].shape[1:]} "
          f"dtype={sorted(dtypes)} device={device} S2={s2['S2_admissible']}",
          flush=True)

    # ---------------------------------------------------------------- A2
    single = Accumulator()
    pooled = Accumulator()
    fss_radii_all = (3,) + FSS_RADII  # radius 3 is the verifier's default anchor
    fss_acc = {r: {k: [] for k in TIER_KEYS} for r in fss_radii_all}
    # amplitude-control scan: accumulate one Accumulator per k on a SUBSET
    subset = max(1, min(n, 64))
    k_acc = {k: Accumulator() for k in args.k_grid}

    for w in range(n):
        p0 = np.asarray(preds[0][w], dtype=np.float32)
        tr = np.asarray(truth[w], dtype=np.float32)
        single.add(p0, tr)
        pm = maxpool(p0, device)
        pooled.add(pm, tr)
        for r in fss_radii_all:
            for thr, key in zip(TIERS, TIER_KEYS):
                fss_acc[r][key].append(fss(p0, tr, thr, r))
        if w < subset:
            for k in args.k_grid:
                k_acc[k].add(p0 * k, tr)
        if (w + 1) % 100 == 0:
            print(f"  A2 window {w + 1}/{n}", flush=True)

    rep_single, rep_pooled = single.report(), pooled.report()
    pool_bias05 = rep_pooled["0.5"]["bias"]
    # pick the k whose BIAS@0.5 on the subset is closest to the pooled one
    k_reports = {k: a.report() for k, a in k_acc.items()}
    k_star = min(args.k_grid,
                 key=lambda k: abs(k_reports[k]["0.5"]["bias"] - pool_bias05))
    # full-data score at k_star (one extra pass, cheap)
    control = Accumulator()
    for w in range(n):
        control.add(np.asarray(preds[0][w], dtype=np.float32) * k_star,
                    np.asarray(truth[w], dtype=np.float32))
    rep_control = control.report()

    a2 = {
        "anchor": None,
        "single": rep_single,
        "pooled_max3": rep_pooled,
        "amplitude_control": {
            "k_star": k_star,
            "k_grid": args.k_grid,
            "subset_windows_used_to_pick_k": subset,
            "subset_bias05_by_k": {f"{k:g}": k_reports[k]["0.5"]["bias"]
                                   for k in args.k_grid},
            "target_bias05": pool_bias05,
            "full_report": rep_control,
        },
        "fss_multi": {f"r{r}": {k: float(np.mean(v))
                                for k, v in fss_acc[r].items()}
                      for r in fss_radii_all},
        "fss_anchor": None,
        "A2B_verdict": None,
        "A2C_bias_rise": {
            "bias05_single": rep_single["0.5"]["bias"],
            "bias05_pooled": pool_bias05,
            "relative_rise": (pool_bias05 - rep_single["0.5"]["bias"])
                             / rep_single["0.5"]["bias"],
            "bias2_single": rep_single["2.0"]["bias"],
            "bias2_pooled": rep_pooled["2.0"]["bias"],
        },
    }
    if args.eval_json:
        d = json.loads(Path(args.eval_json).read_text(encoding="utf-8"))
        exp = d["thresholds"]["2.0"]["csi"]
        a2["anchor"] = {
            "eval_json_csi2": exp,
            "recomputed_csi2": rep_single["2.0"]["csi"],
            "abs_diff": abs(exp - rep_single["2.0"]["csi"]),
            "pass": abs(exp - rep_single["2.0"]["csi"]) <= 1e-6,
        }
        fss_anchor = {}
        for key in TIER_KEYS:
            got = float(np.mean(fss_acc[3][key]))
            want = d["thresholds"][key]["fss"]
            fss_anchor[key] = {
                "eval_json": want, "recomputed": got,
                "abs_diff": abs(want - got),
                "pass": abs(want - got) <= 1e-6,
            }
        a2["fss_anchor"] = {
            "per_tier": fss_anchor,
            "pass": all(v["pass"] for v in fss_anchor.values()),
        }
    gain_pool = rep_pooled["2.0"]["csi"] - rep_single["2.0"]["csi"]
    gain_ctrl = rep_control["2.0"]["csi"] - rep_single["2.0"]["csi"]
    a2["A2B_verdict"] = {
        "maxpool_gain_csi2": gain_pool,
        "amplitude_control_gain_csi2": gain_ctrl,
        "maxpool_beats_equal_bias_control": gain_pool > gain_ctrl,
        "reading": (
            "相位松弛的抬升不能超过等偏置的纯幅度对照；否则只能记为面积/幅度效应"
            if gain_pool <= gain_ctrl else
            "最大池化超过等偏置幅度对照，但 A2-C 偏置上升仍须披露"
        ),
    }
    write_windows(pooled, f"{args.windows_prefix}_maxpool.windows.json")
    write_windows(control, f"{args.windows_prefix}_ampctrl.windows.json")

    # ---------------------------------------------------------------- A4
    a4 = {"n_seeds": len(preds), "enabled": len(preds) >= 2}
    if a4["enabled"]:
        mean_acc = Accumulator()
        vote_acc = Accumulator()
        per_seed_csi2 = [None] * len(preds)
        seed_accs = [Accumulator() for _ in preds]
        for w in range(n):
            stack = np.stack([np.asarray(a[w], dtype=np.float32) for a in preds])
            tr = np.asarray(truth[w], dtype=np.float32)
            mean_acc.add(stack.mean(axis=0), tr)
            votes = (stack >= VOTE_TIER).sum(axis=0)
            vote_field = np.where(votes >= (len(preds) + 1) // 2,
                                  VOTE_TIER * 1.5, 0.0).astype(np.float32)
            vote_acc.add(vote_field, tr)
            for i in range(len(preds)):
                seed_accs[i].add(stack[i], tr)
            if (w + 1) % 100 == 0:
                print(f"  A4 window {w + 1}/{n}", flush=True)
        mean_rep, vote_rep = mean_acc.report(), vote_acc.report()
        seed_reports = [a.report() for a in seed_accs]
        per_seed_csi2 = [r["2.0"]["csi"] for r in seed_reports]
        seed_pos2 = [r["2.0"]["hits"] + r["2.0"]["false_alarms"]
                     for r in seed_reports]
        mean_pos2 = mean_rep["2.0"]["hits"] + mean_rep["2.0"]["false_alarms"]
        median_pos2 = float(np.median(seed_pos2))
        pos_change = (mean_pos2 - median_pos2) / median_pos2 if median_pos2 else None
        a4.update({
            "mean": mean_rep,
            "vote": vote_rep,
            "vote_note": "多数投票是 2.0 mm/6min 档的二值决策（阈值 VOTE_TIER=2.0，"
                         "通过时赋 3.0 mm）；0.1/0.5 档等价同一掩码，4.0 档恒 0 "
                         "不具判别意义，只报 2.0 档结论。",
            "per_seed_csi2": per_seed_csi2,
            "n_seeds_over_floor": sum(v > FLOOR_K1_CSI2 for v in per_seed_csi2),
            "single_seed_median_csi2": float(np.median(per_seed_csi2)),
            "floor_k1_csi2": FLOOR_K1_CSI2,
            "mean_over_floor": mean_rep["2.0"]["csi"] > FLOOR_K1_CSI2,
            "vote_over_floor": vote_rep["2.0"]["csi"] > FLOOR_K1_CSI2,
            "A4C": {
                "mean_positive_pixels_2mm": mean_pos2,
                "seed_median_positive_pixels_2mm": median_pos2,
                "relative_change": pos_change,
                "smoothed_underforecast": (pos_change is not None
                                           and pos_change < -0.5),
            },
            "A4B": "十场平均只产生一个合成模型，没有跨种子方差可压；"
                   "『压平 9.59× 方差带』登记为未测（需 >=20 种子分两组独立集成）",
        })
        write_windows(mean_acc, f"{args.windows_prefix}_ensemble_mean.windows.json")
        write_windows(vote_acc, f"{args.windows_prefix}_ensemble_vote.windows.json")
        for i, a in enumerate(seed_accs):
            write_windows(a, f"{args.windows_prefix}_seed{i:02d}.windows.json")

    out = {
        "meta": {
            "criteria_timestamp": "2026-11-06 21:40",
            "device": str(device),
            "field_dtype": "float32",
            "S2_device_consistency": s2,
            "n_windows": n,
            "n_fields": len(preds),
            "tiers": list(TIER_KEYS),
            "pool_radius": POOL_RADIUS,
            "note": "offline analysis of dumped float32 whole fields; no model, no training",
        },
        "A2": a2,
        "A4": a4,
    }
    Path(args.out_json).write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 92)
    print(f"A2 anchor              : {a2['anchor']}")
    print(f"A2 CSI@2.0 single/pool : {rep_single['2.0']['csi']:.6f} / "
          f"{rep_pooled['2.0']['csi']:.6f}")
    print(f"A2 equal-bias control  : k*={k_star:g} -> "
          f"{rep_control['2.0']['csi']:.6f}  "
          f"maxpool_beats={a2['A2B_verdict']['maxpool_beats_equal_bias_control']}")
    print(f"A2 BIAS@0.5 rise       : {a2['A2C_bias_rise']['relative_rise']:.3f}")
    print(f"A2 FSS r2/r4/r8 @2.0   : "
          + " / ".join(f"{a2['fss_multi'][f'r{r}']['2.0']:.4f}"
                       for r in FSS_RADII))
    if a4["enabled"]:
        print(f"A4 per-seed CSI@2.0    : "
              + ", ".join(f"{v:.4f}" for v in per_seed_csi2))
        print(f"A4 mean / vote         : {mean_rep['2.0']['csi']:.6f} / "
              f"{vote_rep['2.0']['csi']:.6f}  (floor {FLOOR_K1_CSI2:.6f})")
    print(f"-> {args.out_json}")
    return 0


if __name__ == "__main__":
    main()
