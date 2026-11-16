#!/usr/bin/env python3
"""E24 / MASTER 4.26 -- per-case Z-R bias decomposition (zero-compute).

Splits the aggregate "train BIAS@0.5 = 1.7820 / test = 0.9048" drift into
one number per precipitation case, for train / val / test.  Uses exactly the
analytical persistence field, dataset, stitcher, verifier and thresholds as
tools/_scan_zr_k.py, so the k=1 anchors must reproduce bit-for-bit.

No training, no model, no learned parameter.  This is a *descriptive*
decomposition: it cannot identify the cause of the epoch-0 peak (E24 A1-B).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radar_nwp_bench.data import (  # noqa: E402
    NWPAlignment,
    RadarNWPDataset,
    build_dataloader,
)
from radar_nwp_bench.metrics import PatchStitcher  # noqa: E402

THRESHOLDS = (0.1, 0.5, 2.0, 4.0)
TIERS = ("0.1", "0.5", "2.0", "4.0")


class Counter:
    """Threshold contingency counts only.

    PrecipitationVerifier also recomputes a neighbourhood FSS on every
    update; E24 needs none of that, and on 39 cases x 4 thresholds it made
    the probe CPU-bound for >10 min.  This keeps the exact same contingency
    arithmetic (and therefore the same CSI / BIAS) with no FSS work.
    """

    def __init__(self) -> None:
        self.n = 0
        self.total = {
            k: {c: 0.0 for c in
                ("hits", "misses", "false_alarms", "correct_negatives")}
            for k in TIERS
        }

    def update(self, pred: np.ndarray, truth: np.ndarray) -> None:
        self.n += 1
        for thr, key in zip(THRESHOLDS, TIERS):
            p = pred >= thr
            t = truth >= thr
            c = self.total[key]
            c["hits"] += float(np.count_nonzero(p & t))
            c["misses"] += float(np.count_nonzero(~p & t))
            c["false_alarms"] += float(np.count_nonzero(p & ~t))
            c["correct_negatives"] += float(np.count_nonzero(~p & ~t))

    def report(self) -> dict:
        out = {"n_samples": self.n}
        for key in TIERS:
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
# Independent anchors, from MASTER 4.18.5.5 / 4.15.5.
ANCHORS = {
    "test": {"csi2": 0.0924188, "bias05": 0.9048},
    "train": {"csi2": 0.0928886, "bias05": 1.7820045},
    "val": None,
}
SPLITS = ("train", "val", "test")


def zr_to_rain(dbz: torch.Tensor) -> torch.Tensor:
    """Z = 200 R^1.6, hourly rate -> 6-minute accumulation (x0.1)."""
    linear_z = torch.pow(10.0, dbz.clamp_min(0.0) / 10.0)
    return torch.pow(linear_z / 200.0, 1.0 / 1.6) * 0.1


def verifier_row(counter: Counter) -> dict:
    return counter.report()


def pooled(case_rows: dict[str, dict], tier: str) -> float:
    """Aggregate BIAS from summed contingency counts (not a mean of ratios)."""
    h = sum(r[tier]["hits"] for r in case_rows.values())
    m = sum(r[tier]["misses"] for r in case_rows.values())
    f = sum(r[tier]["false_alarms"] for r in case_rows.values())
    return (h + f) / (h + m) if (h + m) > 0 else float("nan")


def run_split(args, split: str) -> dict:
    dataset = RadarNWPDataset(
        args.data_root,
        packed=True,
        in_len=11,
        out_len=10,
        patch_size=args.patch_size,
        alignment=NWPAlignment.FORECAST,
        split=split,
        slide_stride=args.slide_stride,
        random_crop=False,
        tolerate_one_missing_frame=True,
    )
    loader = build_dataloader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, seed=0,
    )
    grid_h, grid_w = dataset.grid_shape
    cr_index = list(dataset.radar_vars).index("CR")
    n_pos = dataset.n_patch_positions

    # With patch_size >= 301 the dataset returns the whole field in one tile,
    # so no Hann stitching is needed.  For the analytical persistence field
    # every overlapping tile carries identical pixels anyway, hence the
    # stitched and unstitched counts coincide -- but we VERIFY that with the
    # k=1 anchors below before trusting E24's numbers.
    stitcher = None
    if n_pos > 1:
        stitcher = PatchStitcher((grid_h, grid_w), n_frames=10, n_positions=n_pos)
    overall = Counter()
    per_case: dict[str, Counter] = {}

    def absorb(cid: str, sp: np.ndarray, st: np.ndarray) -> None:
        overall.update(sp, st)
        if cid not in per_case:
            per_case[cid] = Counter()
        per_case[cid].update(sp, st)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in loader:
            x_radar = batch["x_radar"].to(device)
            y_true = batch["y_rain"].to(device)
            case_idx = batch["case_index"].cpu().numpy()
            pos_idx = batch["position_index"].cpu().numpy()
            origin = batch["patch_origin"].cpu().numpy()

            last_dbz = torch.expm1(
                x_radar[:, -1, cr_index:cr_index + 1].clamp_min(0.0) * 6.0
            ).clamp(0.0, 75.0)
            zr = zr_to_rain(last_dbz)  # (B, 1, H, W)
            pred = zr.unsqueeze(1).expand(-1, y_true.shape[1], -1, -1, -1)
            pred_np = pred.cpu().numpy()
            true_np = y_true.cpu().numpy()

            for b in range(pred_np.shape[0]):
                s_i = int(case_idx[b])
                cid = dataset.samples[s_i].case_path.stem
                if stitcher is None:
                    absorb(cid, pred_np[b, :, 0], true_np[b, :, 0])
                    continue
                stitcher.add(
                    s_i, int(pos_idx[b]),
                    (int(origin[b][0]), int(origin[b][1])),
                    pred_np[b, :, 0], true_np[b, :, 0],
                )
                if stitcher.is_complete(s_i):
                    sp, st = stitcher.finalize(s_i)
                    absorb(cid, sp, st)
                    stitcher.release(s_i)

    cases = {cid: verifier_row(v) for cid, v in sorted(per_case.items())}
    anchor_row = verifier_row(overall)
    anchor = ANCHORS[split]
    anchor_check = None
    if anchor is not None:
        anchor_check = {
            "expected": anchor,
            "observed": {
                "csi2": anchor_row["2.0"]["csi"],
                "bias05": anchor_row["0.5"]["bias"],
            },
            "pass": (
                abs(anchor_row["2.0"]["csi"] - anchor["csi2"]) <= 0.002
                and abs(anchor_row["0.5"]["bias"] - anchor["bias05"]) <= 0.01
            ),
        }
    print(f"[{split}] n_cases={len(cases)} n_windows={anchor_row['n_samples']} "
          f"CSI@2.0={anchor_row['2.0']['csi']:.6f} "
          f"BIAS@0.5={anchor_row['0.5']['bias']:.6f} "
          f"anchor={anchor_check['pass'] if anchor_check else 'n/a'}", flush=True)
    return {
        "n_cases": len(cases),
        "n_windows": anchor_row["n_samples"],
        "anchor": anchor_check,
        "overall": anchor_row,
        "cases": cases,
        "n_patch_positions": n_pos,
        "pending_stitcher_windows": (len(stitcher.pending) if stitcher else 0),
    }


def a1a_analysis(splits: dict) -> dict:
    """Distribution shape of per-case BIAS@0.5 and the top-3 sensitivity test."""
    out = {}
    for name, block in splits.items():
        vals = [r["0.5"]["bias"] for r in block["cases"].values()]
        vals = [v for v in vals if np.isfinite(v)]
        out[name] = {
            "n": len(vals),
            "median": statistics.median(vals) if vals else None,
            "q1": float(np.percentile(vals, 25)) if vals else None,
            "q3": float(np.percentile(vals, 75)) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "n_above_1": sum(v > 1.0 for v in vals),
            "n_below_1": sum(v < 1.0 for v in vals),
            "pooled_bias05": pooled(block["cases"], "0.5"),
            "pooled_csi2": None,  # descriptive only; see E24 limits
        }
    if "train" not in splits or "test" not in splits:
        out["gap_analysis"] = {"skipped": "smoke run without both train and test"}
        out["shift_note"] = "smoke run"
        return out
    train, test = splits["train"]["cases"], splits["test"]["cases"]
    full_gap = pooled(train, "0.5") - pooled(test, "0.5")
    top3 = sorted(train, key=lambda c: train[c]["0.5"]["bias"], reverse=True)[:3]
    reduced = {c: r for c, r in train.items() if c not in top3}
    shrunk_gap = pooled(reduced, "0.5") - pooled(test, "0.5")
    out["gap_analysis"] = {
        "full_gap_bias05": full_gap,
        "train_top3_cases": top3,
        "gap_after_dropping_train_top3": shrunk_gap,
        "relative_shrink": (full_gap - shrunk_gap) / full_gap if full_gap else None,
        "verdict": (
            "少数个例主导" if full_gap and (full_gap - shrunk_gap) / full_gap >= 0.5
            else "未达『少数个例主导』门槛（≥50% 缩小）"
        ),
    }
    # >=80% of cases shift in the same direction, case-wise paired by rank is
    # not possible (different cases), so report per-split fraction instead.
    out["shift_note"] = (
        "训练与测试是不同个例，无法逐 case 配对；A1-A 的『≥80% 同向』"
        "按各划分内 BIAS@0.5 > 1 的占比描述，不作因果判定。"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--slide-stride", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS),
                    help="subset of train/val/test; used for a fast smoke run")
    ap.add_argument("--out-json", default="runs/case_zr_bias.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    splits = {}
    for split in args.splits:
        splits[split] = run_split(args, split)

    result = {
        "meta": {
            "experiment": "E24 / MASTER 4.26",
            "criteria_timestamp": "2026-11-06 21:40",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "note": "analytical Z-R persistence, k=1, no model, no training",
            "thresholds": list(TIERS),
        },
        "splits": splits,
        "A1A": a1a_analysis(splits),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 88)
    print("A1-A 分布形态")
    for name in args.splits:
        d = result["A1A"][name]
        print(f"  {name:>5}: n={d['n']:>2} median={d['median']:.4f} "
              f"IQR=[{d['q1']:.4f},{d['q3']:.4f}] "
              f"min={d['min']:.4f} max={d['max']:.4f} "
              f">1:{d['n_above_1']} <1:{d['n_below_1']} "
              f"pooled={d['pooled_bias05']:.4f}")
    g = result["A1A"]["gap_analysis"]
    if "skipped" not in g:
        print(f"  gap={g['full_gap_bias05']:.4f} "
              f"after_drop_top3={g['gap_after_dropping_train_top3']:.4f} "
              f"shrink={g['relative_shrink']:.3f} -> {g['verdict']}")
        print(f"  top3 = {g['train_top3_cases']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    main()
