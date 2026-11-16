#!/usr/bin/env python3
"""Read-only multi-seed audit on EXACT E45 B0 (s0..s2) and base_concat (s0..s9) fields.

Tests whether the 9.59x seed variance in CSI@2.0 is caused by:
(H_scale) Seed-to-seed output scale/BIAS drift while spatial ranking is stable, OR
(H_feat)  True collapse of learned spatial ranking across seeds.

Zero training, zero model forward pass. Replication unit = SEED.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import numpy as np

LABEL_THRESHOLD_MM = 2.0


def _audit_one_seed(pred_path: Path, truth_path: Path) -> dict[str, float]:
    pred = np.load(pred_path, mmap_mode="r")    # (N_win, 10, 301, 301)
    truth = np.load(truth_path, mmap_mode="r")  # (N_win, 10, 301, 301)
    n_win, t_out, h, w = pred.shape

    t_bin_all = truth >= LABEL_THRESHOLD_MM
    total_truth_pos = int(t_bin_all.sum())
    target_frac = total_truth_pos / float(n_win * t_out * h * w)

    # S3 equal-bias threshold (matching BIAS@2.0 == 1.0 on this seed's field)
    flat_sub = np.asarray(pred[::2], dtype=np.float32).reshape(-1)
    s3_thr = float(np.quantile(flat_sub, max(0.0, 1.0 - target_frac)))

    h_raw = m_raw = f_raw = 0
    h_s3 = m_s3 = f_s3 = 0
    h_topk = m_topk = f_topk = 0

    for wi in range(n_win):
        P_w = np.asarray(pred[wi], dtype=np.float32)
        T_w = t_bin_all[wi]

        # Raw @ 2.0 mm
        p_raw = P_w >= LABEL_THRESHOLD_MM
        h_raw += int(np.logical_and(p_raw, T_w).sum())
        m_raw += int(np.logical_and(~p_raw, T_w).sum())
        f_raw += int(np.logical_and(p_raw, ~T_w).sum())

        # Equal-bias @ s3_thr (BIAS == 1.0)
        p_s3 = P_w >= s3_thr
        h_s3 += int(np.logical_and(p_s3, T_w).sum())
        m_s3 += int(np.logical_and(~p_s3, T_w).sum())
        f_s3 += int(np.logical_and(p_s3, ~T_w).sum())

        # Per-frame Exact Top-K spatial ranking ceiling
        for l in range(t_out):
            tb = T_w[l]
            k = int(tb.sum())
            if k > 0:
                flat_p = P_w[l].reshape(-1)
                idx = np.argsort(flat_p, kind="mergesort")[-k:]
                hits = int(tb.reshape(-1)[idx].sum())
                h_topk += hits
                m_topk += (k - hits)
                f_topk += (k - hits)

    def _csi(h: int, m: int, f: int) -> float:
        return float(h / (h + m + f)) if (h + m + f) > 0 else 0.0

    raw_bias = float((h_raw + f_raw) / max(h_raw + m_raw, 1))
    return {
        "raw_csi2p0": _csi(h_raw, m_raw, f_raw),
        "raw_bias2p0": raw_bias,
        "equal_bias_s3_thr_mm": s3_thr,
        "equal_bias_csi2p0": _csi(h_s3, m_s3, f_s3),
        "exact_topk_ceiling_csi2p0": _csi(h_topk, m_topk, f_topk),
    }


def _summarize_group(rows: dict[str, dict[str, float]]) -> dict:
    if not rows:
        return {}
    keys = ("raw_csi2p0", "raw_bias2p0", "equal_bias_csi2p0", "exact_topk_ceiling_csi2p0")
    summary = {}
    for k in keys:
        vals = np.array([v[k] for v in rows.values()], dtype=np.float64)
        summary[k] = {
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "max_over_min_ratio": float(vals.max() / max(vals.min(), 1e-9)),
            "std": float(vals.std()),
        }
    return summary


def main() -> int:
    e45_b0 = {}
    for s in (0, 1, 2):
        p = Path(f"runs/e45/fields/B0_s{s}_ep00.pred.npy")
        t = Path(f"runs/e45/fields/B0_s{s}_ep00.truth.npy")
        if p.is_file() and t.is_file():
            e45_b0[f"E45_B0_s{s}"] = _audit_one_seed(p, t)

    base_concat_10 = {}
    for s in range(10):
        p = Path(f"runs/fields/base_concat_s{s}.pred.npy")
        t = Path(f"runs/fields/base_concat_s{s}.truth.npy")
        if p.is_file() and t.is_file():
            base_concat_10[f"base_concat_s{s}"] = _audit_one_seed(p, t)

    out = {
        "tool": "tools/_e47_multiseed_ranking_audit.py",
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "E45_B0_3seeds": {"per_seed": e45_b0, "summary": _summarize_group(e45_b0)},
        "base_concat_10seeds": {"per_seed": base_concat_10, "summary": _summarize_group(base_concat_10)},
    }
    out_path = Path("/tmp/e47_multiseed_audit.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["E45_B0_3seeds"]["summary"], indent=2))
    print(json.dumps(out["base_concat_10seeds"]["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
