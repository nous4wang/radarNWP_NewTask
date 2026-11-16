#!/usr/bin/env python3
"""Zero-training diagnostic probe for New Ideas 1 & 2:
Decomposing trained deep model failures into (A) In-frame Spatial Ranking vs
(B) Cross-window Scale Collapse vs (C) Historical Trailing-Wake Contamination.

Reads existing dumped prediction fields (or runs 1 forward pass of B0 if specified)
and compares on the EXACT same windows against G_blur and R4_multich_advancing_front.

Usage:
    python tools/_e47_model_vs_proxy_diagnostic_probe.py \
        --pred-npy runs/fields/base_concat_s0.pred.npy \
        --truth-npy runs/fields/base_concat_s0.truth.npy \
        --out /tmp/e47_idea_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

DATA_ROOT = Path("/home/wf/data/radarNWP/BJ_mmap")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.data.dataset import RadarNWPDataset  # noqa: E402
from tools._e46_ranking_and_yhist_probe import (  # noqa: E402
    LABEL_THRESHOLD_MM,
    _bootstrap_ci,
    _exact_topk_csi,
    _zscore_2d,
)


def _find_default_field_pair() -> tuple[Path, Path]:
    """Locate an existing baseline .pred.npy / .truth.npy pair on remote."""
    candidates = [
        (Path("runs/fields/base_concat_s0.pred.npy"), Path("runs/fields/base_concat_s0.truth.npy")),
        (Path("runs/e45/fields/B0_s0_ep00.pred.npy"), Path("runs/e45/fields/B0_s0_ep00.truth.npy")),
        (Path("runs/e44/fields/B0_s0_ep00.pred.npy"), Path("runs/e44/fields/B0_s0_ep00.truth.npy")),
        (Path("runs/e43/fields/B0_s0_ep00.pred.npy"), Path("runs/e43/fields/B0_s0_ep00.truth.npy")),
    ]
    for p, t in candidates:
        if p.is_file() and t.is_file():
            return p, t
    # Fallback glob search under runs/
    for p in sorted(Path("runs").rglob("*B0*ep00*.pred.npy")):
        t = p.with_name(p.name.replace(".pred.npy", ".truth.npy"))
        if t.is_file():
            return p, t
    for p in sorted(Path("runs").rglob("*.pred.npy")):
        t = p.with_name(p.name.replace(".pred.npy", ".truth.npy"))
        if t.is_file():
            return p, t
    raise FileNotFoundError("No *.pred.npy / *.truth.npy pair found under runs/")


def _contingency_counts(pred_bin: np.ndarray, truth_bin: np.ndarray) -> tuple[int, int, int]:
    h = int(np.logical_and(pred_bin, truth_bin).sum())
    m = int(np.logical_and(~pred_bin, truth_bin).sum())
    f = int(np.logical_and(pred_bin, ~truth_bin).sum())
    return h, m, f


def _csi_from_hmf(h: float, m: float, f: float) -> float:
    den = h + m + f
    return float(h / den) if den > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-npy", default=None)
    ap.add_argument("--truth-npy", default=None)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="/tmp/e47_idea_probe.json")
    args = ap.parse_args()

    if args.pred_npy and args.truth_npy:
        pred_path, truth_path = Path(args.pred_npy), Path(args.truth_npy)
    else:
        pred_path, truth_path = _find_default_field_pair()

    pred_stack = np.load(pred_path, mmap_mode="r")    # (N_win, 10, 301, 301)
    truth_stack = np.load(truth_path, mmap_mode="r")  # (N_win, 10, 301, 301)
    n_win, t_out, h, w = pred_stack.shape

    # Load full-field dataset metadata to align each window's t0 radar input
    ds = RadarNWPDataset(
        args.data_root,
        packed=True,
        patch_size=h,
        random_crop=False,
        split=args.split,
        slide_stride=h,
    )

    n_align = min(n_win, len(ds))

    # 1. Global contingency accumulators for Idea 1 (Scale vs Ranking Decomposition)
    hmf_raw = [0, 0, 0]
    hmf_radar_q = [0, 0, 0]      # Idea 1: Window-adaptive quantile anchored by t0 radar wet fraction
    hmf_oracle_k = [0, 0, 0]     # Upper bound of model's own spatial ranking

    # Global S3 equal-bias multiplier on the model prediction
    total_truth_pos = int((truth_stack[:n_align] >= LABEL_THRESHOLD_MM).sum())
    flat_sample = np.asarray(pred_stack[:n_align:4], dtype=np.float32).reshape(-1)
    target_frac = total_truth_pos / float(n_align * t_out * h * w)
    s3_thresh = float(np.quantile(flat_sample, max(0.0, 1.0 - target_frac)))
    hmf_s3_global = [0, 0, 0]

    # 2. Per-window Exact Top-K CSI comparison: Model vs G_blur vs R4_advancing_front (Idea 2)
    win_topk_model = []
    win_topk_gblur = []
    win_topk_r4 = []
    win_topk_hybrid = []         # Blending model ranking + R4 advancing front ranking

    # 3. Trailing-wake vs advancing-front correlation of model predictions
    corr_with_old_wake = []
    corr_with_new_front = []

    for wi in range(n_align):
        P_win = np.asarray(pred_stack[wi], dtype=np.float64)   # (10, H, W)
        T_win = np.asarray(truth_stack[wi], dtype=np.float64)  # (10, H, W)

        sample = ds[wi]
        # x_radar is normalized log1p(dBZ)/6.0 -> invert to dBZ: expm1(x * 6.0)
        x_rad_norm = sample["x_radar"].numpy()                 # (11, 10, H, W)
        R_hist = np.expm1(np.clip(x_rad_norm * 6.0, 0.0, 10.0))

        cr_t0 = R_hist[-1, 6]
        cr_tm2 = R_hist[-3, 6]
        cr_old = R_hist[:4, 6].mean(axis=0)  # t-60 .. t-42 min old wake

        G_blur = ndi.gaussian_filter(cr_t0, sigma=2.0)

        def _mc(step: int) -> np.ndarray:
            f = (
                0.35 * _zscore_2d(R_hist[step, 6])
                + 0.30 * _zscore_2d(R_hist[step, 9])
                + 0.25 * _zscore_2d(R_hist[step, 8])
                + 0.10 * _zscore_2d(R_hist[step, 7])
            )
            return ndi.gaussian_filter(f, sigma=2.0)

        mc_t0 = _mc(-1)
        mc_tm2 = _mc(-3)
        front_pos = np.maximum(0.0, mc_t0 - mc_tm2)
        R4 = mc_t0 + 1.0 * front_pos

        # Measure wake vs advancing front correlation on wet windows
        p_mean = P_win.mean(axis=0)
        if p_mean.std() > 1e-9:
            wake_only = np.maximum(0.0, ndi.gaussian_filter(cr_old, sigma=2.0) - G_blur)
            if wake_only.std() > 1e-9:
                corr_with_old_wake.append(float(np.corrcoef(p_mean.reshape(-1), wake_only.reshape(-1))[0, 1]))
            if front_pos.std() > 1e-9:
                corr_with_new_front.append(float(np.corrcoef(p_mean.reshape(-1), front_pos.reshape(-1))[0, 1]))

        # Radar-anchored wet fraction for Idea 1 (using radar CR >= 33 dBZ as physical prior)
        rad_wet_k = int((G_blur >= 33.0).sum())

        m_csis, gb_csis, r4_csis, hyb_csis = [], [], [], []

        for lead in range(t_out):
            p_f = P_win[lead]
            t_f = T_win[lead]
            t_bin = t_f >= LABEL_THRESHOLD_MM

            # (1) Raw threshold CSI@2.0
            h0, m0, f0 = _contingency_counts(p_f >= LABEL_THRESHOLD_MM, t_bin)
            hmf_raw[0] += h0; hmf_raw[1] += m0; hmf_raw[2] += f0

            # (2) Global S3 equal-bias threshold CSI@2.0
            hs, ms, fs = _contingency_counts(p_f >= s3_thresh, t_bin)
            hmf_s3_global[0] += hs; hmf_s3_global[1] += ms; hmf_s3_global[2] += fs

            # (3) Idea 1: Window-local quantile threshold anchored by t0 radar wet area
            if rad_wet_k > 0:
                flat_p = p_f.reshape(-1)
                top_r_idx = np.argsort(flat_p, kind="mergesort")[-rad_wet_k:]
                p_rad_bin = np.zeros_like(flat_p, dtype=bool)
                p_rad_bin[top_r_idx] = True
                p_rad_bin = p_rad_bin.reshape(t_bin.shape)
            else:
                p_rad_bin = np.zeros_like(t_bin, dtype=bool)
            hr, mr, fr = _contingency_counts(p_rad_bin, t_bin)
            hmf_radar_q[0] += hr; hmf_radar_q[1] += mr; hmf_radar_q[2] += fr

            # (4) Exact Top-K spatial ranking evaluation on wet frames
            k_true = int(t_bin.sum())
            if k_true > 0:
                flat_p = p_f.reshape(-1)
                top_k_idx = np.argsort(flat_p, kind="mergesort")[-k_true:]
                p_ok_bin = np.zeros_like(flat_p, dtype=bool)
                p_ok_bin[top_k_idx] = True
                ho, mo, fo = _contingency_counts(p_ok_bin.reshape(t_bin.shape), t_bin)
                hmf_oracle_k[0] += ho; hmf_oracle_k[1] += mo; hmf_oracle_k[2] += fo

            if k_true >= 5:
                m_csis.append(_exact_topk_csi(p_f, t_bin, k_true))
                gb_csis.append(_exact_topk_csi(G_blur, t_bin, k_true))
                r4_csis.append(_exact_topk_csi(R4, t_bin, k_true))
                # Rank fusion of model prediction and R4 advancing front
                hyb_field = 0.5 * _zscore_2d(p_f) + 0.5 * _zscore_2d(R4)
                hyb_csis.append(_exact_topk_csi(hyb_field, t_bin, k_true))

        if m_csis:
            win_topk_model.append(float(np.mean(m_csis)))
            win_topk_gblur.append(float(np.mean(gb_csis)))
            win_topk_r4.append(float(np.mean(r4_csis)))
            win_topk_hybrid.append(float(np.mean(hyb_csis)))

    arr_m = np.array(win_topk_model)
    arr_gb = np.array(win_topk_gblur)
    arr_r4 = np.array(win_topk_r4)
    arr_hyb = np.array(win_topk_hybrid)

    report = {
        "tool": "tools/_e47_model_vs_proxy_diagnostic_probe.py",
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "pred_file": str(pred_path),
        "n_windows_aligned": n_align,
        " n_wet_windows_ranked": len(arr_m),
        "Idea1_Scale_vs_Ranking_Decomposition_CSI2p0": {
            "1_raw_model_csi2p0": _csi_from_hmf(*hmf_raw),
            "1_raw_model_bias2p0": float((hmf_raw[0] + hmf_raw[2]) / max(hmf_raw[0] + hmf_raw[1], 1)),
            "2_global_S3_equal_bias_csi2p0": _csi_from_hmf(*hmf_s3_global),
            "3_window_radar_quantile_anchored_csi2p0": _csi_from_hmf(*hmf_radar_q),
            "4_exact_per_frame_topk_ranking_ceiling_csi2p0": _csi_from_hmf(*hmf_oracle_k),
        },
        "Idea2_Spatial_Ranking_Model_vs_R4_Advancing_Front": {
            "model_exact_topk_mean": float(arr_m.mean()),
            "G_blur_exact_topk_mean": float(arr_gb.mean()),
            "R4_advancing_front_exact_topk_mean": float(arr_r4.mean()),
            "Hybrid_Model_plus_R4_exact_topk_mean": float(arr_hyb.mean()),
            "diff_R4_minus_Model": {
                "mean": float((arr_r4 - arr_m).mean()),
                "ci95": _bootstrap_ci(arr_r4 - arr_m),
            },
            "diff_Hybrid_minus_Model": {
                "mean": float((arr_hyb - arr_m).mean()),
                "ci95": _bootstrap_ci(arr_hyb - arr_m),
            },
        },
        "Trailing_Wake_Diagnostic": {
            "corr_model_with_old_wake_t60_t42": float(np.mean(corr_with_old_wake)) if corr_with_old_wake else None,
            "corr_model_with_new_front_t0_tm12": float(np.mean(corr_with_new_front)) if corr_with_new_front else None,
        },
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())