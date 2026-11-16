#!/usr/bin/env python3
"""Zero-training mechanism probe for Candidates A, B, and C (y_hist assimilation).

Evaluates directly on the 10-frame FUTURE forecast horizon (t+6 .. t+60 min)
across the TRAIN split (23 cases) using EXACT Top-K spatial ranking
(deterministic tie-breaking -> BIAS == 1.000000 on every single frame).

Probes:
- Baseline G0 : t0 single-frame CR raw vs G_blur (sigma=2.0).
- Probe A     : Multi-frame temporal residence (11-frame EMA) & multi-channel
                (CR + VIL + HBR + ET) spatial ranking.
- Probe B     : NWP moisture/convergence spatial modulation (PWAT, RH700/850,
                DVG925, CAPE) suppressing dry/overhanging radar false alarms.
- Probe C     : Rational usage of historical ground truth `y_hist` under
                realistic operational lags (lag=0, lag=2 [12m], lag=4 [24m]):
                raw persistence vs multi-frame gauge residence vs
                radar-gauge geometric fusion anchor.

Usage:
    python tools/_e46_ranking_and_yhist_probe.py --out /tmp/e46_abc_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

DATA_ROOT = Path("/home/wf/data/radarNWP/BJ_mmap")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.data.index import split_cases  # noqa: E402

# Frozen probe constants
T_IN = 11
T_OUT = 10
MAX_WINDOWS_PER_CASE = 20
LABEL_THRESHOLD_MM = 2.0
MIN_WET_PIXELS = 20          # ~0.022% of 301x301 grid
N_BOOTSTRAP = 10000
BOOT_SEED = 20261108
MIN_CASES = 20


def _bootstrap_ci(values: np.ndarray, seed: int = BOOT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _zscore_2d(field: np.ndarray) -> np.ndarray:
    """Within-frame spatial standardization (immune to train/test calibration shift)."""
    x = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    s = float(x.std())
    if s <= 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / s


def _minmax_pos(field: np.ndarray) -> np.ndarray:
    """Map spatial field to [0, 1] smoothly via rank-preserving sigmoid of z-score."""
    z = _zscore_2d(field)
    return 1.0 / (1.0 + np.exp(-z))


def _exact_topk_csi(proxy: np.ndarray, target_mask: np.ndarray, k: int) -> float:
    """Exact Top-K CSI with deterministic index tie-breaking (guarantees BIAS == 1.0)."""
    if k <= 0:
        return 0.0
    flat = np.nan_to_num(proxy.reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    # argsort with kind='mergesort' breaks ties deterministically by pixel index,
    # selecting EXACTLY k pixels every time (BIAS === 1.000000).
    top_idx = np.argsort(flat, kind="mergesort")[-k:]
    t_flat = target_mask.reshape(-1)
    hits = float(t_flat[top_idx].sum())
    # Since |pred| == k and |target| == k, union = 2*k - hits
    union = float(2 * k - hits)
    return hits / union if union > 0.0 else 0.0


def _nwp_to_radar_grid(nwp_slice: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Align NWP slice to (301, 301) radar grid if shapes differ."""
    if nwp_slice.shape == target_hw:
        return nwp_slice
    # If raw 167x167 @ 0.03 deg -> x3 bilinear upsample (501x501) -> center crop (301x301)
    h, w = nwp_slice.shape
    if h == 167 and w == 167 and target_hw == (301, 301):
        up = ndi.zoom(nwp_slice, zoom=3.0, order=1)
        y0 = (up.shape[0] - 301) // 2
        x0 = (up.shape[1] - 301) // 2
        return up[y0 : y0 + 301, x0 : x0 + 301]
    return ndi.zoom(
        nwp_slice,
        zoom=(target_hw[0] / h, target_hw[1] / w),
        order=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default="/tmp/e46_abc_probe.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)

    report: dict = {
        "tool": "tools/_e46_ranking_and_yhist_probe.py",
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": (
            "10-frame future horizon (t+6..t+60) exact Top-K (BIAS===1.0) "
            "evaluation on TRAIN split (23 cases)."
        ),
        "arms": {},
        "comparisons": {},
    }

    try:
        all_cases = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and (p / "radar.npy").is_file()
        )
        parts = split_cases(all_cases)
        train_cases = list(parts["train"])

        arm_names = [
            "G0_cr_t0_raw",
            "G_blur_cr_t0_s2",
            # Candidate A: Temporal Residence & Multi-channel Radar
            "A1_cr_ema6_s2",
            "A2_multich_t0_s2",
            "A3_multich_ema6_s2",
            # Candidate B: NWP Moisture / Convergence Spatial Ranking
            "B1_cr_x_nwp_moisture",
            "B2_multich_ema_x_nwp",
            # Candidate C: Rational Historical Ground-Truth (y_hist) Usage
            "C1_yhist_lag0_raw",
            "C2_yhist_lag0_ema3_s2",
            "C3_hybrid_lag0_yhist_radar",
            "C4_hybrid_lag2_12min_yhist_radar",
            "C5_hybrid_lag4_24min_yhist_radar",
        ]
        per_case_scores: list[dict[str, float]] = []

        for case_id in train_cases:
            case_dir = data_root / case_id
            meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
            radar = np.load(case_dir / "radar.npy", mmap_mode="r")
            label = np.load(case_dir / "label.npy", mmap_mode="r")
            nwp_path = case_dir / "nwp.npy"
            nwp = np.load(nwp_path, mmap_mode="r") if nwp_path.is_file() else None

            rv = [str(v) for v in meta.get("radar_vars", [])]
            nv = [str(v) for v in meta.get("nwp_vars", [])]

            cr_idx = rv.index("CR") if "CR" in rv else 6
            vil_idx = rv.index("VIL") if "VIL" in rv else 9
            hbr_idx = rv.index("HBR") if "HBR" in rv else 8
            et_idx = rv.index("ET") if "ET" in rv else 7

            r_times = [str(t) for t in meta.get("radar_times", [])]
            l_times = [str(t) for t in meta.get("label_times", [])]
            l_map = {t: i for i, t in enumerate(l_times)}

            # Build valid contiguous 21-frame windows (11 history + 10 future)
            valid_t0_indices = []
            for idx_t0 in range(T_IN - 1, len(r_times) - T_OUT):
                win_stamps = r_times[idx_t0 - (T_IN - 1) : idx_t0 + T_OUT + 1]
                if all(s in l_map for s in win_stamps):
                    valid_t0_indices.append(idx_t0)

            if not valid_t0_indices:
                continue
            if len(valid_t0_indices) > MAX_WINDOWS_PER_CASE:
                pick = np.linspace(0, len(valid_t0_indices) - 1, MAX_WINDOWS_PER_CASE).astype(int)
                valid_t0_indices = [valid_t0_indices[int(i)] for i in pick]

            hw = tuple(label.shape[-2:])

            # Extract NWP moisture/convergence spatial potential for this case if available
            nwp_mod = np.ones(hw, dtype=np.float64)
            if nwp is not None and len(nv) > 0:
                nwp_mid = np.nan_to_num(np.asarray(nwp[len(nwp) // 2], dtype=np.float64), nan=0.0)
                mod_fields = []
                for vname, sign in [("PWAT", 1.0), ("RH850", 1.0), ("RH700", 1.0), ("CAPE", 1.0), ("DVG925", -1.0)]:
                    if vname in nv:
                        f2d = _nwp_to_radar_grid(nwp_mid[nv.index(vname)], hw)
                        mod_fields.append(_minmax_pos(sign * f2d))
                if mod_fields:
                    nwp_mod = np.mean(mod_fields, axis=0)

            case_accum = {k: [] for k in arm_names}

            # Exponential weights for 6-frame history (t_{-30}..t_0)
            ema6_w = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.32], dtype=np.float64)
            ema3_w = np.array([0.20, 0.30, 0.50], dtype=np.float64)

            for idx_t0 in valid_t0_indices:
                hist_r_idx = list(range(idx_t0 - (T_IN - 1), idx_t0 + 1))
                fut_r_idx = list(range(idx_t0 + 1, idx_t0 + T_OUT + 1))
                hist_l_idx = [l_map[r_times[i]] for i in hist_r_idx]
                fut_l_idx = [l_map[r_times[i]] for i in fut_r_idx]

                # Future 10-frame ground truth
                Y_fut = np.nan_to_num(np.asarray(label[fut_l_idx, 0], dtype=np.float64), nan=0.0)
                total_wet = int(np.sum(Y_fut >= LABEL_THRESHOLD_MM))
                if total_wet < MIN_WET_PIXELS:
                    continue

                # History 11-frame radar & ground truth
                R_hist = np.nan_to_num(np.asarray(radar[hist_r_idx], dtype=np.float64), nan=0.0)
                Y_hist = np.nan_to_num(np.asarray(label[hist_l_idx, 0], dtype=np.float64), nan=0.0)

                # --- Construct Spatial Ranking Proxies at issue time t0 ---
                cr_t0 = R_hist[-1, cr_idx]
                G0 = cr_t0
                G_blur = ndi.gaussian_filter(cr_t0, sigma=2.0)

                # A1: 6-frame temporal residence EMA of CR + blur
                cr_ema6 = np.tensordot(ema6_w, R_hist[-6:, cr_idx], axes=(0, 0))
                A1 = ndi.gaussian_filter(cr_ema6, sigma=2.0)

                # A2: Multi-channel (CR + VIL + HBR + ET) normalized fusion at t0
                mc_t0 = (
                    0.35 * _zscore_2d(R_hist[-1, cr_idx])
                    + 0.30 * _zscore_2d(R_hist[-1, vil_idx])
                    + 0.25 * _zscore_2d(R_hist[-1, hbr_idx])
                    + 0.10 * _zscore_2d(R_hist[-1, et_idx])
                )
                A2 = ndi.gaussian_filter(mc_t0, sigma=2.0)

                # A3: Multi-channel + 6-frame temporal residence EMA
                mc_seq = [
                    0.35 * _zscore_2d(R_hist[t, cr_idx])
                    + 0.30 * _zscore_2d(R_hist[t, vil_idx])
                    + 0.25 * _zscore_2d(R_hist[t, hbr_idx])
                    + 0.10 * _zscore_2d(R_hist[t, et_idx])
                    for t in range(T_IN - 6, T_IN)
                ]
                A3 = ndi.gaussian_filter(np.tensordot(ema6_w, np.asarray(mc_seq), axes=(0, 0)), sigma=2.0)

                # B1 & B2: NWP moisture/convergence spatial modulation
                B1 = _minmax_pos(G_blur) * (0.7 + 0.3 * nwp_mod)
                B2 = _minmax_pos(A3) * (0.7 + 0.3 * nwp_mod)

                # C1..C5: Historical Ground-Truth (y_hist) Anchors
                C1 = Y_hist[-1]
                y_ema3_lag0 = np.tensordot(ema3_w, Y_hist[-3:], axes=(0, 0))
                C2 = ndi.gaussian_filter(y_ema3_lag0, sigma=2.0)

                # C3 (lag=0): Geometric/additive rank fusion of y_hist(t0) and radar A3(t0)
                C3 = 0.65 * _minmax_pos(C2) + 0.35 * _minmax_pos(A3)

                # C4 (lag=2, 12-min operational delay: only Y_hist[:-2] available)
                y_lag2 = ndi.gaussian_filter(np.tensordot(ema3_w, Y_hist[-5:-2], axes=(0, 0)), sigma=2.5)
                C4 = 0.55 * _minmax_pos(y_lag2) + 0.45 * _minmax_pos(A3)

                # C5 (lag=4, 24-min operational delay: only Y_hist[:-4] available)
                y_lag4 = ndi.gaussian_filter(np.tensordot(ema3_w, Y_hist[-7:-4], axes=(0, 0)), sigma=3.0)
                C5 = 0.45 * _minmax_pos(y_lag4) + 0.55 * _minmax_pos(A3)

                proxies = {
                    "G0_cr_t0_raw": G0,
                    "G_blur_cr_t0_s2": G_blur,
                    "A1_cr_ema6_s2": A1,
                    "A2_multich_t0_s2": A2,
                    "A3_multich_ema6_s2": A3,
                    "B1_cr_x_nwp_moisture": B1,
                    "B2_multich_ema_x_nwp": B2,
                    "C1_yhist_lag0_raw": C1,
                    "C2_yhist_lag0_ema3_s2": C2,
                    "C3_hybrid_lag0_yhist_radar": C3,
                    "C4_hybrid_lag2_12min_yhist_radar": C4,
                    "C5_hybrid_lag4_24min_yhist_radar": C5,
                }

                # Evaluate over the 10 future frames (t+6 .. t+60) under exact Top-K
                for name, p_field in proxies.items():
                    frame_csis = []
                    for lead in range(T_OUT):
                        t_mask = Y_fut[lead] >= LABEL_THRESHOLD_MM
                        k_px = int(t_mask.sum())
                        if k_px >= 5:
                            frame_csis.append(_exact_topk_csi(p_field, t_mask, k_px))
                    if frame_csis:
                        case_accum[name].append(float(np.mean(frame_csis)))

            if case_accum["G_blur_cr_t0_s2"]:
                per_case_scores.append({k: float(np.mean(case_accum[k])) for k in arm_names})

        n_used = len(per_case_scores)
        report["n_cases_used"] = n_used

        for k in arm_names:
            arr = np.array([c[k] for c in per_case_scores], dtype=np.float64)
            report["arms"][k] = {
                "mean_exact_topk_csi_fut10": float(arr.mean()),
                "ci95": _bootstrap_ci(arr),
            }

        # Paired differences against G_blur_cr_t0_s2 (and C3 vs C1 Oracle persistence)
        base_arr = np.array([c["G_blur_cr_t0_s2"] for c in per_case_scores], dtype=np.float64)
        c1_arr = np.array([c["C1_yhist_lag0_raw"] for c in per_case_scores], dtype=np.float64)

        for k in arm_names:
            if k == "G_blur_cr_t0_s2":
                continue
            diff = np.array([c[k] for c in per_case_scores], dtype=np.float64) - base_arr
            ci = _bootstrap_ci(diff)
            report["comparisons"][f"{k}_minus_G_blur"] = {
                "mean_diff": float(diff.mean()),
                "ci95": ci,
                "pos_cases": int(np.sum(diff > 0)),
                "neg_cases": int(np.sum(diff < 0)),
                "strictly_beats_G_blur": bool(ci[0] > 0.0),
            }

        diff_c3_c1 = np.array([c["C3_hybrid_lag0_yhist_radar"] for c in per_case_scores]) - c1_arr
        ci_c3_c1 = _bootstrap_ci(diff_c3_c1)
        report["comparisons"]["C3_hybrid_minus_C1_raw_yhist"] = {
            "mean_diff": float(diff_c3_c1.mean()),
            "ci95": ci_c3_c1,
            "strictly_beats_raw_yhist": bool(ci_c3_c1[0] > 0.0),
        }

        print(f"=== EXACT TOP-K FUTURE 10-FRAME PROBE (n_cases={n_used}) ===")
        for k, v in report["arms"].items():
            print(f"  {k:34s}: CSI = {v['mean_exact_topk_csi_fut10']:.6f}  CI {v['ci95']}")
        print("\n=== PAIRED DIFFS vs G_blur_cr_t0_s2 ===")
        for k, v in report["comparisons"].items():
            print(
                f"  {k:40s}: dCSI = {v['mean_diff']:+.6f}  "
                f"CI [{v['ci95'][0]:+.6f}, {v['ci95'][1]:+.6f}]"
            )

    except Exception:  # noqa: BLE001
        report["error"] = traceback.format_exc()
        print("ERROR:", report["error"])
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())