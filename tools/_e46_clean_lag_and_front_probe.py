#!/usr/bin/env python3
"""Diagnostic supplement to e46_abc_probe: isolate pure single-frame y_hist lags
and test advancing-front (positive temporal tendency) vs trailing EMA.

Read-only on TRAIN split (23 cases), exact Top-K (BIAS===1.0) over 10 future frames.
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

from radar_nwp_bench.data.index import split_cases  # noqa: E402
from tools._e46_ranking_and_yhist_probe import (  # noqa: E402
    BOOT_SEED,
    LABEL_THRESHOLD_MM,
    MAX_WINDOWS_PER_CASE,
    MIN_WET_PIXELS,
    T_IN,
    T_OUT,
    _bootstrap_ci,
    _exact_topk_csi,
    _minmax_pos,
    _zscore_2d,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default="/tmp/e46_supplement_probe.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    all_cases = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and (p / "radar.npy").is_file()
    )
    train_cases = list(split_cases(all_cases)["train"])

    arm_names = [
        "G_blur_cr_t0_s2",
        # 1. Pure Radar: Advancing front (positive tendency) vs trailing EMA
        "R1_cr_t0_s1p5",
        "R2_multich_t0_s2",
        "R3_cr_advancing_front",       # G_blur(t0) + alpha * max(0, G_blur(t0) - G_blur(t-2))
        "R4_multich_advancing_front",  # A2(t0) + alpha * max(0, A2(t0) - A2(t-2))
        # 2. Pure single-frame y_hist at operational lags (uncorrupted by radar or EMA)
        "Y0_yhist_lag0_raw",
        "Y0_yhist_lag0_s1p5",
        "Y2_yhist_lag2_12m_raw",
        "Y2_yhist_lag2_12m_s2",
        "Y4_yhist_lag4_24m_raw",
        "Y4_yhist_lag4_24m_s3",
        # 3. Support-Gated Multiplicative Fusion (Radar only allowed near y_hist envelope)
        "M0_lag0_support_gated_radar",
        "M2_lag2_support_gated_radar",
    ]
    per_case: list[dict[str, float]] = []

    for case_id in train_cases:
        case_dir = data_root / case_id
        meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
        radar = np.load(case_dir / "radar.npy", mmap_mode="r")
        label = np.load(case_dir / "label.npy", mmap_mode="r")

        rv = [str(v) for v in meta.get("radar_vars", [])]
        cr_idx = rv.index("CR") if "CR" in rv else 6
        vil_idx = rv.index("VIL") if "VIL" in rv else 9
        hbr_idx = rv.index("HBR") if "HBR" in rv else 8
        et_idx = rv.index("ET") if "ET" in rv else 7

        r_times = [str(t) for t in meta.get("radar_times", [])]
        l_times = [str(t) for t in meta.get("label_times", [])]
        l_map = {t: i for i, t in enumerate(l_times)}

        valid_t0 = [
            i for i in range(T_IN - 1, len(r_times) - T_OUT)
            if all(s in l_map for s in r_times[i - (T_IN - 1) : i + T_OUT + 1])
        ]
        if not valid_t0:
            continue
        if len(valid_t0) > MAX_WINDOWS_PER_CASE:
            pick = np.linspace(0, len(valid_t0) - 1, MAX_WINDOWS_PER_CASE).astype(int)
            valid_t0 = [valid_t0[int(i)] for i in pick]

        accum = {k: [] for k in arm_names}
        for idx_t0 in valid_t0:
            hist_r_idx = list(range(idx_t0 - (T_IN - 1), idx_t0 + 1))
            fut_r_idx = list(range(idx_t0 + 1, idx_t0 + T_OUT + 1))
            hist_l_idx = [l_map[r_times[i]] for i in hist_r_idx]
            fut_l_idx = [l_map[r_times[i]] for i in fut_r_idx]

            Y_fut = np.nan_to_num(np.asarray(label[fut_l_idx, 0], dtype=np.float64), nan=0.0)
            if int(np.sum(Y_fut >= LABEL_THRESHOLD_MM)) < MIN_WET_PIXELS:
                continue

            R_hist = np.nan_to_num(np.asarray(radar[hist_r_idx], dtype=np.float64), nan=0.0)
            Y_hist = np.nan_to_num(np.asarray(label[hist_l_idx, 0], dtype=np.float64), nan=0.0)

            cr_t0 = R_hist[-1, cr_idx]
            cr_tm2 = R_hist[-3, cr_idx]  # t - 12 min
            G_blur = ndi.gaussian_filter(cr_t0, sigma=2.0)
            G_blur_tm2 = ndi.gaussian_filter(cr_tm2, sigma=2.0)

            def _mc(step: int) -> np.ndarray:
                f = (
                    0.35 * _zscore_2d(R_hist[step, cr_idx])
                    + 0.30 * _zscore_2d(R_hist[step, vil_idx])
                    + 0.25 * _zscore_2d(R_hist[step, hbr_idx])
                    + 0.10 * _zscore_2d(R_hist[step, et_idx])
                )
                return ndi.gaussian_filter(f, sigma=2.0)

            mc_t0 = _mc(-1)
            mc_tm2 = _mc(-3)

            # Positive tendency (advancing convective front, zeroing out decaying wake)
            front_cr = G_blur + 0.5 * np.maximum(0.0, G_blur - G_blur_tm2)
            front_mc = mc_t0 + 0.5 * np.maximum(0.0, mc_t0 - mc_tm2)

            # Pure single-frame historical ground truth at lag=0, 2, 4
            y_lag0 = Y_hist[-1]
            y_lag2 = Y_hist[-3]   # t - 12 min
            y_lag4 = Y_hist[-5]   # t - 24 min

            y_lag0_s = ndi.gaussian_filter(y_lag0, sigma=1.5)
            y_lag2_s = ndi.gaussian_filter(y_lag2, sigma=2.0)
            y_lag4_s = ndi.gaussian_filter(y_lag4, sigma=3.0)

            # Support-gated multiplicative fusion:
            # Dilate/smooth y_hist to form a spatial support envelope, then modulate by radar front
            env_lag0 = ndi.gaussian_filter(ndi.maximum_filter(y_lag0, size=7), sigma=2.0)
            env_lag2 = ndi.gaussian_filter(ndi.maximum_filter(y_lag2, size=11), sigma=3.0)
            M0 = env_lag0 * (1.0 + 0.3 * _minmax_pos(front_mc))
            M2 = env_lag2 * (1.0 + 0.5 * _minmax_pos(front_mc))

            proxies = {
                "G_blur_cr_t0_s2": G_blur,
                "R1_cr_t0_s1p5": ndi.gaussian_filter(cr_t0, sigma=1.5),
                "R2_multich_t0_s2": mc_t0,
                "R3_cr_advancing_front": front_cr,
                "R4_multich_advancing_front": front_mc,
                "Y0_yhist_lag0_raw": y_lag0,
                "Y0_yhist_lag0_s1p5": y_lag0_s,
                "Y2_yhist_lag2_12m_raw": y_lag2,
                "Y2_yhist_lag2_12m_s2": y_lag2_s,
                "Y4_yhist_lag4_24m_raw": y_lag4,
                "Y4_yhist_lag4_24m_s3": y_lag4_s,
                "M0_lag0_support_gated_radar": M0,
                "M2_lag2_support_gated_radar": M2,
            }

            for name, p_field in proxies.items():
                csis = [
                    _exact_topk_csi(p_field, Y_fut[lead] >= LABEL_THRESHOLD_MM, int((Y_fut[lead] >= LABEL_THRESHOLD_MM).sum()))
                    for lead in range(T_OUT)
                    if int((Y_fut[lead] >= LABEL_THRESHOLD_MM).sum()) >= 5
                ]
                if csis:
                    accum[name].append(float(np.mean(csis)))

        if accum["G_blur_cr_t0_s2"]:
            per_case.append({k: float(np.mean(accum[k])) for k in arm_names})

    base = np.array([c["G_blur_cr_t0_s2"] for c in per_case])
    y0_arr = np.array([c["Y0_yhist_lag0_raw"] for c in per_case])
    y2_arr = np.array([c["Y2_yhist_lag2_12m_raw"] for c in per_case])

    out = {"n_cases": len(per_case), "arms": {}, "diffs_vs_G_blur": {}, "key_checks": {}}
    for k in arm_names:
        arr = np.array([c[k] for c in per_case])
        diff = arr - base
        ci = _bootstrap_ci(diff)
        out["arms"][k] = {"mean": float(arr.mean()), "ci95": _bootstrap_ci(arr)}
        out["diffs_vs_G_blur"][k] = {
            "mean_diff": float(diff.mean()),
            "ci95": ci,
            "pos_cases": int((diff > 0).sum()),
            "neg_cases": int((diff < 0).sum()),
            "strictly_beats_G_blur": bool(ci[0] > 0.0),
        }

    out["key_checks"]["M0_minus_Y0_raw"] = {
        "mean_diff": float((np.array([c["M0_lag0_support_gated_radar"] for c in per_case]) - y0_arr).mean()),
        "ci95": _bootstrap_ci(np.array([c["M0_lag0_support_gated_radar"] for c in per_case]) - y0_arr),
    }
    out["key_checks"]["M2_minus_Y2_raw"] = {
        "mean_diff": float((np.array([c["M2_lag2_support_gated_radar"] for c in per_case]) - y2_arr).mean()),
        "ci95": _bootstrap_ci(np.array([c["M2_lag2_support_gated_radar"] for c in per_case]) - y2_arr),
    }

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"=== SUPPLEMENT PROBE (n_cases={len(per_case)}) ===")
    for k in arm_names:
        d = out["diffs_vs_G_blur"][k]
        print(
            f"  {k:30s}: CSI={out['arms'][k]['mean']:.6f} | "
            f"d_blur={d['mean_diff']:+.6f} CI[{d['ci95'][0]:+.6f},{d['ci95'][1]:+.6f}] "
            f"({d['pos_cases']}+/{d['neg_cases']}-)"
        )
    print("KEY CHECKS:", json.dumps(out["key_checks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())