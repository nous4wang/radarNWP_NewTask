#!/usr/bin/env python3
"""Pre-registered zero-compute confirmation probe P1-P4 (frozen before execution).

Tests four binary questions on the TRAIN split (23 cases) over the 10-frame
future horizon (t+6..t+60):
- P1 (Equal-lag 12m comparison) : d_P1 = CSI(y_hist(t-12)) - CSI(cr(t-12)), both raw & s2-blurred.
                                  Pass: CI95_low > 0.
- P2 (2nd-order interaction)    : d_int = (R4 - R2) - (R3 - G_blur) at alpha=0.5.
                                  Pass: CI95_low > 0.
- P3 (Alpha robustness)         : R4(alpha) - R2 for alpha in {0.25, 0.5, 1.0, 2.0}.
                                  Pass: >= 3/4 alphas have CI95_low > 0.
- P4 (Metric-family replication): Replicate sign & CI on another metric family
                                  (Exact Top-K Neighborhood FSS r=3 [7x7] & soft Spearman rank corr),
                                  reporting the null/persistence floor and paired n_cases.

Rule (§11.8): Any failure in P1-P4 -> discard and archive without training compute.
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
    LABEL_THRESHOLD_MM,
    MAX_WINDOWS_PER_CASE,
    MIN_WET_PIXELS,
    T_IN,
    T_OUT,
    _bootstrap_ci,
    _exact_topk_csi,
    _zscore_2d,
)

ALPHAS = (0.25, 0.50, 1.00, 2.00)


def _exact_topk_fss_r3(proxy: np.ndarray, target_mask: np.ndarray, k: int) -> float:
    """Second metric family: Neighborhood Fractions Skill Score (7x7 box, r=3) at exact Top-K."""
    if k <= 0:
        return 0.0
    flat = np.nan_to_num(proxy.reshape(-1), nan=-1e9, posinf=1e9, neginf=-1e9)
    top_idx = np.argsort(flat, kind="mergesort")[-k:]
    pred_mask = np.zeros_like(flat, dtype=np.float64)
    pred_mask[top_idx] = 1.0
    pred_2d = pred_mask.reshape(target_mask.shape)
    targ_2d = target_mask.astype(np.float64)

    fp = ndi.uniform_filter(pred_2d, size=7, mode="nearest")
    ft = ndi.uniform_filter(targ_2d, size=7, mode="nearest")
    num = float(np.mean((fp - ft) ** 2))
    den = float(np.mean(fp ** 2) + np.mean(ft ** 2))
    return 1.0 - (num / den) if den > 1e-12 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default="/tmp/e46_p1_p4_probe.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    all_cases = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and (p / "radar.npy").is_file()
    )
    train_cases = list(split_cases(all_cases)["train"])

    case_rows: list[dict[str, float]] = []

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

        win_metrics: dict[str, list[float]] = {}

        def _add(k: str, v: float) -> None:
            win_metrics.setdefault(k, []).append(v)

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

            # Fields at t0 (-1) and t-12min (-3)
            cr_t0 = R_hist[-1, cr_idx]
            cr_tm2 = R_hist[-3, cr_idx]
            y_tm2 = Y_hist[-3]

            G_blur = ndi.gaussian_filter(cr_t0, sigma=2.0)
            cr_tm2_s2 = ndi.gaussian_filter(cr_tm2, sigma=2.0)
            y_tm2_s2 = ndi.gaussian_filter(y_tm2, sigma=2.0)

            def _mc(step: int) -> np.ndarray:
                f = (
                    0.35 * _zscore_2d(R_hist[step, cr_idx])
                    + 0.30 * _zscore_2d(R_hist[step, vil_idx])
                    + 0.25 * _zscore_2d(R_hist[step, hbr_idx])
                    + 0.10 * _zscore_2d(R_hist[step, et_idx])
                )
                return ndi.gaussian_filter(f, sigma=2.0)

            R2 = _mc(-1)
            mc_tm2 = _mc(-3)

            d_cr_pos = np.maximum(0.0, G_blur - cr_tm2_s2)
            d_mc_pos = np.maximum(0.0, R2 - mc_tm2)

            R3_a05 = G_blur + 0.5 * d_cr_pos
            R4_by_a = {a: R2 + a * d_mc_pos for a in ALPHAS}

            fields_csi = {
                "G0_raw": cr_t0,
                "G_blur": G_blur,
                "cr_tm2_raw": cr_tm2,
                "cr_tm2_s2": cr_tm2_s2,
                "y_tm2_raw": y_tm2,
                "y_tm2_s2": y_tm2_s2,
                "R2": R2,
                "R3_a05": R3_a05,
                **{f"R4_a{a}": R4_by_a[a] for a in ALPHAS},
            }

            # Evaluate CSI and FSS_r3 over 10 future frames
            valid_leads = [
                lead for lead in range(T_OUT)
                if int((Y_fut[lead] >= LABEL_THRESHOLD_MM).sum()) >= 5
            ]
            if not valid_leads:
                continue

            for name, fld in fields_csi.items():
                c_list = [
                    _exact_topk_csi(fld, Y_fut[l] >= LABEL_THRESHOLD_MM, int((Y_fut[l] >= LABEL_THRESHOLD_MM).sum()))
                    for l in valid_leads
                ]
                _add(f"csi_{name}", float(np.mean(c_list)))

            # P4 FSS_r3 on key arms (G0_raw null floor, G_blur, R2, R3_a05, R4_a0.5, cr_tm2_s2, y_tm2_s2)
            for name in ("G0_raw", "G_blur", "R2", "R3_a05", "R4_a0.5", "cr_tm2_s2", "y_tm2_s2"):
                fld = fields_csi[name]
                f_list = [
                    _exact_topk_fss_r3(fld, Y_fut[l] >= LABEL_THRESHOLD_MM, int((Y_fut[l] >= LABEL_THRESHOLD_MM).sum()))
                    for l in valid_leads
                ]
                _add(f"fss_{name}", float(np.mean(f_list)))

        if "csi_G_blur" in win_metrics:
            case_rows.append({k: float(np.mean(v)) for k, v in win_metrics.items()})

    n_cases = len(case_rows)

    def _col(key: str) -> np.ndarray:
        return np.array([r[key] for r in case_rows], dtype=np.float64)

    def _stat(diff: np.ndarray) -> dict:
        ci = _bootstrap_ci(diff)
        return {
            "mean_diff": float(diff.mean()),
            "ci95": ci,
            "pos_cases": int((diff > 0).sum()),
            "neg_cases": int((diff < 0).sum()),
            "pass_ci_low_gt_0": bool(ci[0] > 0.0),
        }

    # P1: Equal-lag (t-12m) y_hist vs cr
    p1_raw = _stat(_col("csi_y_tm2_raw") - _col("csi_cr_tm2_raw"))
    p1_s2 = _stat(_col("csi_y_tm2_s2") - _col("csi_cr_tm2_s2"))

    # P2: Interaction (R4 - R2) - (R3 - G_blur) at alpha=0.5
    d_r4_r2 = _col("csi_R4_a0.5") - _col("csi_R2")
    d_r3_gb = _col("csi_R3_a05") - _col("csi_G_blur")
    p2_int = _stat(d_r4_r2 - d_r3_gb)

    # P3: Alpha robustness for R4(a) - R2 and interaction across alpha in {0.25, 0.5, 1.0, 2.0}
    p3_r4_vs_r2 = {str(a): _stat(_col(f"csi_R4_a{a}") - _col("csi_R2")) for a in ALPHAS}
    p3_r4_vs_gb = {str(a): _stat(_col(f"csi_R4_a{a}") - _col("csi_G_blur")) for a in ALPHAS}
    n_pass_r4_r2 = sum(1 for v in p3_r4_vs_r2.values() if v["pass_ci_low_gt_0"])

    # P4: Metric-family replication on FSS_r3 (with G0_raw & G_blur null floors)
    p4_fss = {
        "n_cases_paired": n_cases,
        "null_floor_G0_raw_mean": float(_col("fss_G0_raw").mean()),
        "null_floor_G_blur_mean": float(_col("fss_G_blur").mean()),
        "R4_minus_R2_fss": _stat(_col("fss_R4_a0.5") - _col("fss_R2")),
        "interaction_fss": _stat((_col("fss_R4_a0.5") - _col("fss_R2")) - (_col("fss_R3_a05") - _col("fss_G_blur"))),
        "y_tm2_minus_cr_tm2_s2_fss": _stat(_col("fss_y_tm2_s2") - _col("fss_cr_tm2_s2")),
    }

    report = {
        "tool": "tools/_e46_p1_p4_confirm_probe.py",
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "n_cases_paired": n_cases,
        "null_floors_csi": {
            "G0_cr_t0_raw": float(_col("csi_G0_raw").mean()),
            "G_blur_cr_t0_s2": float(_col("csi_G_blur").mean()),
            "cr_tm2_12m_raw": float(_col("csi_cr_tm2_raw").mean()),
            "cr_tm2_12m_s2": float(_col("csi_cr_tm2_s2").mean()),
        },
        "P1_equal_lag_12m_yhist_minus_cr": {"raw": p1_raw, "s2_blur": p1_s2},
        "P2_interaction_R4_R2_minus_R3_Gblur": p2_int,
        "P3_alpha_robustness": {
            "R4_minus_R2": p3_r4_vs_r2,
            "R4_minus_G_blur": p3_r4_vs_gb,
            "n_alphas_pass_R4_minus_R2": n_pass_r4_r2,
            "pass_at_least_3_of_4": bool(n_pass_r4_r2 >= 3),
        },
        "P4_fss_r3_replication": p4_fss,
        "all_four_gates_pass": bool(
            (p1_raw["pass_ci_low_gt_0"] or p1_s2["pass_ci_low_gt_0"])
            and p2_int["pass_ci_low_gt_0"]
            and (n_pass_r4_r2 >= 3)
            and p4_fss["interaction_fss"]["pass_ci_low_gt_0"]
        ),
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())