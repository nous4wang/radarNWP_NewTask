#!/usr/bin/env python3
"""Zero-training mechanism probes for E40 (P1) and E46 candidates (P_3D & P_SYM).

Strictly adheres to MASTER.md §11 protocol:
- Read-only on TRAIN split (23 cases / 548 frames from split_cases).
- Never opens val/test data files.
- Zero model training, zero checkpoint modification.
- 10,000x case-level bootstrap with seed=20261108.

Usage:
    python tools/_e46_pre_probes.py --out /tmp/e46_pre_probes.json
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

DATA_ROOT = Path("/home/wf/data/radarNWP/BJ_mmap")
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from radar_nwp_bench.data.index import split_cases  # noqa: E402
from radar_nwp_bench.models.common import BalancedExtremePrecipLoss  # noqa: E402

# Frozen constants (aligned with E39 §10 & E40 §3, 2026-11-08 20:28)
MAX_FRAMES = 24
LABEL_THRESHOLD_MM = 2.0
MIN_COVERAGE = 0.001
LOW_FREQ_CUTOFF = 0.125
HIGH_FREQ_CUTOFF = 0.50
H1B_CORR_UPPER = 0.30
N_BOOTSTRAP = 10000
BOOT_SEED = 20261108
MIN_CASES = 20

FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _radial_grid(shape: tuple[int, int]) -> np.ndarray:
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    r = np.sqrt(ky * ky + kx * kx)
    r_max = float(np.sqrt(0.5 ** 2 + 0.5 ** 2))
    return r / r_max


def _bootstrap_ci(values: np.ndarray, seed: int = BOOT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _pearson_2d(a: np.ndarray, b: np.ndarray) -> float | None:
    sa = float(a.std())
    sb = float(b.std())
    if sa <= 1e-12 or sb <= 1e-12:
        return None
    za = (a - a.mean()) / sa
    zb = (b - b.mean()) / sb
    return float(np.mean(za * zb))


def _mean_kernel_pixels(mask: np.ndarray) -> float:
    _, n = ndi.label(mask, structure=FOUR_CONNECTED)
    if n == 0:
        return 0.0
    return float(mask.sum()) / int(n)


def _equal_cov_csi_and_kernel(
    proxy: np.ndarray, label_mask: np.ndarray, cov: float
) -> tuple[float, float]:
    """Evaluate spatial support CSI & connected kernel at exact BIAS=1.0 (top-p)."""
    thr = float(np.quantile(proxy, 1.0 - cov))
    pred_mask = proxy >= thr
    if pred_mask.mean() <= 0.0:
        return 0.0, 0.0
    hits = float(np.logical_and(pred_mask, label_mask).sum())
    union = float(np.logical_or(pred_mask, label_mask).sum())
    csi = hits / union if union > 0.0 else 0.0
    k_px = _mean_kernel_pixels(pred_mask)
    return csi, k_px


def _estimate_integer_shift(f_prev: np.ndarray, f_curr: np.ndarray, max_shift: int = 6) -> tuple[int, int]:
    """Fast phase-correlation integer displacement (dy, dx) from f_prev to f_curr."""
    if f_prev.std() <= 1e-9 or f_curr.std() <= 1e-9:
        return 0, 0
    fa = np.fft.rfft2(f_prev - f_prev.mean())
    fb = np.fft.rfft2(f_curr - f_curr.mean())
    cross = fa.conj() * fb
    cc = np.fft.irfft2(cross / (np.abs(cross) + 1e-8), s=f_prev.shape)
    h, w = f_prev.shape
    iy, ix = np.unravel_index(int(np.argmax(cc)), (h, w))
    dy = iy if iy <= h // 2 else iy - h
    dx = ix if ix <= w // 2 else ix - w
    dy = int(np.clip(dy, -max_shift, max_shift))
    dx = int(np.clip(dx, -max_shift, max_shift))
    return dy, dx


# ---------------------------------------------------------------------------
# Sub-probe 1: E40 P1 (Cross-modal frequency band correlation & low-pass guard)
# ---------------------------------------------------------------------------

def _measure_e40_frame(
    L: np.ndarray, R: np.ndarray, grid: np.ndarray
) -> dict[str, float] | None:
    if L.std() <= 1e-12 or R.std() <= 1e-12:
        return None
    zL = (L - L.mean()) / L.std()
    zR = (R - R.mean()) / R.std()
    fL = np.fft.rfft2(zL)
    fR = np.fft.rfft2(zR)

    mask_low = (grid > 0.0) & (grid < LOW_FREQ_CUTOFF)
    mask_mid = (grid >= LOW_FREQ_CUTOFF) & (grid < HIGH_FREQ_CUTOFF)
    mask_high = grid >= HIGH_FREQ_CUTOFF

    def _band_field(f_spec: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.fft.irfft2(f_spec * m, s=L.shape)

    L_low, R_low = _band_field(fL, mask_low), _band_field(fR, mask_low)
    L_mid, R_mid = _band_field(fL, mask_mid), _band_field(fR, mask_mid)
    L_high, R_high = _band_field(fL, mask_high), _band_field(fR, mask_high)

    c_low = _pearson_2d(L_low, R_low)
    c_mid = _pearson_2d(L_mid, R_mid)
    c_high = _pearson_2d(L_high, R_high)
    # H2 guard: pure low-pass filtered radar against full label L
    c_low_lp = _pearson_2d(L, R_low)

    pL = np.abs(fL) ** 2
    pR = np.abs(fR) ** 2
    pL[0, 0] = 0.0
    pR[0, 0] = 0.0
    lf_share_L = float(pL[mask_low].sum() / (pL[grid > 0].sum() + 1e-12))
    lf_share_R = float(pR[mask_low].sum() / (pR[grid > 0].sum() + 1e-12))

    if c_low is None or c_mid is None or c_high is None or c_low_lp is None:
        return None
    return {
        "corr_low": c_low,
        "corr_mid": c_mid,
        "corr_high": c_high,
        "d_band": c_low - c_high,
        "corr_low_lp": c_low_lp,
        "lf_share_L": lf_share_L,
        "lf_share_R": lf_share_R,
    }


# ---------------------------------------------------------------------------
# Sub-probe 2: Candidate 1 P_3D (3D CAP column + shear/swept footprint)
# ---------------------------------------------------------------------------

def _measure_p3d_frame(
    L: np.ndarray,
    rad_curr: np.ndarray,
    rad_prev: np.ndarray | None,
    cap_indices: list[int],
    cr_idx: int,
) -> dict[str, float] | None:
    cov = float(np.mean(L >= LABEL_THRESHOLD_MM))
    if cov < MIN_COVERAGE:
        return None
    label_mask = L >= LABEL_THRESHOLD_MM

    # G0: single-level instantaneous CR
    G0 = np.asarray(rad_curr[cr_idx], dtype=np.float64)

    # G1: static 3D vertical column envelope across CAP20..CAP70
    caps_curr = np.asarray(rad_curr[cap_indices], dtype=np.float64)  # (Z, H, W)
    G1 = 0.5 * caps_curr.max(axis=0) + 0.5 * caps_curr.mean(axis=0)

    # G2 & G3: 6-min swept footprint and vertical wind-shear tilted column
    if rad_prev is not None:
        caps_prev = np.asarray(rad_prev[cap_indices], dtype=np.float64)
        low_prev = caps_prev[:2].mean(axis=0)
        low_curr = caps_curr[:2].mean(axis=0)
        high_prev = caps_prev[-2:].mean(axis=0)
        high_curr = caps_curr[-2:].mean(axis=0)

        v_low = _estimate_integer_shift(low_prev, low_curr)
        v_high = _estimate_integer_shift(high_prev, high_curr)
        dv_shear = (v_high[0] - v_low[0], v_high[1] - v_low[1])

        # G2: 6-min swept integration along low-level phase velocity
        swept = [
            np.roll(low_curr, shift=(-int(round(a * v_low[0])), -int(round(a * v_low[1]))), axis=(0, 1))
            for a in (0.0, 0.5, 1.0)
        ]
        G2 = np.mean(swept, axis=0)

        # G3: tilt-align upper CAP layers along shear vector + 6-min swept integration
        n_z = len(cap_indices)
        aligned_layers = []
        for z_i in range(n_z):
            frac = z_i / max(n_z - 1, 1)
            sy = int(round(frac * dv_shear[0]))
            sx = int(round(frac * dv_shear[1]))
            aligned_layers.append(np.roll(caps_curr[z_i], shift=(sy, sx), axis=(0, 1)))
        col_tilted = 0.5 * np.max(aligned_layers, axis=0) + 0.5 * np.mean(aligned_layers, axis=0)
        G3 = 0.5 * col_tilted + 0.5 * G2
    else:
        G2 = G1
        G3 = G1

    # Pure isotropic Gaussian blur control (to verify G3 vs naive smoothing)
    G_blur = ndi.gaussian_filter(G0, sigma=2.0)

    csi_g0, k_g0 = _equal_cov_csi_and_kernel(G0, label_mask, cov)
    csi_g1, k_g1 = _equal_cov_csi_and_kernel(G1, label_mask, cov)
    csi_g2, k_g2 = _equal_cov_csi_and_kernel(G2, label_mask, cov)
    csi_g3, k_g3 = _equal_cov_csi_and_kernel(G3, label_mask, cov)
    csi_blur, k_blur = _equal_cov_csi_and_kernel(G_blur, label_mask, cov)
    k_label = _mean_kernel_pixels(label_mask)

    return {
        "csi_g0": csi_g0,
        "csi_g1": csi_g1,
        "csi_g2": csi_g2,
        "csi_g3": csi_g3,
        "csi_blur": csi_blur,
        "d_csi_g3_g0": csi_g3 - csi_g0,
        "d_csi_g3_blur": csi_g3 - csi_blur,
        "k_g0": k_g0,
        "k_g3": k_g3,
        "k_label": k_label,
    }


# ---------------------------------------------------------------------------
# Sub-probe 3: Candidate 2 P_SYM (Asymmetric vs Symmetric Regression Weight)
# ---------------------------------------------------------------------------

def _symmetric_balanced_loss(
    pred_rain: torch.Tensor,
    target_rain: torch.Tensor,
    thresholds: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 4.0),
    alpha_csi: float = 0.5,
    neighborhood_pool: int = 3,
) -> torch.Tensor:
    """Candidate 2 symmetric-weight + shift-tolerant regression loss."""
    pred = pred_rain.clamp_min(0.0)
    target = target_rain.clamp_min(0.0)

    pred_log = torch.log1p(pred)
    target_log = torch.log1p(target)

    # Key fix 1: Symmetric weight w(max(target, stop_grad(pred))) eliminates free spikes
    joint_mag = torch.maximum(target, pred.detach())
    weight_sym = 1.0 + 2.0 * torch.tanh(joint_mag / 2.0)

    # Key fix 2: 3x3 local-max tolerance on log regression prevents 2px double-penalty shrinkage
    if neighborhood_pool > 1:
        pad = neighborhood_pool // 2
        p_pool = F.max_pool2d(pred_log, kernel_size=neighborhood_pool, stride=1, padding=pad)
        t_pool = F.max_pool2d(target_log, kernel_size=neighborhood_pool, stride=1, padding=pad)
        reg_err = 0.5 * F.smooth_l1_loss(pred_log, target_log, reduction="none") + \
                  0.5 * F.smooth_l1_loss(p_pool, t_pool, reduction="none")
    else:
        reg_err = F.smooth_l1_loss(pred_log, target_log, reduction="none")

    reg_loss = torch.mean(weight_sym * reg_err)

    soft_csi = pred.new_zeros(())
    for i, th in enumerate(thresholds):
        p_mask = torch.sigmoid((pred - th) * 5.0)
        t_mask = (target >= th).to(pred.dtype)
        hits = torch.sum(p_mask * t_mask)
        misses = torch.sum((1.0 - p_mask) * t_mask)
        fa = torch.sum(p_mask * (1.0 - t_mask))
        csi = (hits + 1e-5) / (hits + misses + fa + 1e-5)
        soft_csi = soft_csi + (1.0 + 0.5 * i) * (1.0 - csi)

    return reg_loss + alpha_csi * soft_csi / len(thresholds)


def _measure_psym_frame(L: np.ndarray, orig_loss_fn: BalancedExtremePrecipLoss) -> dict[str, float] | None:
    if float(np.mean(L >= LABEL_THRESHOLD_MM)) < MIN_COVERAGE:
        return None
    t_tensor = torch.from_numpy(L).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    # 2-pixel shifted field
    shifted_2px = torch.roll(t_tensor, shifts=(2, 2), dims=(-2, -1))

    alphas = np.linspace(0.10, 1.40, 27)
    losses_orig = []
    losses_sym = []
    with torch.no_grad():
        for a in alphas:
            cand = float(a) * shifted_2px
            losses_orig.append(float(orig_loss_fn(cand, t_tensor).item()))
            losses_sym.append(float(_symmetric_balanced_loss(cand, t_tensor).item()))

    alpha_star_orig = float(alphas[int(np.argmin(losses_orig))])
    alpha_star_sym = float(alphas[int(np.argmin(losses_sym))])

    # Test 2: Free needle spike penalty (inject 200mm spike on a dry pixel)
    dry_coords = np.argwhere(L == 0.0)
    if len(dry_coords) == 0:
        return None
    y0, x0 = int(dry_coords[0][0]), int(dry_coords[0][1])

    base_pred = shifted_2px.clone()
    spike_pred = shifted_2px.clone()
    spike_pred[0, 0, y0, x0] = 200.0

    with torch.no_grad():
        dl_orig = float(orig_loss_fn(spike_pred, t_tensor).item() - orig_loss_fn(base_pred, t_tensor).item())
        dl_sym = float(_symmetric_balanced_loss(spike_pred, t_tensor).item() -
                       _symmetric_balanced_loss(base_pred, t_tensor).item())

    # Gradient top-1% share on the spiked field
    p_o = spike_pred.clone().requires_grad_(True)
    orig_loss_fn(p_o, t_tensor).backward()
    g_o = p_o.grad.detach().abs().reshape(-1).numpy()
    k1 = max(1, int(0.01 * g_o.size))
    top1_orig = float(np.sort(g_o)[-k1:].sum() / (g_o.sum() + 1e-12))

    p_s = spike_pred.clone().requires_grad_(True)
    _symmetric_balanced_loss(p_s, t_tensor).backward()
    g_s = p_s.grad.detach().abs().reshape(-1).numpy()
    top1_sym = float(np.sort(g_s)[-k1:].sum() / (g_s.sum() + 1e-12))

    return {
        "alpha_star_orig_2px": alpha_star_orig,
        "alpha_star_sym_2px": alpha_star_sym,
        "spike_penalty_ratio_sym_over_orig": float(dl_sym / max(dl_orig, 1e-12)),
        "grad_top1pct_orig": top1_orig,
        "grad_top1pct_sym": top1_sym,
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default="/tmp/e46_pre_probes.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_path = Path(args.out)

    report: dict = {
        "tool": "tools/_e46_pre_probes.py",
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(data_root),
        "probes": {},
    }

    try:
        all_cases = sorted(
            p.name for p in data_root.iterdir()
            if p.is_dir() and (p / "radar.npy").is_file()
        )
        parts = split_cases(all_cases)
        train_cases = list(parts["train"])
        report["split_audit"] = {
            "n_train_cases": len(train_cases),
            "files_opened": "train split only",
        }

        orig_loss_fn = BalancedExtremePrecipLoss(alpha_csi=0.5)

        e40_cases, p3d_cases, psym_cases = [], [], []

        for case_id in train_cases:
            case_dir = data_root / case_id
            meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
            radar = np.load(case_dir / "radar.npy", mmap_mode="r")
            label = np.load(case_dir / "label.npy", mmap_mode="r")

            rv = [str(v) for v in meta.get("radar_vars", [])]
            cr_idx = rv.index("CR") if "CR" in rv else 6
            cap_indices = [i for i, v in enumerate(rv) if v.startswith("CAP")]
            if not cap_indices:
                cap_indices = list(range(6))

            r_map = {str(t): i for i, t in enumerate(meta.get("radar_times", []))}
            l_map = {str(t): i for i, t in enumerate(meta.get("label_times", []))}
            common = sorted(set(r_map) & set(l_map), key=lambda s: (len(s), s))
            if not common:
                continue
            if len(common) > MAX_FRAMES:
                pick = np.linspace(0, len(common) - 1, MAX_FRAMES).astype(int)
                common = [common[int(i)] for i in pick]

            shape = tuple(label.shape[-2:])
            grid = _radial_grid(shape)

            c_e40, c_p3d, c_psym = [], [], []
            for stamp in common:
                li = l_map[stamp]
                ri = r_map[stamp]
                L = np.nan_to_num(np.asarray(label[li, 0], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                R_all = np.nan_to_num(np.asarray(radar[ri], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                R_prev = (
                    np.nan_to_num(np.asarray(radar[ri - 1], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                    if ri > 0 else None
                )

                m1 = _measure_e40_frame(L, R_all[cr_idx], grid)
                if m1 is not None:
                    c_e40.append(m1)

                m2 = _measure_p3d_frame(L, R_all, R_prev, cap_indices, cr_idx)
                if m2 is not None:
                    c_p3d.append(m2)

                m3 = _measure_psym_frame(L, orig_loss_fn)
                if m3 is not None:
                    c_psym.append(m3)

            if c_e40:
                e40_cases.append({k: float(np.mean([x[k] for x in c_e40])) for k in c_e40[0]})
            if c_p3d:
                p3d_cases.append({k: float(np.mean([x[k] for x in c_p3d])) for k in c_p3d[0]})
            if c_psym:
                psym_cases.append({k: float(np.mean([x[k] for x in c_psym])) for k in c_psym[0]})

        # --- Aggregate E40 P1 ---
        dband = np.array([c["d_band"] for c in e40_cases])
        clow = np.array([c["corr_low"] for c in e40_cases])
        chigh = np.array([c["corr_high"] for c in e40_cases])
        clp = np.array([c["corr_low_lp"] for c in e40_cases])
        ci_dband = _bootstrap_ci(dband)
        h1a = bool(ci_dband[1] < 0.0)
        h1b = bool(clow.mean() < H1B_CORR_UPPER)
        h2 = bool(clp.mean() < H1B_CORR_UPPER)
        report["probes"]["E40_P1"] = {
            "n_cases": len(e40_cases),
            "mean_corr_low": float(clow.mean()),
            "mean_corr_high": float(chigh.mean()),
            "mean_d_band": float(dband.mean()),
            "ci95_d_band": ci_dband,
            "mean_corr_low_lp": float(clp.mean()),
            "H1a_ci_strictly_negative": h1a,
            "H1b_corr_low_lt_0p30": h1b,
            "H2_lowpass_guard_lt_0p30": h2,
            "verdict": "PASS" if (len(e40_cases) >= MIN_CASES and h1a and h1b and h2) else "FAIL",
        }

        # --- Aggregate Candidate 1: P_3D ---
        d_g3_g0 = np.array([c["d_csi_g3_g0"] for c in p3d_cases])
        d_g3_blur = np.array([c["d_csi_g3_blur"] for c in p3d_cases])
        ci_g3_g0 = _bootstrap_ci(d_g3_g0)
        ci_g3_blur = _bootstrap_ci(d_g3_blur)
        k_g0_m = float(np.mean([c["k_g0"] for c in p3d_cases]))
        k_g3_m = float(np.mean([c["k_g3"] for c in p3d_cases]))
        p3d_pass = bool(ci_g3_g0[0] > 0.0 and ci_g3_blur[0] > 0.0 and k_g3_m > k_g0_m)
        report["probes"]["P_3D_Column_Footprint"] = {
            "n_cases": len(p3d_cases),
            "mean_csi_top_p_G0_cr": float(np.mean([c["csi_g0"] for c in p3d_cases])),
            "mean_csi_top_p_G1_col": float(np.mean([c["csi_g1"] for c in p3d_cases])),
            "mean_csi_top_p_G2_swept": float(np.mean([c["csi_g2"] for c in p3d_cases])),
            "mean_csi_top_p_G3_shear_swept": float(np.mean([c["csi_g3"] for c in p3d_cases])),
            "mean_csi_top_p_G_blur": float(np.mean([c["csi_blur"] for c in p3d_cases])),
            "mean_d_csi_G3_minus_G0": float(d_g3_g0.mean()),
            "ci95_d_csi_G3_minus_G0": ci_g3_g0,
            "ci95_d_csi_G3_minus_blur": ci_g3_blur,
            "mean_kernel_px_G0": k_g0_m,
            "mean_kernel_px_G3": k_g3_m,
            "mean_kernel_px_LABEL": float(np.mean([c["k_label"] for c in p3d_cases])),
            "verdict": "PASS" if p3d_pass else "FAIL",
        }

        # --- Aggregate Candidate 2: P_SYM ---
        a_orig = float(np.mean([c["alpha_star_orig_2px"] for c in psym_cases]))
        a_sym = float(np.mean([c["alpha_star_sym_2px"] for c in psym_cases]))
        pen_ratio = float(np.mean([c["spike_penalty_ratio_sym_over_orig"] for c in psym_cases]))
        top1_o = float(np.mean([c["grad_top1pct_orig"] for c in psym_cases]))
        top1_s = float(np.mean([c["grad_top1pct_sym"] for c in psym_cases]))
        psym_pass = bool(a_sym > a_orig + 0.15 and pen_ratio >= 5.0)
        report["probes"]["P_SYM_Symmetric_Loss"] = {
            "n_cases": len(psym_cases),
            "alpha_star_orig_at_2px_shift": a_orig,
            "alpha_star_sym_at_2px_shift": a_sym,
            "spike_200mm_penalty_ratio_sym_over_orig": pen_ratio,
            "grad_top1pct_share_orig": top1_o,
            "grad_top1pct_share_sym": top1_s,
            "verdict": "PASS" if psym_pass else "FAIL",
        }

        print("=== E40_P1 ===", json.dumps(report["probes"]["E40_P1"], indent=2))
        print("=== P_3D ===", json.dumps(report["probes"]["P_3D_Column_Footprint"], indent=2))
        print("=== P_SYM ===", json.dumps(report["probes"]["P_SYM_Symmetric_Loss"], indent=2))

    except Exception:  # noqa: BLE001
        report["error"] = traceback.format_exc()
        print("ERROR:", report["error"])
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())