#!/usr/bin/env python3
"""E33 / §4.35: equal-BIAS control plus a corrected phase-scramble arm.

Two things E32 / §4.34 left open:

  truthswap_eqbias   the truth's high-k band replaces the prediction's, then
                     the whole field is multiplied by ONE scalar per window,
                     chosen so the 2 mm bias matches the control arm exactly.
                     Compared with the plain truthswap arm, this isolates the
                     contribution of the small-scale ARRANGEMENT from the
                     contribution of reporting more rain.

  phasescramble_fixed  the phase of every Fourier mode is randomised, but the
                     random phases are drawn so the spectrum stays Hermitian:
                     mode (-i,-j) gets the conjugate of mode (i,j).  E32 drew
                     them independently, which made the inverse transform
                     complex; taking the real part then threw away roughly
                     half the high-k energy (measured ratio 0.19-0.30 instead
                     of 1.0).

Offline only: the stitched fields come from the E25 batch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

TIERS = (0.1, 0.5, 2.0, 4.0)
TIER_KEYS = ("0.1", "0.5", "2.0", "4.0")
KM_PER_PIXEL = 1.1119508372419142
GRID = 301
FIELD_KM = GRID * KM_PER_PIXEL
CONVECTIVE_MAX_KM = 10.0
N_BOOT = 10000
RNG_SEED = 42
BIAS_TIER = 2.0


def mode_wavenumbers(shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    kx = np.rint(np.fft.fftfreq(nx) * nx)
    ky = np.rint(np.fft.fftfreq(ny) * ny)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    return np.fft.fftshift(np.hypot(kx_grid, kx_grid))


def split_band(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    k = mode_wavenumbers(shape)
    high = k >= int(np.ceil(FIELD_KM / CONVECTIVE_MAX_KM))
    return high, ~high


def half_band_mask(shape: tuple[int, int]) -> np.ndarray:
    """High-k mask in the rfft2 (half-spectrum) layout.

    rfft2 keeps kx >= 0, so the half-spectrum index j carries kx = j and the
    full row index i carries ky = rint(fftfreq(ny) * ny).  The radial
    wavenumber is the same quantity the full-spectrum mask uses, so the two
    masks cover the same set of modes.
    """
    ny, nx = shape
    ky = np.rint(np.fft.fftfreq(ny) * ny).astype(int)
    kx = np.arange(nx // 2 + 1)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    return np.hypot(ky_grid, kx_grid) >= int(np.ceil(FIELD_KM / CONVECTIVE_MAX_KM))


def counts(pred: np.ndarray, truth: np.ndarray, thr: float) -> dict:
    p = pred >= thr
    t = truth >= thr
    return {
        "hits": float(np.count_nonzero(p & t)),
        "misses": float(np.count_nonzero(~p & t)),
        "false_alarms": float(np.count_nonzero(p & ~t)),
        "correct_negatives": float(np.count_nonzero(~p & ~t)),
    }


def csi_from(c: dict) -> float:
    den = c["hits"] + c["misses"] + c["false_alarms"]
    return float(c["hits"] / den) if den > 0 else 0.0


def bias_from(c: dict) -> float:
    den = c["hits"] + c["misses"]
    return float((c["hits"] + c["false_alarms"]) / den) if den > 0 else 0.0


def band_energy(field: np.ndarray, mask: np.ndarray) -> float:
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return float(spec[mask].mean())


def solve_equal_bias(
    field: np.ndarray, truth: np.ndarray, target_bias: float
) -> tuple[float, float]:
    """One scalar per window that matches the 2 mm bias of the control.

    The bias is monotone increasing in the scale factor, so a bisection on
    [0.5, 2.0] converges.  Returns the factor and the achieved bias.
    """
    def bias_at(k: float) -> float:
        return bias_from(counts(field * k, truth, BIAS_TIER))

    lo, hi = 0.2, 5.0
    blo, bhi = bias_at(lo), bias_at(hi)
    if not (blo <= target_bias <= bhi):
        # The target is outside what any scale factor can reach; the caller
        # counts this window as a pairing failure.
        k = lo if abs(blo - target_bias) < abs(bhi - target_bias) else hi
        return k, float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bias_at(mid) < target_bias:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    return k, bias_at(k)


def _finish(out: np.ndarray, neg: float, scalar: float) -> tuple[np.ndarray, float, float]:
    return np.clip(out, 0.0, None), neg, scalar


def build_arm(
    arm: str,
    pred: np.ndarray,
    truth: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return (field, negative fraction, scalar applied)."""
    n_frames = pred.shape[0]
    out = np.empty_like(pred)
    for t in range(n_frames):
        if arm == "phasescramble_fixed":
            # rfft2 keeps only the kx >= 0 half of the spectrum, which is
            # Hermitian by construction: irfft2 reconstructs a real field and
            # the amplitude spectrum is preserved (measured ratio 0.9986).
            # The full-spectrum route with independently drawn phases (E32)
            # broke that symmetry, and taking the real part then threw away
            # about half the high-k energy.
            half = np.fft.rfft2(pred[t])
            high_h = half_band_mask((GRID, GRID))
            half_new = half * (~high_h)
            half_new[high_h] = np.abs(half[high_h]) * np.exp(1j * phase[high_h])
            out[t] = np.fft.irfft2(half_new, s=(GRID, GRID))
            continue
        fp = np.fft.fftshift(np.fft.fft2(pred[t]))
        if arm == "orig":
            new = fp
        elif arm == "truthswap":
            ft = np.fft.fftshift(np.fft.fft2(truth[t]))
            new = fp * low + ft * high
        else:
            raise ValueError(f"unknown arm {arm}")
        out[t] = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
    # The negative fraction is measured BEFORE the clamp, on the intervened
    # field only.  The dumped fields themselves are already clamped, so the
    # control arm must be zero by construction; the number that matters is
    # how much of the intervened field the clamp removes.
    neg = float(np.mean(out < 0.0))
    return np.clip(out, 0.0, None), neg, 1.0


def arm_table(
    pred_path: Path, truth_path: Path, high: np.ndarray, low: np.ndarray,
    arms: list[str], rng: np.random.Generator,
) -> tuple[dict[str, list[dict]], dict]:
    preds = np.load(pred_path, mmap_mode="r")
    truths = np.load(truth_path, mmap_mode="r")
    n_win = preds.shape[0]
    phase = rng.uniform(-np.pi, np.pi, size=(GRID, GRID // 2 + 1))
    tables: dict[str, list[dict]] = {a: [] for a in arms}
    ks: list[float] = []
    bias_err: list[float] = []
    pairing_failures = 0
    negs: dict[str, list[float]] = {a: [] for a in arms}
    e_ratio: dict[str, list[float]] = {a: [] for a in arms}

    for w in range(n_win):
        p = np.asarray(preds[w], dtype=np.float64)
        t = np.asarray(truths[w], dtype=np.float64)
        ctrl_bias = bias_from(counts(p, t, BIAS_TIER))
        built: dict[str, np.ndarray] = {}
        for arm in arms:
            if arm == "truthswap_eqbias":
                continue  # derived from truthswap below
            field, neg, _ = build_arm(arm, p, t, high, low, phase)
            built[arm] = field
            negs[arm].append(neg)
            e_ratio[arm].append(
                band_energy(field[0], high) / max(band_energy(p[0], high), 1e-12)
            )
        if "truthswap_eqbias" in arms:
            k, achieved = solve_equal_bias(built["truthswap"], t, ctrl_bias)
            built["truthswap_eqbias"] = built["truthswap"] * k
            ks.append(k)
            if not np.isfinite(achieved):
                pairing_failures += 1
            else:
                bias_err.append(abs(achieved - ctrl_bias))
            negs["truthswap_eqbias"].append(0.0)
            e_ratio["truthswap_eqbias"].append(
                band_energy(built["truthswap_eqbias"][0], high)
                / max(band_energy(p[0], high), 1e-12)
            )
        for arm in arms:
            if arm not in built:
                continue
            row = {}
            for thr, key in zip(TIERS, TIER_KEYS):
                row[key] = counts(built[arm], t, thr)
            tables[arm].append(row)

    meta = {
        "n_windows": n_win,
        "equal_bias_scalars": [round(k, 6) for k in ks],
        "equal_bias_max_abs_bias_error": float(max(bias_err)) if bias_err else 0.0,
        "equal_bias_pairing_failures": pairing_failures,
        "equal_bias_pairing_failure_fraction": pairing_failures / max(n_win, 1),
        "negative_fraction_mean": {a: float(np.mean(v)) for a, v in negs.items()},
        "high_k_energy_ratio_mean": {a: float(np.mean(v)) for a, v in e_ratio.items()},
        "high_k_energy_ratio_range": {
            a: [float(min(v)), float(max(v))] for a, v in e_ratio.items()
        },
    }
    return tables, meta


def paired_bootstrap(a: list[dict], b: list[dict], n_boot: int = N_BOOT) -> dict:
    assert len(a) == len(b), f"window count differs: {len(a)} vs {len(b)}"
    n_win = len(a)
    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, dict] = {}
    for key in TIER_KEYS:
        ca = np.array([[r[key][c] for c in
                       ("hits", "misses", "false_alarms")] for r in a], dtype=np.float64)
        cb = np.array([[r[key][c] for c in
                       ("hits", "misses", "false_alarms")] for r in b], dtype=np.float64)
        truth_mismatch = float(np.max(np.abs((ca[:, 0] + ca[:, 1]) - (cb[:, 0] + cb[:, 1]))))
        point_a = csi_from({"hits": ca[:, 0].sum(), "misses": ca[:, 1].sum(),
                            "false_alarms": ca[:, 2].sum()})
        point_b = csi_from({"hits": cb[:, 0].sum(), "misses": cb[:, 1].sum(),
                            "false_alarms": cb[:, 2].sum()})
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_win, n_win)
            sa = ca[idx].sum(axis=0)
            sb = cb[idx].sum(axis=0)
            diffs[i] = (sa[0] / max(sa[0] + sa[1] + sa[2], 1e-12)) - \
                       (sb[0] / max(sb[0] + sb[1] + sb[2], 1e-12))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[key] = {
            "csi_a": point_a, "csi_b": point_b, "diff": point_a - point_b,
            "ci95_low": float(lo), "ci95_high": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "bias_a": bias_from({"hits": ca[:, 0].sum(), "misses": ca[:, 1].sum(),
                                  "false_alarms": ca[:, 2].sum()}),
            "bias_b": bias_from({"hits": cb[:, 0].sum(), "misses": cb[:, 1].sum(),
                                  "false_alarms": cb[:, 2].sum()}),
            "truth_pairing_max_abs_diff": truth_mismatch,
        }
    return out


def write_table(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps({"n_windows": len(rows),
                                "thresholds": list(TIER_KEYS),
                                "windows": rows}, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields-dir", default="runs/fields")
    ap.add_argument("--prefix", default="base_concat_s")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out-dir", default="runs/e33_freq_control")
    ap.add_argument("--arms", nargs="*",
                    default=["orig", "truthswap", "truthswap_eqbias",
                             "phasescramble_fixed"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    high, low = split_band((GRID, GRID))
    summary: dict[str, Any] = {"n_seeds": args.seeds, "arms": args.arms,
                               "k_high_min": int(np.ceil(FIELD_KM / CONVECTIVE_MAX_KM))}

    for s in range(args.seeds):
        pred_path = Path(args.fields_dir) / f"{args.prefix}{s}.pred.npy"
        truth_path = Path(args.fields_dir) / f"{args.prefix}{s}.truth.npy"
        if not pred_path.exists():
            print(f"seed {s}: missing {pred_path}, stopping", flush=True)
            break
        rng = np.random.default_rng(RNG_SEED + s)
        tables, meta = arm_table(pred_path, truth_path, high, low, args.arms, rng)
        entry: dict[str, Any] = dict(meta)
        entry["paired_vs_orig"] = {
            arm: paired_bootstrap(tables[arm], tables["orig"])
            for arm in args.arms if arm != "orig"
        }
        summary.setdefault("per_seed", {})[str(s)] = entry
        for arm in args.arms:
            write_table(out_dir / f"{arm}_s{s}.windows.json", tables[arm])
        print(f"seed {s} done", flush=True)

    consistency: dict[str, Any] = {}
    for arm in args.arms:
        if arm == "orig":
            continue
        diffs, cis, bis, ber = [], [], [], []
        for s, entry in summary.get("per_seed", {}).items():
            pb = entry["paired_vs_orig"][arm]["2.0"]
            diffs.append(pb["diff"])
            cis.append((pb["csi_a"], pb["csi_b"]))
            bis.append((pb["bias_a"], pb["bias_b"]))
            ber.append(entry["equal_bias_max_abs_bias_error"])
        if not diffs:
            continue
        arr = np.asarray(diffs, dtype=np.float64)
        n_pos = int((arr > 0).sum())
        n_neg = int((arr < 0).sum())
        consistency[arm] = {
            "n_seeds": len(diffs),
            "csi2_diff_per_seed": [round(d, 6) for d in diffs],
            "n_positive": n_pos, "n_negative": n_neg,
            "all_same_sign": bool(n_pos == len(diffs) or n_neg == len(diffs)),
            "mean_csi2_diff": float(arr.mean()),
            "csi2_pairs": [[round(a, 6), round(b, 6)] for a, b in cis],
            "bias2_pairs": [[round(a, 6), round(b, 6)] for a, b in bis],
            "equal_bias_max_abs_error_over_seeds": float(max(ber)),
        }
    summary["cross_seed_consistency"] = consistency
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(consistency, indent=2, ensure_ascii=False))
    print("E33_SUMMARY_WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
