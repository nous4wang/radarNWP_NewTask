#!/usr/bin/env python3
"""E32 / §4.34: frequency-domain intervention on already-dumped fields.

No inference and no training.  The stitched whole-field predictions from the
E25 batch (runs/fields/base_concat_s{0..9}.{pred,truth}.npy, float32) are
read back and four arms are produced by editing the PREDICTION's frequency
content:

  orig          the prediction as it was scored in E25 (control)
  lowpass       the high-k band (lambda < 10 km) is zeroed out
  truthswap     the high-k band is replaced by the TRUTH's high-k band
                (oracle upper bound on what correct small-scale detail buys)
  phasescramble the high-k band's amplitudes are kept, its phase is
                randomised (structure destroyed, energy spectrum kept)

The radial split uses the same convention as scripts/rapsd.py: modes are
counted in cycles per field, the wavelength of mode k is field_km / k with
field_km = 301 * 1.11195 km, so lambda < 10 km means k >= 34.
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


def mode_wavenumbers(shape: tuple[int, int]) -> np.ndarray:
    """Radial wavenumber of every DFT mode, fftshifted to match fft2."""
    ny, nx = shape
    kx = np.rint(np.fft.fftfreq(nx) * nx)
    ky = np.rint(np.fft.fftfreq(ny) * ny)
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing="ij")
    return np.fft.fftshift(np.hypot(kx_grid, kx_grid))


def split_band(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the high-k band (lambda < 10 km) and its complement."""
    k = mode_wavenumbers(shape)
    k_hi = int(np.ceil(FIELD_KM / CONVECTIVE_MAX_KM))
    high = k >= k_hi
    return high, ~high


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
    """Mean power of the field inside one radial band (Parseval-consistent)."""
    spec = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    return float(spec[mask].mean())


def intervene(
    pred: np.ndarray,
    truth: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    arm: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one frequency-domain intervention to every frame of a window."""
    out = np.empty_like(pred)
    for t in range(pred.shape[0]):
        fp = np.fft.fftshift(np.fft.fft2(pred[t]))
        ft = np.fft.fftshift(np.fft.fft2(truth[t]))
        if arm == "orig":
            new = fp
        elif arm == "lowpass":
            new = fp * low
        elif arm == "truthswap":
            new = fp * low + ft * high
        elif arm == "phasescramble":
            # Keep the amplitude spectrum of the high band, randomise phase.
            hp = fp * high
            amp = np.abs(hp)
            phase = np.angle(hp)
            new_phase = rng.uniform(-np.pi, np.pi, size=hp.shape)
            new = fp * low + amp * np.exp(1j * new_phase) * high
            del phase
        else:
            raise ValueError(f"unknown arm {arm}")
        field = np.real(np.fft.ifft2(np.fft.ifftshift(new)))
        out[t] = field
    return np.clip(out, 0.0, None)


def arm_table(
    pred_path: Path,
    truth_path: Path,
    high: np.ndarray,
    low: np.ndarray,
    arm: str,
    rng: np.random.Generator,
) -> tuple[list[dict], dict]:
    """Per-window contingency tables for one arm, plus energy bookkeeping."""
    preds = np.load(pred_path, mmap_mode="r")
    truths = np.load(truth_path, mmap_mode="r")
    n_win = preds.shape[0]
    rows: list[dict] = []
    hi_before: list[float] = []
    hi_after: list[float] = []
    clipped: list[float] = []
    for w in range(n_win):
        p = np.asarray(preds[w], dtype=np.float64)
        t = np.asarray(truths[w], dtype=np.float64)
        hi_before.append(band_energy(p[0], high))
        new = intervene(p, t, high, low, arm, rng)
        clipped.append(float(np.mean(new != np.clip(new, 0.0, None))))
        hi_after.append(band_energy(new[0], high))
        row = {}
        for thr, key in zip(TIERS, TIER_KEYS):
            row[key] = counts(new, t, thr)
        rows.append(row)
    meta = {
        "n_windows": n_win,
        "high_k_energy_before_mean": float(np.mean(hi_before)),
        "high_k_energy_after_mean": float(np.mean(hi_after)),
        "high_k_energy_ratio": float(np.mean(hi_after) / max(np.mean(hi_before), 1e-12)),
        "clipped_fraction_mean": float(np.mean(clipped)),
    }
    return rows, meta


def paired_bootstrap(a: list[dict], b: list[dict], n_boot: int = N_BOOT) -> dict:
    """Paired bootstrap on the WINDOW counts, arm a minus arm b."""
    assert len(a) == len(b), f"window count differs: {len(a)} vs {len(b)}"
    n_win = len(a)
    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, dict] = {}
    for key in TIER_KEYS:
        ca = np.array([[r[key][c] for c in
                       ("hits", "misses", "false_alarms")] for r in a], dtype=np.float64)
        cb = np.array([[r[key][c] for c in
                       ("hits", "misses", "false_alarms")] for r in b], dtype=np.float64)
        # Same truth field in both arms => hits+misses must match window by window.
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
            "csi_a": point_a,
            "csi_b": point_b,
            "diff": point_a - point_b,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "bias_a": bias_from({"hits": ca[:, 0].sum(), "misses": ca[:, 1].sum(),
                                  "false_alarms": ca[:, 2].sum()}),
            "bias_b": bias_from({"hits": cb[:, 0].sum(), "misses": cb[:, 1].sum(),
                                  "false_alarms": cb[:, 2].sum()}),
            "truth_pairing_max_abs_diff": truth_mismatch,
        }
    return out


def write_table(path: Path, rows: list[dict]) -> None:
    path.write_text(
        json.dumps({"n_windows": len(rows), "thresholds": list(TIER_KEYS),
                    "windows": rows}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields-dir", default="runs/fields")
    ap.add_argument("--prefix", default="base_concat_s")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out-dir", default="runs/e32_freq_intervention")
    ap.add_argument("--arms", nargs="*",
                    default=["orig", "lowpass", "truthswap", "phasescramble"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    high, low = split_band((GRID, GRID))
    k_hi = int(np.ceil(FIELD_KM / CONVECTIVE_MAX_KM))
    print(f"radial split: high band k >= {k_hi} (lambda < {CONVECTIVE_MAX_KM} km), "
          f"field {FIELD_KM:.2f} km")

    summary: dict[str, Any] = {
        "n_seeds": args.seeds,
        "arms": args.arms,
        "k_high_min": k_hi,
        "field_km": FIELD_KM,
        "per_seed": {},
    }

    for s in range(args.seeds):
        pred_path = Path(args.fields_dir) / f"{args.prefix}{s}.pred.npy"
        truth_path = Path(args.fields_dir) / f"{args.prefix}{s}.truth.npy"
        if not pred_path.exists():
            print(f"seed {s}: missing {pred_path}, stopping")
            break
        # A fixed per-seed generator: the phase scramble must be reproducible.
        rng = np.random.default_rng(RNG_SEED + s)
        seed_entry: dict[str, Any] = {}
        tables: dict[str, list[dict]] = {}
        for arm in args.arms:
            rows, meta = arm_table(pred_path, truth_path, high, low, arm, rng)
            tables[arm] = rows
            write_table(out_dir / f"{arm}_s{s}.windows.json", rows)
            seed_entry[arm] = meta
        seed_entry["paired_vs_orig"] = {
            arm: paired_bootstrap(tables[arm], tables["orig"])
            for arm in args.arms if arm != "orig"
        }
        summary["per_seed"][str(s)] = seed_entry
        print(f"seed {s} done", flush=True)

    # Cross-seed consistency: does the sign of the CSI@2.0 effect agree?
    consistency: dict[str, Any] = {}
    if "orig" in args.arms:
        for arm in args.arms:
            if arm == "orig":
                continue
            diffs, cis, bis = [], [], []
            for s, entry in summary["per_seed"].items():
                pb = entry["paired_vs_orig"][arm]["2.0"]
                diffs.append(pb["diff"])
                cis.append((pb["csi_a"], pb["csi_b"]))
                bis.append((pb["bias_a"], pb["bias_b"]))
            if not diffs:
                continue
            arr = np.asarray(diffs, dtype=np.float64)
            n_pos = int((arr > 0).sum())
            n_neg = int((arr < 0).sum())
            consistency[arm] = {
                "n_seeds": len(diffs),
                "csi2_diff_per_seed": [round(d, 6) for d in diffs],
                "n_positive": n_pos,
                "n_negative": n_neg,
                "all_same_sign": bool(n_pos == len(diffs) or n_neg == len(diffs)),
                "mean_csi2_diff": float(arr.mean()),
                "csi2_pairs": [[round(a, 6), round(b, 6)] for a, b in cis],
                "bias2_pairs": [[round(a, 6), round(b, 6)] for a, b in bis],
            }
    summary["cross_seed_consistency"] = consistency

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["cross_seed_consistency"], indent=2, ensure_ascii=False))
    print("E32_SUMMARY_WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
