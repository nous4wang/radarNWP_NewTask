"""E39 C0 -- read-only data-premise probe on the TRAIN split.

Protocol is frozen in docs/experiments/E39-...md section 10 (2026-11-08
19:46), BEFORE this script was first run. Nothing here is a training run,
a model forward pass, or an evaluation: it only reads packed mmap arrays
and writes descriptive statistics plus a bootstrap CI.

    python tools/_c0_probe.py --out /tmp/c0_result.json

The script deliberately reads val/test case NAMES (for an audit trail) but
never opens a val/test data file.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/home/wf/data/radarNWP/BJ_mmap")
REPO_ROOT = Path("/home/wf/radarNWP")
PROTOCOL = "E39 §10 (frozen 2026-11-08 19:46, before first execution)"

# Frozen in §10.1 / §10.3 / §10.4 -- do not tune after seeing results.
MAX_FRAMES = 24
LABEL_THRESHOLD_MM = 2.0
MIN_COVERAGE = 0.001
RADAR_STRONG_DBZ = 35.0
HIGH_FREQ_CUTOFF = 0.5           # fraction of r_max, outermost half of band
N_BOOTSTRAP = 10000
BOOT_SEED = 20261108
MIN_CASES = 20
FALLBACK_CR_INDEX = 6            # RADAR_VARS: CAP20..CAP70, CR, ET, HBR, VIL

FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


def _radial_grid(shape: tuple[int, int]) -> np.ndarray:
    """Normalised radial frequency r / r_max for an rfft2 power spectrum."""
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    r = np.sqrt(ky * ky + kx * kx)
    r_max = float(np.sqrt(0.5 ** 2 + 0.5 ** 2))
    return r / r_max


def _high_frequency_ratio(field: np.ndarray, grid: np.ndarray) -> float | None:
    """RAPSD high-frequency energy share, z-scored first (unit-free)."""
    x = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0,
                      posinf=0.0, neginf=0.0)
    std = float(x.std())
    if std <= 0.0:
        return None                                    # §10.3: skip degenerate frame
    z = (x - x.mean()) / std
    power = np.abs(np.fft.rfft2(z)) ** 2
    power[0, 0] = 0.0                                  # remove DC
    denom = float(power[grid > 0].sum())
    if denom <= 0.0:
        return None
    return float(power[grid >= HIGH_FREQ_CUTOFF].sum() / denom)


def _mean_kernel_pixels(mask: np.ndarray, label_fn) -> float:
    """Mean connected-component size; log(max(K,1)) keeps §10.3 well-defined."""
    _, n = label_fn(mask, structure=FOUR_CONNECTED)
    if n == 0:
        return 0.0
    return float(mask.sum()) / int(n)


def _bootstrap_ci(values: np.ndarray, seed: int = BOOT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _time_indices(times: list) -> dict[str, int]:
    return {str(t): i for i, t in enumerate(times)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/c0_result.json")
    args = ap.parse_args()

    result: dict = {
        "probe": "E39 C0",
        "protocol": PROTOCOL,
        "executed_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(DATA_ROOT),
        "frozen_constants": {
            "max_frames_per_case": MAX_FRAMES,
            "label_threshold_mm": LABEL_THRESHOLD_MM,
            "min_coverage": MIN_COVERAGE,
            "radar_strong_dbz": RADAR_STRONG_DBZ,
            "high_freq_cutoff_r_over_rmax": HIGH_FREQ_CUTOFF,
            "n_bootstrap": N_BOOTSTRAP,
            "boot_seed": BOOT_SEED,
            "min_cases": MIN_CASES,
        },
        "verdict": "INCONCLUSIVE",
        "notes": [],
    }

    try:
        from scipy import ndimage as _ndi
        label_fn = _ndi.label
        result["notes"].append("connected components: scipy.ndimage.label, 4-connectivity")
    except Exception as exc:                                   # noqa: BLE001
        result["notes"].append(f"scipy unavailable ({type(exc).__name__}) -> INCONCLUSIVE")
        label_fn = None

    try:
        sys.path.insert(0, str(REPO_ROOT))
        from radar_nwp_bench.data.index import split_cases       # noqa: E402
    except Exception as exc:                                   # noqa: BLE001
        result["notes"].append(f"cannot import split_cases ({type(exc).__name__}) -> INCONCLUSIVE")
        split_cases = None

    out_path = Path(args.out)
    if label_fn is None or split_cases is None:
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("VERDICT: INCONCLUSIVE (missing scipy or split_cases)")
        return 0

    try:
        all_cases = sorted(
            p.name for p in DATA_ROOT.iterdir()
            if p.is_dir() and (p / "radar.npy").is_file()
        )
        parts = split_cases(all_cases)
        train = list(parts["train"])
        result["split_audit"] = {
            "n_cases_total": len(all_cases),
            "n_train": len(train),
            "n_val": len(parts["val"]),
            "n_test": len(parts["test"]),
            "train_first": train[0] if train else None,
            "train_last": train[-1] if train else None,
            "val_first": parts["val"][0] if parts["val"] else None,
            "test_first": parts["test"][0] if parts["test"] else None,
            "files_opened": "train case directories only",
        }

        per_case: list[dict] = []
        skipped_no_overlap: list[str] = []
        skipped_no_frames: list[str] = []

        for case_id in train:
            case_dir = DATA_ROOT / case_id
            meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
            radar = np.load(case_dir / "radar.npy", mmap_mode="r")
            label = np.load(case_dir / "label.npy", mmap_mode="r")

            rv = [str(v) for v in meta.get("radar_vars", [])]
            cr_idx = rv.index("CR") if "CR" in rv else FALLBACK_CR_INDEX
            r_idx = _time_indices(meta.get("radar_times", []))
            l_idx = _time_indices(meta.get("label_times", []))
            common = sorted(set(r_idx) & set(l_idx), key=lambda s: (len(s), s))
            if not common:
                skipped_no_overlap.append(case_id)
                continue
            if len(common) > MAX_FRAMES:
                pick = np.linspace(0, len(common) - 1, MAX_FRAMES).astype(int)
                common = [common[int(i)] for i in pick]

            shape = tuple(label.shape[-2:])
            grid = _radial_grid(shape)
            d_hf, d_kernel, p_label, p_cr35 = [], [], [], []

            for stamp in common:
                L = np.asarray(label[l_idx[stamp], 0], dtype=np.float64)
                R = np.asarray(radar[r_idx[stamp], cr_idx], dtype=np.float64)
                L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
                R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                if L.shape != R.shape or L.shape != shape:
                    # A shape mismatch is an unforeseen condition (§10.4):
                    # skip the frame and record it, never silently reshape.
                    result["notes"].append(
                        f"shape mismatch in {case_id}/{stamp}: L{L.shape} R{R.shape}"
                    )
                    continue

                cov = float(np.mean(L >= LABEL_THRESHOLD_MM))
                cov = max(cov, MIN_COVERAGE)
                thr_r = float(np.quantile(R, 1.0 - cov))
                cr35 = float(np.mean(R >= RADAR_STRONG_DBZ))

                hf_l = _high_frequency_ratio(L, grid)
                hf_r = _high_frequency_ratio(R, grid)
                if hf_l is None or hf_r is None:
                    continue

                k_l = _mean_kernel_pixels(L >= LABEL_THRESHOLD_MM, label_fn)
                k_r = _mean_kernel_pixels(R >= thr_r, label_fn)
                # §10.3 uses log(K); max(K,1) is the n=0 degenerate guard.
                dk = float(np.log(max(k_l, 1.0)) - np.log(max(k_r, 1.0)))

                d_hf.append(hf_l - hf_r)
                d_kernel.append(dk)
                p_label.append(cov)
                p_cr35.append(cr35)

            if not d_hf:
                skipped_no_frames.append(case_id)
                continue
            per_case.append({
                "case_id": case_id,
                "n_frames": len(d_hf),
                "d_hf": float(np.mean(d_hf)),
                "d_kernel": float(np.mean(d_kernel)),
                "p_label_ge2mm": float(np.mean(p_label)),
                "p_cr35dbz": float(np.mean(p_cr35)),
                "cr_index": cr_idx,
                "shape": list(shape),
            })

        n = len(per_case)
        result["cases"] = per_case
        result["skipped_no_time_overlap"] = skipped_no_overlap
        result["skipped_no_usable_frame"] = skipped_no_frames
        result["n_cases_used"] = n

        dhf = np.array([c["d_hf"] for c in per_case], dtype=np.float64)
        dker = np.array([c["d_kernel"] for c in per_case], dtype=np.float64)
        pl = np.array([c["p_label_ge2mm"] for c in per_case], dtype=np.float64)
        pc = np.array([c["p_cr35dbz"] for c in per_case], dtype=np.float64)

        ci_hf = _bootstrap_ci(dhf) if n >= 2 else [float("nan"), float("nan")]
        ci_ker = _bootstrap_ci(dker) if n >= 2 else [float("nan"), float("nan")]
        result["aggregate"] = {
            "mean_d_hf": float(dhf.mean()) if n else None,
            "ci95_d_hf": ci_hf,
            "mean_d_kernel": float(dker.mean()) if n else None,
            "ci95_d_kernel": ci_ker,
            "mean_p_label_ge2mm": float(pl.mean()) if n else None,
            "mean_p_cr35dbz": float(pc.mean()) if n else None,
            "p_label_lt_p_cr35": bool(pl.mean() < pc.mean()) if n else False,
        }

        checks = {
            "min_cases": n >= MIN_CASES,
            "B1_ci_low_strictly_positive": bool(ci_hf[0] > 0),
            "B2_ci_high_strictly_negative": bool(ci_ker[1] < 0),
            "auxiliary_sparsity_consistent": bool(pl.mean() < pc.mean()) if n else False,
        }
        result["checks"] = checks

        # §10.4: never relax. Missing sample / inconsistent auxiliary -> INCONCLUSIVE.
        if not checks["min_cases"]:
            result["verdict"] = "INCONCLUSIVE"
            result["notes"].append(f"n_cases={n} < {MIN_CASES}")
        elif not checks["auxiliary_sparsity_consistent"]:
            result["verdict"] = "INCONCLUSIVE"
            result["notes"].append("label >=2mm coverage not sparser than radar >=35dBZ")
        elif checks["B1_ci_low_strictly_positive"] and checks["B2_ci_high_strictly_negative"]:
            result["verdict"] = "PASS"
        else:
            result["verdict"] = "FAIL"
            result["notes"].append(
                "B1 or B2 CI crosses zero / points the wrong way -> §5 C0 discards the route"
            )

        a = result["aggregate"]
        print(
            f"VERDICT: C0 {result['verdict']} | n_cases={n} | "
            f"d_hf={a['mean_d_hf']:.6f} CI[{ci_hf[0]:.6f},{ci_hf[1]:.6f}] | "
            f"d_kernel={a['mean_d_kernel']:.6f} CI[{ci_ker[0]:.6f},{ci_ker[1]:.6f}] | "
            f"p_label={a['mean_p_label_ge2mm']:.6f} p_cr35={a['mean_p_cr35dbz']:.6f}"
        )
    except Exception:                                          # noqa: BLE001
        result["verdict"] = "INCONCLUSIVE"
        result["error"] = traceback.format_exc()
        result["notes"].append("unexpected exception -> INCONCLUSIVE, no criteria relaxed")
        print(f"VERDICT: INCONCLUSIVE (exception: {result['error'].splitlines()[-1]})")
    finally:
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
