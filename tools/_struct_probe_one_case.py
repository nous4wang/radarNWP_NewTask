#!/usr/bin/env python3
"""READ-ONLY local structural probe: radar field vs LABEL(RA) on the SAME frame.

Purpose (user 2026-11-08): stop arguing from intuition. Quantify, on real data,
(1) which field is smoother, (2) whether gauge merging leaves amplitude-heavy
tails in LABEL, (3) whether those extreme values sit on coherent large cores.

E41 (2026-11-09) correction: the "strange high values" are extreme in AMPLITUDE
(max/p99 = 4.0 vs CR 1.33), NOT isolated spatial spikes -- LABEL's local-peak
ring contrast (1.03) is lower than CR (1.11), while its connected core is 6x
larger (108.8 px vs 18.2).  C0's "more coherent" finding is confirmed, not
overturned.  VIL has an even heavier tail (8.32) but a much smaller core (28.3).

Zero training / zero model inference / zero remote / zero test-truth leakage.
All metrics are unit-free (z-scored), so dBZ and mm are comparable structurally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("R:/1radarNWPprediction/data/BJ_packed")
DEFAULT_OUT = Path("R:/1radarNWPprediction/radar-nwp-bench/tools/_evals/struct_probe_case.json")
HIGH_FREQ_CUTOFF = 0.5          # same as E39 section 10 / C0 probe
RADAR_VARS = ["CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70",
              "CR", "ET", "HBR", "VIL"]
PROBE_VARS = ["CAP30", "CAP50", "CR", "VIL"]   # low/mid layers, composite, liquid


def radial_grid(shape):
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    r = np.sqrt(ky ** 2 + kx ** 2)
    return r / np.sqrt(0.5 ** 2 + 0.5 ** 2)


def local_maxima(x):
    m = x.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            m = np.maximum(m, np.roll(np.roll(x, dy, 0), dx, 1))
    out = x >= m - 1e-12
    out[:2, :] = out[-2:, :] = False
    out[:, :2] = out[:, -2:] = False
    return out


def ring_median(x, r=2):
    stack = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if abs(dy) <= 1 and abs(dx) <= 1:
                continue
            stack.append(np.roll(np.roll(x, dy, 0), dx, 1))
    return np.median(np.stack(stack, 0), axis=0)


def kernel_mean_px(mask):
    """Mean 4-connected component size, pure numpy BFS on the mask."""
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        if n == 0:
            return 0.0
        sizes = np.bincount(labeled.ravel())[1:]
        return float(sizes.mean()) if sizes.size else 0.0
    except ImportError:
        return None


def metrics(x, grid):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    s = x.std() + 1e-8
    z = (x - x.mean()) / s
    tv = float((np.abs(np.diff(z, axis=0)).mean() +
                np.abs(np.diff(z, axis=1)).mean()) / 2)
    lap = (z[2:, 1:-1] + z[:-2, 1:-1] + z[1:-1, 2:] + z[1:-1, :-2]
           - 4 * z[1:-1, 1:-1])
    lap_e = float(np.abs(lap).mean())
    p = np.abs(np.fft.rfft2(z)) ** 2
    hf = float(p[grid >= HIGH_FREQ_CUTOFF].sum() / p.sum())

    p99 = float(np.quantile(x, 0.99))
    p999 = float(np.quantile(x, 0.999))
    mx = float(x.max())
    lm = local_maxima(x) & (x > p99)
    n_peak = int(lm.sum())
    ring = ring_median(x)
    contrast = x[lm] / (ring[lm] + 0.05 * s + 1e-8) if n_peak else np.array([])
    spike_contrast = float(np.median(contrast)) if contrast.size else None

    # dispersion of top 0.1% pixels over a 16x16 coarse grid (higher = more scattered)
    thr = float(np.quantile(x, 0.999))
    top = x >= thr
    h, w = x.shape
    gh, gw = 16, 16
    cells = set()
    ys, xs = np.nonzero(top)
    for yy, xx in zip(ys, xs):
        cells.add((int(yy * gh // h), int(xx * gw // w)))
    n_top = max(int(top.sum()), 1)
    dispersion = len(cells) / n_top

    zero_frac = float(np.mean(x <= (x.min() + 1e-9)))
    km = kernel_mean_px(x >= p99)
    return {
        "tv_z": round(tv, 6),
        "lap_z": round(lap_e, 6),
        "hf_share": round(hf, 6),
        "n_local_peaks_gt_p99": n_peak,
        "peak_ring_contrast_med": None if spike_contrast is None else round(spike_contrast, 4),
        "tail_p999_over_p99": round(p999 / (p99 + 1e-8), 4),
        "tail_max_over_p99": round(mx / (p99 + 1e-8), 4),
        "top01_dispersion": round(dispersion, 4),
        "zero_frac": round(zero_frac, 6),
        "kernel_px_gt_p99": None if km is None else round(km, 2),
        "p99": round(p99, 4),
        "max": round(mx, 4),
    }


def stamp_map(times):
    return {int(t): i for i, t in enumerate(times)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="directory of packed *.npz cases (also usable on the remote host)")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="where to write the JSON result")
    args = ap.parse_args()
    root, out_path = Path(args.root), Path(args.out)

    grid = None
    cases = []
    focus = None
    focus_name = "CP_01_07291341_00001.npz"   # highest label>=2mm coverage
    files = sorted(root.glob("*.npz"))
    for f in files:
        z = np.load(f)
        lab = np.asarray(z["label"][:, 0], dtype=np.float64)
        rad = np.asarray(z["radar"], dtype=np.float64)
        li = stamp_map(z["label_times"])
        ri = stamp_map(z["radar_times"])
        common = sorted(set(li) & set(ri))
        if not common:
            continue
        # frame with strongest label coverage in the common timeline
        best_t, best_cov = None, -1.0
        for t in common:
            c = float(np.mean(lab[li[t]] >= 2.0))
            if c > best_cov:
                best_t, best_cov = t, c
        L = lab[li[best_t]]
        if grid is None:
            grid = radial_grid(L.shape)
        row = {"case": f.name, "stamp": int(best_t), "label_cov_ge2mm": round(best_cov, 6),
               "n_common_frames": len(common)}
        row["LABEL_RA"] = metrics(L, grid)
        for v in PROBE_VARS:
            j = list(z["radar_vars"]).index(v) if v in list(z["radar_vars"]) else RADAR_VARS.index(v)
            R = rad[ri[best_t], j]
            row[v] = metrics(R, grid)
            if v == "CR":
                a, b = R.ravel(), L.ravel()
                row["corr_CR_vs_LABEL"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
        cases.append(row)
        if f.name == focus_name:
            focus = {"case": f.name, "stamp": int(best_t),
                     "label_cov_ge2mm": round(best_cov, 6),
                     "shape": list(L.shape),
                     # profile through the label global max, for visual comparison
                     "profile": None}
            yy, xx = np.unravel_index(int(np.argmax(L)), L.shape)
            j = list(z["radar_vars"]).index("CR")
            cr = rad[ri[best_t], j]
            sl = slice(max(0, xx - 60), min(L.shape[1], xx + 60))
            focus["profile"] = {
                "row": int(yy), "x0": int(sl.start),
                "CR": [round(float(q), 2) for q in cr[yy, sl]],
                "LABEL": [round(float(q), 2) for q in L[yy, sl]],
            }
            focus["metrics_label"] = row["LABEL_RA"]
            focus["metrics_CR"] = row["CR"]
            focus["corr_CR_vs_LABEL"] = row["corr_CR_vs_LABEL"]

    # aggregate over all cases: median of each metric, per field
    def agg(field):
        keys = [k for k in cases[0][field].keys() if isinstance(cases[0][field][k], (int, float))]
        return {k: round(float(np.median([c[field][k] for c in cases
                                          if c[field][k] is not None])), 4)
                for k in keys}
    summary = {f: agg(f) for f in ["LABEL_RA"] + PROBE_VARS}
    summary["corr_CR_vs_LABEL_median"] = round(float(np.median(
        [c["corr_CR_vs_LABEL"] for c in cases])), 4)
    out = {
        "purpose": "read-only structural comparison radar vs LABEL, one frame per case",
        "n_cases": len(cases),
        "unit_free": True,
        "high_freq_cutoff": HIGH_FREQ_CUTOFF,
        "focus_case": focus,
        "median_over_all_cases": summary,
        "all_cases": cases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("n_cases=", len(cases))
    print("--- MEDIAN over all cases (unit-free) ---")
    hdr = ["metric"] + ["LABEL_RA"] + PROBE_VARS
    print("  ".join(f"{h:>12s}" for h in hdr))
    for k in ["tv_z", "lap_z", "hf_share", "n_local_peaks_gt_p99",
              "peak_ring_contrast_med", "tail_p999_over_p99", "tail_max_over_p99",
              "top01_dispersion", "zero_frac", "kernel_px_gt_p99"]:
        vals = [summary[f].get(k) for f in ["LABEL_RA"] + PROBE_VARS]
        print(f"{k:>26s}  " + "  ".join(f"{v:>11}" if v is not None else f"{'-':>11}" for v in vals))
    print("corr_CR_vs_LABEL_median=", summary["corr_CR_vs_LABEL_median"])
    print("--- FOCUS CASE ---", focus["case"], "stamp", focus["stamp"],
          "cov>=2mm", focus["label_cov_ge2mm"], "corr", focus["corr_CR_vs_LABEL"])
    print("  LABEL_RA:", focus["metrics_label"])
    print("  CR      :", focus["metrics_CR"])
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
