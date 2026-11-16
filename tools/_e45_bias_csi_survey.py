#!/usr/bin/env python3
"""E45-prep -- is CSI@2.0 essentially a function of BIAS@2.0?

This is a read-only survey over every evaluation that has ever been run
in this project.  It asks one question: across all the arms measured so
far, how much of the CSI@2.0 variation is explained by the bias -- the
total predicted rain against the total observed rain -- and how much is
left over for anything else to explain?

The answer decides where the next branch of work should go.  If CSI is
almost a curve in BIAS, then a mechanism that changes the shape of the
field without changing its amount cannot move the score, and the next
branch has to be about the amount.  If a large residual survives, then
the shape mechanisms have room to work and the deficit is elsewhere.

The survey is deliberately dumb: it reads the evaluation JSON files and
fits two lines.

    CSI = a * f(BIAS) + b

with f a monotone transform, and reports the residual spread.  Two
transforms are fitted, because which one is right matters:

  * identity -- CSI is linear in BIAS;
  * logit -- CSI is linear in the log-odds of the bias, which is the
    natural form if the score is a rate that saturates.

Nothing is trained, nothing is scored here, and no checkpoint is
touched: this only reads report files that already exist.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _read(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _rows_from_report(path: Path) -> List[Dict[str, Any]]:
    """One row per evaluation file: the tag, the bias and the score."""
    d = _read(path)
    if not d:
        return []
    th = d.get("thresholds")
    if not isinstance(th, dict):
        return []
    t2 = th.get("2.0")
    if not isinstance(t2, dict):
        return []
    csi = t2.get("csi")
    bias = t2.get("bias")
    if csi is None or bias is None:
        return []
    try:
        csi = float(csi)
        bias = float(bias)
    except (TypeError, ValueError):
        return []
    if not (math.isfinite(csi) and math.isfinite(bias)):
        return []
    tag = path.name.replace(".eval.json", "")
    return [{
        "tag": tag,
        "path": str(path),
        "csi": csi,
        "bias": bias,
        "n_samples": d.get("n_samples", -1),
        "csi_05": _get(th, "0.5", "csi"),
        "csi_40": _get(th, "4.0", "csi"),
    }]


def _get(th: Dict[str, Any], key: str, field: str):
    v = th.get(key)
    if isinstance(v, dict):
        x = v.get(field)
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    return None


def _logit(bias: np.ndarray) -> np.ndarray:
    b = np.clip(bias, 1e-3, 1.0 - 1e-3)
    return np.log(b / (1.0 - b))


def _fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least squares ``y = a x + b``; returns ``(a, b, r2)``."""
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(b), r2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/home/wf/radarNWP/runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-samples", type=int, default=538,
                    help="only rows scored on the full test window set")
    args = ap.parse_args()

    root = Path(args.runs)
    rows: List[Dict[str, Any]] = []
    for p in sorted(root.rglob("*.eval.json")):
        if "_dump_smoke" in p.name:
            continue
        for r in _rows_from_report(p):
            if args.min_samples > 0 and int(r["n_samples"]) != args.min_samples:
                continue
            rows.append(r)

    print(f"rows read: {len(rows)}")
    if not rows:
        print("no rows")
        return 1

    csi = np.array([r["csi"] for r in rows], dtype=np.float64)
    bias = np.array([r["bias"] for r in rows], dtype=np.float64)

    # --- the two fits ---------------------------------------------------
    a_id, b_id, r2_id = _fit(bias, csi)
    lg = _logit(bias)
    a_lg, b_lg, r2_lg = _fit(lg, csi)

    # a third, monotone but saturating: CSI against sqrt(BIAS)
    a_sq, b_sq, r2_sq = _fit(np.sqrt(np.clip(bias, 0, None)), csi)

    # --- the residual of the best fit -----------------------------------
    best = max((("identity", r2_id), ("logit", r2_lg), ("sqrt", r2_sq)),
               key=lambda kv: (kv[1] if kv[1] == kv[1] else -1))
    name = best[0]
    resid = {
        "identity": csi - (a_id * bias + b_id),
        "logit": csi - (a_lg * lg + b_lg),
        "sqrt": csi - (a_sq * np.sqrt(np.clip(bias, 0, None)) + b_sq),
    }[name]

    out: Dict[str, Any] = {
        "n_rows": len(rows),
        "min_samples": args.min_samples,
        "fits": {
            "identity": {"slope": a_id, "intercept": b_id, "r2": r2_id},
            "logit": {"slope": a_lg, "intercept": b_lg, "r2": r2_lg},
            "sqrt": {"slope": a_sq, "intercept": b_sq, "r2": r2_sq},
        },
        "best_fit": name,
        "best_r2": float(best[1]),
        "residual": {
            "mean": float(resid.mean()),
            "std": float(resid.std()),
            "abs_median": float(np.median(np.abs(resid))),
            "abs_max": float(np.abs(resid).max()),
        },
        "csi_range": [float(csi.min()), float(csi.max())],
        "bias_range": [float(bias.min()), float(bias.max())],
        "rows": sorted(rows, key=lambda r: r["bias"]),
    }

    print(f"identity fit : CSI = {a_id:.6f} * BIAS + {b_id:.6f}   R2 = {r2_id:.6f}")
    print(f"logit fit    : CSI = {a_lg:.6f} * logit(BIAS) + {b_lg:.6f}   R2 = {r2_lg:.6f}")
    print(f"sqrt fit     : CSI = {a_sq:.6f} * sqrt(BIAS) + {b_sq:.6f}   R2 = {r2_sq:.6f}")
    print(f"best: {name}  R2 = {best[1]:.6f}")
    print(f"residual (best fit): mean {resid.mean():+.6f}  "
          f"std {resid.std():.6f}  |median| {np.median(np.abs(resid)):.6f}  "
          f"max {np.abs(resid).max():.6f}")
    print(f"CSI range {csi.min():.6f} .. {csi.max():.6f}")
    print(f"BIAS range {bias.min():.6f} .. {bias.max():.6f}")

    # --- the arms with the largest residual (the interesting ones) ------
    order = np.argsort(-np.abs(resid))
    print("\nlargest residuals (the score these arms have that BIAS does "
          "not explain):")
    for k in order[:12]:
        r = rows[k]
        print(f"  {r['tag']:44s} BIAS {r['bias']:8.4f}  CSI {r['csi']:8.6f}  "
              f"resid {resid[k]:+.6f}")

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
