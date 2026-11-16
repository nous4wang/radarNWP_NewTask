#!/usr/bin/env python3
"""E29 / §4.31: offline summary of the score-aware spectral matrix rescan.

No inference, no training.  Reads the ``rapsd.json`` files produced by
``scripts/rapsd.py`` on the remote host, the seventeen existing standard-loss
spectra that were distilled into ``e29_std_high_k.json``, and the already
existing ``runs/score_matrix/*.eval.json`` verification reports.

It deliberately does NOT pass a verdict: the pre-registered criteria live in
docs/experiments/E29-score-aware-spectral-rescan.md and are judged when the
results are written back.  This script only reports the numbers.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

MODELS_18 = [
    "convlstm", "predrnn", "earthformer", "simvp_v2",
    "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
    "simvp_radar_only", "simvp_early_concat",
    "simvp_v2_nwp_concat", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_v2_quantum",
]
ARMS = {
    "base_concat": "simvp_v2_nwp_concat",
    "qpe_0": "simvp_v2_qpe_guided",
}
EPOCHS = ("epoch_00", "epoch_19")


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def norm_high_share(rep: dict) -> float | None:
    """Band power over all non-DC modes, i.e. the variance-normalised share.

    ``predicted`` holds mean power PER MODE in each radial bin, so the bin
    count has to be put back before summing (Parseval: the count-weighted sum
    equals the field variance).  The DC bin k=0 is excluded because it is the
    field mean, not spatial structure.
    """
    p = np.asarray(rep["predicted"], dtype=np.float64)
    n = np.asarray(rep["n_modes"], dtype=np.float64)
    lo, hi = rep["bands"]["convective"]["k"]
    k_max = len(p) - 1
    hi = min(int(hi), k_max)
    power = p * n
    total = float(power[1:].sum())
    if not math.isfinite(total) or total <= 0:
        return None
    return float(power[int(lo):hi + 1].sum() / total)


def partial_corr(x: np.ndarray, y: np.ndarray, covar: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(covar)
    if ok.sum() < 5:
        return None
    x, y, c = x[ok], y[ok], covar[ok]
    rx = x - np.polyval(np.polyfit(c, x, 1), c)
    ry = y - np.polyval(np.polyfit(c, y, 1), c)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(stats.pearsonr(rx, ry)[0])


def corr(a: list, b: list, kind: str = "pearson") -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return None
    fn = stats.pearsonr if kind == "pearson" else stats.spearmanr
    return float(fn(x[ok], y[ok])[0])


def read_model(rep_path: Path, eval_path: Path, std_high_k: float | None) -> dict:
    rep = load(rep_path)
    ev = load(eval_path)
    sa = rep["summary"]
    out = {
        "sa_high_k_ratio": sa.get("high_k_ratio"),
        "sa_low_k_ratio": sa.get("low_k_ratio"),
        "sa_norm_high_k_share": norm_high_share(rep),
        "sa_n_samples": rep.get("n_samples"),
        "sa_n_frames": rep.get("n_frames"),
        "sa_csi_2_0": ev["thresholds"]["2.0"]["csi"],
        "sa_bias_0_5": ev["thresholds"]["0.5"]["bias"],
        "sa_bias_2_0": ev["thresholds"]["2.0"]["bias"],
        "sa_eval_n_samples": ev.get("n_samples"),
        "std_high_k_ratio": std_high_k,
    }
    if std_high_k is not None and out["sa_high_k_ratio"] is not None:
        out["elevated_over_std"] = bool(out["sa_high_k_ratio"] > std_high_k)
        out["lift_ratio"] = float(out["sa_high_k_ratio"] / std_high_k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-dir", default="runs/e29_rapsd")
    ap.add_argument("--epoch-dir", default="runs/e29_rapsd_epoch")
    ap.add_argument("--anchor-dir", default="runs/e29_anchor")
    ap.add_argument("--std-json", default="runs/e29_std_high_k.json")
    ap.add_argument("--eval-dir", default="runs/score_matrix")
    ap.add_argument("--out", default="runs/e29_spectral_summary.json")
    args = ap.parse_args()

    std = load(Path(args.std_json))
    std_models = std["models"]
    table: dict[str, dict] = {}
    missing_reports: list[str] = []
    for m in MODELS_18:
        rp = Path(args.score_dir) / m / "rapsd.json"
        ep = Path(args.eval_dir) / f"{m}.eval.json"
        if not rp.exists() or not ep.exists():
            missing_reports.append(m)
            continue
        table[m] = read_model(rp, ep, std_models.get(m, {}).get("high_k_ratio"))

    paired = {
        m: v for m, v in table.items()
        if v["std_high_k_ratio"] is not None
        and v["sa_high_k_ratio"] is not None
        and math.isfinite(float(v["sa_high_k_ratio"]))
    }
    n_elevated = sum(1 for v in paired.values() if v.get("elevated_over_std"))
    if len(paired) >= 5:
        a = [v["sa_high_k_ratio"] for v in paired.values()]
        b = [v["std_high_k_ratio"] for v in paired.values()]
        w_stat, w_p = stats.wilcoxon(a, b, alternative="greater")
        wilcoxon = {"statistic": float(w_stat), "p_greater": float(w_p)}
    else:
        wilcoxon = {"statistic": None, "p_greater": None}

    raw = [v["sa_high_k_ratio"] for v in table.values()]
    norm = [v["sa_norm_high_k_share"] for v in table.values()]
    csi = [v["sa_csi_2_0"] for v in table.values()]
    bias = [v["sa_bias_0_5"] for v in table.values()]

    # Anchors: same-parameter re-runs of two standard-loss checkpoints.
    anchor: dict[str, Any] = {}
    for name in ("simvp_v2", "pysteps_adv"):
        p = Path(args.anchor_dir) / name / "rapsd.json"
        if p.exists():
            r = load(p)
            anchor[name] = {
                "high_k_ratio": r["summary"].get("high_k_ratio"),
                "n_samples": r.get("n_samples"),
                "norm_high_k_share": norm_high_share(r),
            }
    anchor["expected"] = {
        "simvp_v2_high_k_ratio": 0.01633818588260064,
        "pysteps_adv_high_k_ratio": 3.5661170724227205,
        "n_samples": 314,
    }
    if "simvp_v2" in anchor and "pysteps_adv" in anchor:
        anchor["simvp_v2_high_k_abs_diff"] = abs(
            anchor["simvp_v2"]["high_k_ratio"] - 0.01633818588260064
        )
        anchor["pysteps_adv_high_k_abs_diff"] = abs(
            anchor["pysteps_adv"]["high_k_ratio"] - 3.5661170724227205
        )
        anchor["n_samples_all_314"] = all(
            anchor[n]["n_samples"] == 314 for n in ("simvp_v2", "pysteps_adv")
        )
        anchor["PASS"] = (
            anchor["simvp_v2_high_k_abs_diff"] <= 1e-4
            and anchor["pysteps_adv_high_k_abs_diff"] <= 1e-3
            and anchor["n_samples_all_314"]
        )

    # Epoch 00 vs 19: two arms, two epochs each.
    epoch: dict[str, Any] = {}
    for arm, _model in ARMS.items():
        epoch[arm] = {}
        for ep in EPOCHS:
            p = Path(args.epoch_dir) / f"{arm}_{ep}" / "rapsd.json"
            if not p.exists():
                continue
            r = load(p)
            epoch[arm][ep] = {
                "high_k_ratio": r["summary"].get("high_k_ratio"),
                "norm_high_k_share": norm_high_share(r),
                "n_samples": r.get("n_samples"),
            }
        if len(epoch[arm]) == 2:
            e0, e19 = epoch[arm]["epoch_00"], epoch[arm]["epoch_19"]
            for key in ("high_k_ratio", "norm_high_k_share"):
                v0, v19 = e0[key], e19[key]
                if v0 and v19:
                    epoch[arm][f"{key}_relative_change"] = (v19 - v0) / abs(v0)

    summary = {
        "n_score_models_run": len(table),
        "missing_score_reports": missing_reports,
        "n_std_loss_spectra_available": len(std_models),
        "std_loss_models_without_spectrum": [
            m for m in MODELS_18 if m not in std_models
        ],
        "E29_A": {
            "n_paired": len(paired),
            "n_elevated": n_elevated,
            "wilcoxon_one_sided_greater": wilcoxon,
        },
        "E29_B": {
            "r_raw_pearson": corr(raw, csi),
            "r_raw_spearman": corr(raw, csi, "spearman"),
            "r_norm_pearson": corr(norm, csi),
            "r_norm_spearman": corr(norm, csi, "spearman"),
            "r_partial_raw_given_bias_0_5": partial_corr(
                np.asarray(raw, dtype=np.float64),
                np.asarray(csi, dtype=np.float64),
                np.asarray(bias, dtype=np.float64),
            ),
            "reference_plus_0_950_is_a_DIFFERENT_population": True,
        },
        "E29_C_epochs": epoch,
        "E29_anchor2": anchor,
        "models": table,
    }
    Path(args.out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
