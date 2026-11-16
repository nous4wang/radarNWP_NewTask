#!/usr/bin/env python3
"""E31 / §4.33: recompute the STANDARD-LOSS spectra on the SAME window set.

E29's cross-loss pairing was void because the stored standard-loss spectra
were computed on 314 stitched windows while the score-aware ones used 538.
This script reruns scripts/rapsd.py for the seventeen standard-loss
checkpoints with exactly the parameters E29 used, so the two sides live on
the same window set and can be paired model by model.

Zero training compute: only inference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# The seventeen standard-loss models that also exist in the score-aware
# matrix.  pysteps_adv is a physical extrapolation, not a trained model, so
# it is the anchor rather than a pair.  simvp_early_concat has no
# standard-loss checkpoint (pre-registered 2026-11-07 02:03).
STD_MODELS = [
    "convlstm", "predrnn", "earthformer", "simvp_v2",
    "metnet3", "diffcast", "cascast_nwp", "mocast",
    "pw_foucast", "fusioncast", "mmf_rnn", "furecast",
    "simvp_radar_only",
    "simvp_v2_nwp_concat", "simvp_v2_nwp_xattn",
    "simvp_v2_nwp_delta", "simvp_v2_quantum",
]
# The anchor values E29 measured with these exact parameters.
ANCHOR = {
    "simvp_v2": 0.01966529165260319,
    "pysteps_adv": 4.196664147243225,
}
ANCHOR_TOL = 1e-5
N_WINDOWS = 538


def find_ckpt(root: Path, model: str) -> str | None:
    base = root / "runs" / "tgrs" / model
    if not base.is_dir():
        return None
    hits = sorted(base.rglob("last.pt"))
    return str(hits[0]) if hits else None


def band_share(rep: dict) -> float | None:
    """Variance-normalised share of the high-k band."""
    p = np.asarray(rep["predicted"], dtype=np.float64)
    n = np.asarray(rep["n_modes"], dtype=np.float64)
    lo, hi = rep["bands"]["convective"]["k"]
    hi = min(int(hi), len(p) - 1)
    power = p * n
    total = float(power[1:].sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return float(power[int(lo):hi + 1].sum() / total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/wf/radarNWP")
    ap.add_argument("--out-dir", default="runs/e31_std_rapsd")
    ap.add_argument("--score-dir", default="runs/e29_rapsd")
    ap.add_argument("--anchor-dir", default="runs/e29_anchor")
    ap.add_argument("--eval-dir", default="runs/score_matrix")
    ap.add_argument("--out", default="runs/e31_spectral_paired.json")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--data-root", default="/home/wf/data/radarNWP/BJ_mmap")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- the self-check, because the anchor below is meaningless without it
    chk = subprocess.run(
        [args.python, "scripts/rapsd.py", "--self-check"],
        cwd=root, capture_output=True, text=True,
    )
    self_check_ok = chk.returncode == 0
    print("SELF_CHECK_EXIT", chk.returncode, flush=True)
    if not self_check_ok:
        print(chk.stdout[-2000:], chk.stderr[-2000:])

    # ---- the two anchors E29 already produced with these exact parameters
    anchors: dict[str, dict] = {}
    for name in ("simvp_v2", "pysteps_adv"):
        p = root / args.anchor_dir / name / "rapsd.json"
        if p.exists():
            rep = json.loads(p.read_text(encoding="utf-8"))
            anchors[name] = {
                "high_k_ratio": rep["summary"].get("high_k_ratio"),
                "n_samples": rep.get("n_samples"),
                "norm_high_k_share": band_share(rep),
            }
        else:
            anchors[name] = {"high_k_ratio": None, "n_samples": None,
                             "norm_high_k_share": None}
    anchor_pass = all(
        anchors[n]["high_k_ratio"] is not None
        and abs(anchors[n]["high_k_ratio"] - ANCHOR[n]) <= ANCHOR_TOL
        and anchors[n]["n_samples"] == N_WINDOWS
        for n in ANCHOR
    )
    print("ANCHOR_PASS", anchor_pass, json.dumps(anchors), flush=True)

    # ---- rerun the seventeen standard-loss models
    produced: dict[str, dict] = {}
    missing: list[str] = []
    for model in STD_MODELS:
        ck = find_ckpt(root, model)
        if ck is None:
            missing.append(model)
            print(f"SKIP {model}: no standard-loss checkpoint", flush=True)
            continue
        dest = out_dir / model
        if (dest / "rapsd.json").exists():
            print(f"SKIP {model}: already computed", flush=True)
        else:
            cmd = [
                args.python, "scripts/rapsd.py",
                "--data-root", args.data_root, "--packed", "--split", "test",
                "--patch-size", "64", "--slide-stride", "64", "--taper", "none",
                "--model", model, "--checkpoint", ck, "--out-dir", str(dest),
            ]
            r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            print(f"RAPSD {model} exit={r.returncode}", flush=True)
            if r.returncode != 0:
                print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
                missing.append(model)
                continue
        rep = json.loads((dest / "rapsd.json").read_text(encoding="utf-8"))
        produced[model] = {
            "std_high_k_ratio": rep["summary"].get("high_k_ratio"),
            "std_norm_high_k_share": band_share(rep),
            "n_samples": rep.get("n_samples"),
            "n_frames": rep.get("n_frames"),
        }

    # ---- pair against the score-aware spectra E29 produced
    pairs: dict[str, dict] = {}
    for model, std in produced.items():
        p = root / args.score_dir / model / "rapsd.json"
        ep = root / args.eval_dir / f"{model}.eval.json"
        if not p.exists() or not ep.exists():
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        ev = json.loads(ep.read_text(encoding="utf-8"))
        sa_high = rep["summary"].get("high_k_ratio")
        sa_share = band_share(rep)
        std_high = std["std_high_k_ratio"]
        pairs[model] = {
            "std_high_k_ratio": std_high,
            "sa_high_k_ratio": sa_high,
            "elevated_over_std": (None if sa_high is None or not std_high
                                  else bool(sa_high > std_high)),
            "lift_ratio": (None if sa_high is None or not std_high
                           else float(sa_high / std_high)),
            "std_norm_high_k_share": std["std_norm_high_k_share"],
            "sa_norm_high_k_share": sa_share,
            "sa_csi_2_0": ev["thresholds"]["2.0"]["csi"],
            "sa_bias_0_5": ev["thresholds"]["0.5"]["bias"],
            "std_n_samples": std["n_samples"],
            "sa_n_samples": rep.get("n_samples"),
            "same_window_set": (std["n_samples"] == rep.get("n_samples")
                                == N_WINDOWS),
        }

    valid = {m: v for m, v in pairs.items()
             if v["elevated_over_std"] is not None and v["same_window_set"]}
    n_elevated = sum(1 for v in valid.values() if v["elevated_over_std"])
    if len(valid) >= 5:
        a = [v["sa_high_k_ratio"] for v in valid.values()]
        b = [v["std_high_k_ratio"] for v in valid.values()]
        w_stat, w_p = stats.wilcoxon(a, b, alternative="greater")
        wilcoxon = {"statistic": float(w_stat), "p_greater": float(w_p)}
    else:
        wilcoxon = {"statistic": None, "p_greater": None}

    # E29-B must not be overturned by the recomputation.
    shares = [v["sa_norm_high_k_share"] for v in valid.values()]
    csi = [v["sa_csi_2_0"] for v in valid.values()]
    bias = [v["sa_bias_0_5"] for v in valid.values()]
    def corr(x, y):
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5: return None
        return float(stats.pearsonr(x[ok], y[ok])[0])
    def partial(x, y, c):
        x = np.asarray(x, float); y = np.asarray(y, float); c = np.asarray(c, float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        x, y, c = x[ok], y[ok], c[ok]
        if len(x) < 5: return None
        rx = x - np.polyval(np.polyfit(c, x, 1), c)
        ry = y - np.polyval(np.polyfit(c, y, 1), c)
        if np.std(rx) == 0 or np.std(ry) == 0: return None
        return float(stats.pearsonr(rx, ry)[0])
    sa_raw = [v["sa_high_k_ratio"] for v in valid.values()]
    r_raw = corr(sa_raw, csi)
    r_norm = corr(shares, csi)
    r_part = partial(sa_raw, csi, bias)
    e29b_still_negative = (
        r_norm is not None and r_norm < 0.0 and
        (r_part is None or r_part < 0.0)
    )

    summary = {
        "self_check_ok": self_check_ok,
        "anchors": anchors,
        "anchor_pass": anchor_pass,
        "n_std_models_requested": len(STD_MODELS),
        "n_std_models_produced": len(produced),
        "missing_or_failed": missing,
        "n_pairs": len(pairs),
        "n_pairs_same_window_set": len(valid),
        "E31_A": {
            "n_elevated": n_elevated,
            "n_pairs": len(valid),
            "threshold_elevated": 15,
            "wilcoxon_one_sided_greater": wilcoxon,
            "wilcoxon_threshold": 0.001,
            "pass": bool(anchor_pass and self_check_ok and len(valid) >= 5
                         and n_elevated >= 15
                         and wilcoxon["p_greater"] is not None
                         and wilcoxon["p_greater"] < 0.001),
        },
        "E31_B": {
            "r_raw_pearson": r_raw,
            "r_norm_pearson": r_norm,
            "r_partial_raw_given_bias_0_5": r_part,
            "e29b_still_negative": e29b_still_negative,
            "note": "E29-B was measured on 538 windows with the score-aware "
                    "spectra; the standard-loss side is now on the same set",
        },
        "pairs": pairs,
    }
    Path(root / args.out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in
                      ("anchor_pass", "n_pairs", "n_pairs_same_window_set",
                       "E31_A", "E31_B")}, indent=2, ensure_ascii=False))
    print("E31_SUMMARY_WRITTEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
