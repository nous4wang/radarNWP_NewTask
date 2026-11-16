#!/usr/bin/env python
"""E45 round 63 part B -- mechanism DIRECTION reproducibility in weight space.

Part A (tools/_e45_weight_space_probe.py) measured the MAGNITUDE of each
mechanism's parameter-space footprint vs the seed-noise footprint.
Part B asks the sharper question: is the mechanism's perturbation DIRECTION
reproducible across seeds?

  delta_arm,s = w(arm,seed=s) - w(B0,seed=s)          (same seed, mechanism on)
  cos(delta_arm,s1, delta_arm,s2)                      -> direction reproducibility

Null model (same discipline as rounds 46/52/56/57/59):
  the SAME statistic computed on pure seed noise of the control arm:
  nu_B0,01 = w(B0,s0) - w(B0,s1),  nu_B0,02 = w(B0,s0) - w(B0,s2)
  cos(nu_B0,01, nu_B0,02)                              -> how reproducible a pure
                                                         re-seed is in direction

Also:
  * relative perturbation size ||delta|| / ||w(B0,s)|| (magnitude, denominator fixed)
  * cross-arm direction alignment at the same seed (A3a vs A3b, A4 vs A4b, all pairs)

Read-only. No training, no inference, no GPU.
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/wf/radarNWP"
RUNS = os.path.join(ROOT, "runs", "e45")
OUT = "/tmp/e45_ws_dir63.json"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]
POS = []
FIND = []


def check(name, ok, detail=""):
    POS.append({"name": name, "pass": bool(ok), "detail": str(detail)})


def load_tensors(path):
    import torch
    obj = None
    for kwargs in ({"map_location": "cpu", "weights_only": False},
                   {"map_location": "cpu"}):
        try:
            obj = torch.load(path, **kwargs)
            break
        except Exception:
            continue
    if obj is None:
        raise RuntimeError("cannot load " + path)

    def find_dict(o, depth=0):
        if depth > 6:
            return None
        if isinstance(o, dict):
            if any(torch.is_tensor(v) for v in o.values()):
                return o
            for key in ("state_dict", "model", "model_state_dict", "ema", "net",
                        "network", "params"):
                if key in o:
                    got = find_dict(o[key], depth + 1)
                    if got is not None:
                        return got
            for v in o.values():
                got = find_dict(v, depth + 1)
                if got is not None:
                    return got
        return None

    sd = find_dict(obj)
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to(torch.float64).cpu().numpy()
    return out


def flat(sd, keys):
    return np.concatenate([sd[k].reshape(-1) for k in keys]) if keys else np.zeros(0)


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def main():
    cells = {}
    for arm in ARMS:
        for seed in SEEDS:
            hits = sorted(glob.glob(os.path.join(RUNS, "%s_s%d" % (arm, seed),
                                                 "train", "*", "epoch_00.pt")))
            sd = load_tensors(hits[0])
            cells["%s_s%d" % (arm, seed)] = sd
    check("P01_all_21_cells_loaded", len(cells) == 21, "loaded=%d" % len(cells))

    shared = set(cells["B0_s0"].keys())
    for tag in cells:
        shared &= set(cells[tag].keys())
    shared = sorted(shared)

    W = {tag: flat(sd, shared) for tag, sd in cells.items()}

    # ---- relative perturbation magnitude (fixed denominator) ----------------------
    relmag = {}
    for arm in ARMS[1:]:
        vals = {}
        for s in SEEDS:
            d = W["%s_s%d" % (arm, s)] - W["B0_s%d" % s]
            nb = float(np.linalg.norm(W["B0_s%d" % s]))
            vals["s%d" % s] = float(np.linalg.norm(d)) / nb if nb > 0 else None
        relmag[arm] = {"per_seed": vals,
                       "median": float(np.median([v for v in vals.values() if v is not None]))}
    FIND.append({"item": "relative_perturbation_magnitude", "value": relmag})

    # ---- null model: pure seed-noise direction reproducibility -------------------
    nu = {("%d-%d" % (a, b)): unit(W["B0_s%d" % a] - W["B0_s%d" % b])
          for a, b in [(0, 1), (0, 2), (1, 2)]}
    null_pairs = [("0-1", "0-2"), ("0-1", "1-2"), ("0-2", "1-2")]
    null_cos = [float(np.dot(nu[p], nu[q])) for p, q in null_pairs]
    FIND.append({"item": "null_model_seed_noise_direction_cosine",
                 "per_pair": [{"pair_a": p, "pair_b": q, "cos": float(np.dot(nu[p], nu[q]))}
                              for p, q in null_pairs],
                 "median": float(np.median(null_cos)), "n": len(null_cos)})
    null_median = float(np.median(null_cos))

    # ---- mechanism direction reproducibility across seeds ------------------------
    dir_cos = {}
    for arm in ARMS[1:]:
        d = {s: unit(W["%s_s%d" % (arm, s)] - W["B0_s%d" % s]) for s in SEEDS}
        pairs = [(0, 1), (0, 2), (1, 2)]
        vals = [float(np.dot(d[a], d[b])) for a, b in pairs]
        dir_cos[arm] = {"per_pair": [{"seeds": "%d-%d" % (a, b), "cos": float(np.dot(d[a], d[b]))}
                                     for a, b in pairs],
                        "median": float(np.median(vals)),
                        "mean": float(np.mean(vals)),
                        "n_negative_pairs": int(sum(1 for v in vals if v < 0))}
    FIND.append({"item": "mechanism_direction_cosine_across_seeds", "value": dir_cos})
    check("P02_direction_cosine_computed_for_six_arms", len(dir_cos) == 6,
          "arms=%d" % len(dir_cos))

    # ---- cross-arm direction alignment at the same seed -------------------------
    align = {}
    for i, a1 in enumerate(ARMS[1:]):
        for a2 in ARMS[1:][i + 1:]:
            vals = []
            for s in SEEDS:
                d1 = unit(W["%s_s%d" % (a1, s)] - W["B0_s%d" % s])
                d2 = unit(W["%s_s%d" % (a2, s)] - W["B0_s%d" % s])
                vals.append(float(np.dot(d1, d2)))
            align["%s|%s" % (a1, a2)] = {"per_seed": vals, "median": float(np.median(vals))}
    FIND.append({"item": "cross_arm_direction_alignment_same_seed", "value": align})

    # ---- summary comparison -----------------------------------------------------
    summary = {}
    for arm in ARMS[1:]:
        summary[arm] = {
            "rel_perturbation_median": relmag[arm]["median"],
            "direction_cos_median": dir_cos[arm]["median"],
            "direction_cos_min": min(x["cos"] for x in dir_cos[arm]["per_pair"]),
            "n_negative_direction_pairs": dir_cos[arm]["n_negative_pairs"],
        }
    FIND.append({"item": "arm_summary", "value": summary,
                 "null_model_seed_direction_cos_median": null_median})

    out = {"round": 63, "part": "B_weight_space_direction",
           "shared_keys": shared, "findings": FIND, "pos": POS}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print("=== E45 ROUND 63 PART B (WEIGHT-SPACE DIRECTION) ===")
    print("null model: seed-noise direction cosine median %.4f  (pairs %s)"
          % (null_median, [round(c, 4) for c in null_cos]))
    print("relative perturbation magnitude ||dw||/||w_B0||:")
    for arm in ARMS[1:]:
        print("  %-4s median %.6f  per-seed %s"
              % (arm, relmag[arm]["median"],
                 {k: round(v, 6) for k, v in relmag[arm]["per_seed"].items()}))
    print("mechanism direction cosine across seeds:")
    for arm in ARMS[1:]:
        v = dir_cos[arm]
        print("  %-4s median %+.4f  min %+.4f  negative-pairs %d/3  per-pair %s"
              % (arm, v["median"], min(x["cos"] for x in v["per_pair"]),
                 v["n_negative_pairs"], [round(x["cos"], 4) for x in v["per_pair"]]))
    print("cross-arm direction alignment (same seed), top/bottom:")
    items = sorted(align.items(), key=lambda kv: -kv[1]["median"])
    for k, v in items[:4] + items[-4:]:
        print("  %-10s median %+.4f  per-seed %s" % (k, v["median"],
                                                     [round(x, 4) for x in v["per_seed"]]))
    npass = sum(1 for p in POS if p["pass"])
    print("POS %d/%d" % (npass, len(POS)))
    for p in POS:
        print("  %-46s %s  %s" % (p["name"], "PASS" if p["pass"] else "FAIL", p["detail"]))
    print("VERDICT: %s" % ("PASS" if npass == len(POS) else "FAIL"))
    print("artifact: %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
