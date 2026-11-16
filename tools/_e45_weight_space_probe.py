#!/usr/bin/env python
"""E45 round 63 -- weight-space distance axis (read-only probe, v2).

NEW AXIS: the 21 ``epoch_00.pt`` checkpoints of E45 (7 arms x 3 seeds) have never
been opened by any previous round (rounds 40-62 read eval/windows/struct/probe/
train-logs/configs, never the parameter tensors themselves).

Question this axis answers (orthogonal to every previous axis):
  How far does each mechanism move the *fitted function* in parameter space,
  compared with how far merely changing the seed moves it?

Null model (same discipline as rounds 46/52/54/57/59):
  within-arm cross-seed distance == "replace the seed, keep the arm" -> seed noise
  within-seed cross-arm distance == "apply the mechanism, keep the seed" -> effect

Confound control (new in this round):
  If two arms had *different random initialisations*, the cross-arm distance would
  be dominated by init difference and the comparison would be meaningless.  We
  therefore compare, PER TENSOR, the arm-vs-control distance against the
  B0-cross-seed distance (a full re-initialisation at the same tensor).  If the
  shared tensors start identical across arms, the arm-vs-control distance must be
  systematically SMALLER than the cross-seed distance at every tensor.

Read-only: loads files, writes one JSON artifact on stdout path. No training,
no inference, no GPU.
"""

import glob
import json
import os
import sys

import numpy as np

ROOT = "/home/wf/radarNWP"
RUNS = os.path.join(ROOT, "runs", "e45")
OUT = "/tmp/e45_ws_round63.json"

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]

POS = []
FIND = []


def check(name, ok, detail=""):
    POS.append({"name": name, "pass": bool(ok), "detail": str(detail)})
    return bool(ok)


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
    if sd is None:
        raise RuntimeError("no tensor dict in " + path)
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to(torch.float64).cpu().numpy()
    return out


def flat(vecs, keys):
    if not keys:
        return np.zeros(0)
    return np.concatenate([vecs[k].reshape(-1) for k in keys])


def rel_l2(x, y):
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0.0 and ny == 0.0:
        return 0.0
    return float(np.linalg.norm(x - y) / (0.5 * (nx + ny)))


def group_of(key):
    return key.rsplit(".", 1)[0] if "." in key else key


def exact_csi(obj, arm, seed):
    """Exact key lookup: obj['<ARM>_s<SEED>_ep00']['csi_at_2.0']."""
    d = obj.get("%s_s%d_ep00" % (arm, seed))
    if isinstance(d, dict) and "csi_at_2.0" in d:
        return float(d["csi_at_2.0"])
    return None


def main():
    cells, meta = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            tag = "%s_s%d" % (arm, seed)
            hits = sorted(glob.glob(os.path.join(RUNS, tag, "train", "*", "epoch_00.pt")))
            if not hits:
                FIND.append({"item": "missing_checkpoint", "cell": tag})
                continue
            sd = load_tensors(hits[0])
            cells[tag] = sd
            meta[tag] = {
                "path": hits[0],
                "n_tensors": len(sd),
                "n_params": int(sum(v.size for v in sd.values())),
                "norm_total": float(sum(float(np.linalg.norm(v.reshape(-1))) for v in sd.values())),
                "nonfinite": int(sum(int((~np.isfinite(v)).sum()) for v in sd.values())),
            }

    n_cells = len(cells)
    check("P01_all_21_cells_loaded", n_cells == 21, "loaded=%d/21" % n_cells)

    total_nonfinite = sum(m["nonfinite"] for m in meta.values())
    FIND.append({"item": "nonfinite_parameters_total", "value": total_nonfinite})
    check("P02_no_nan_inf_parameters", total_nonfinite == 0,
          "nonfinite=%d across %d cells" % (total_nonfinite, n_cells))

    shared = set(cells[ARMS[0] + "_s0"].keys())
    for tag in cells:
        shared &= set(cells[tag].keys())
    shared = sorted(shared)
    FIND.append({"item": "shared_key_count", "value": len(shared),
                 "note": "published round-57 value: n_shared_keys == 66"})
    check("P03_shared_keys_match_published_66", len(shared) == 66,
          "shared=%d (published 66)" % len(shared))

    new_keys = {}
    for arm in ARMS:
        extra = set()
        for seed in SEEDS:
            extra |= set(cells["%s_s%d" % (arm, seed)].keys()) - set(shared)
        new_keys[arm] = sorted(extra)
    FIND.append({"item": "arm_specific_keys", "value": new_keys})

    vecs = {tag: flat(sd, shared) for tag, sd in cells.items()}

    # ---- pairwise distances -----------------------------------------------------
    tags = sorted(cells)
    pair = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            pair.append({"a": a, "b": b, "rel_l2": rel_l2(vecs[a], vecs[b])})
    byid = {(p["a"], p["b"]): p for p in pair}

    def pairrec(a, b):
        return byid[(a, b)] if (a, b) in byid else byid[(b, a)]

    def dist(a, b):
        return pairrec(a, b)["rel_l2"]

    check("P04_distance_matrix_symmetric_zero_diag",
          all(abs(dist(a, b) - dist(b, a)) < 1e-12 for a in tags for b in tags if a != b)
          and all(rel_l2(vecs[a], vecs[a]) == 0.0 for a in tags),
          "%d unordered pairs" % len(pair))

    # ---- null model: within-arm cross-seed --------------------------------------
    seed_noise = []
    for arm in ARMS:
        for s1 in SEEDS:
            for s2 in SEEDS:
                if s1 < s2:
                    seed_noise.append({"arm": arm, "pair": "s%d-s%d" % (s1, s2),
                                       "rel_l2": dist("%s_s%d" % (arm, s1),
                                                      "%s_s%d" % (arm, s2))})
    sn = np.array([x["rel_l2"] for x in seed_noise])

    # ---- effect: arm vs B0, same seed -------------------------------------------
    effect = []
    for arm in ARMS[1:]:
        for seed in SEEDS:
            effect.append({"arm": arm, "seed": seed,
                           "rel_l2": dist("%s_s%d" % (arm, seed), "B0_s%d" % seed)})
    ef = np.array([x["rel_l2"] for x in effect])

    ratio = float(np.median(ef) / np.median(sn))
    FIND.append({"item": "seed_noise_rel_l2",
                 "median": float(np.median(sn)), "min": float(sn.min()),
                 "max": float(sn.max()), "n": int(sn.size),
                 "per_arm_median": {a: float(np.median([x["rel_l2"] for x in seed_noise
                                                        if x["arm"] == a])) for a in ARMS}})
    FIND.append({"item": "mechanism_effect_rel_l2_vs_B0",
                 "median": float(np.median(ef)), "min": float(ef.min()),
                 "max": float(ef.max()), "n": int(ef.size),
                 "per_arm_median": {a: float(np.median([x["rel_l2"] for x in effect
                                                        if x["arm"] == a]))
                                    for a in ARMS[1:]}})
    FIND.append({"item": "effect_over_seed_noise_median_ratio", "value": ratio})

    # fraction of seed-noise distances that exceed the mechanism effect (rank test)
    frac = float(np.mean(sn > float(np.median(ef))))
    FIND.append({"item": "frac_seed_noise_above_median_effect", "value": frac})
    per_arm_ratio = {}
    for arm in ARMS[1:]:
        a_sn = float(np.median([x["rel_l2"] for x in seed_noise if x["arm"] == arm]))
        a_ef = float(np.median([x["rel_l2"] for x in effect if x["arm"] == arm]))
        per_arm_ratio[arm] = {"arm_seed_noise_median": a_sn,
                              "arm_effect_vs_B0_median": a_ef,
                              "ratio_vs_own_seed": (a_ef / a_sn) if a_sn > 0 else None,
                              "effect_vs_global_seed_median": (a_ef / float(np.median(sn)))
                              if np.median(sn) > 0 else None,
                              "per_seed_effect": {str(x["seed"]): x["rel_l2"] for x in effect
                                                  if x["arm"] == arm},
                              "own_seed_noise_values": [x["rel_l2"] for x in seed_noise
                                                        if x["arm"] == arm]}
    FIND.append({"item": "arm_level_effect_vs_own_seed_noise", "value": per_arm_ratio})
    check("P05_effect_distances_computed_for_all_18_cells", ef.size == 18,
          "n=%d" % ef.size)
    check("P06_mechanism_moves_weights_nonzero", float(np.median(ef)) > 0.0,
          "median mechanism rel_l2 = %.6g" % np.median(ef))

    # ---- confound control: per-tensor comparison vs B0 cross-seed --------------
    # per-tensor: d_arm(k) = median over seeds of rel_l2(arm_s[k], B0_s[k])
    #              d_seed(k) = median over B0 seed pairs of rel_l2(B0_s1[k], B0_s2[k])
    b0_seed_pairs = [("B0_s0", "B0_s1"), ("B0_s0", "B0_s2"), ("B0_s1", "B0_s2")]
    d_seed_key = {}
    for k in shared:
        vals = [rel_l2(cells[a][k], cells[b][k]) for a, b in b0_seed_pairs]
        d_seed_key[k] = float(np.median(vals))
    init_ctrl = {}
    for arm in ARMS[1:]:
        d_arm_key, ratios, ident_keys = {}, [], []
        for k in shared:
            vals = [rel_l2(cells["%s_s%d" % (arm, s)][k], cells["B0_s%d" % s][k])
                    for s in SEEDS]
            d_arm_key[k] = float(np.median(vals))
            ds = d_seed_key[k]
            ratios.append(d_arm_key[k] / ds if ds > 0 else None)
            if all(np.array_equal(cells["%s_s%d" % (arm, s)][k], cells["B0_s%d" % s][k])
                   for s in SEEDS):
                ident_keys.append(k)
        rr = [x for x in ratios if x is not None]
        init_ctrl[arm] = {
            "n_tensors_compared": len(shared),
            "n_tensors_identical_to_B0": len(ident_keys),
            "identical_keys": ident_keys[:20],
            "median_ratio_arm_vs_seed": float(np.median(rr)) if rr else None,
            "max_ratio_arm_vs_seed": float(np.max(rr)) if rr else None,
            "frac_tensors_arm_effect_below_seed_reinit": float(np.mean([x < 1.0 for x in rr]))
            if rr else None,
            "frac_tensors_arm_effect_above_half_seed_reinit":
                float(np.mean([x > 0.5 for x in rr])) if rr else None,
        }
    FIND.append({"item": "init_identity_control_per_tensor", "value": init_ctrl})
    check("P07_shared_tensors_start_identical_across_arms",
          all(v["median_ratio_arm_vs_seed"] is not None
              and v["median_ratio_arm_vs_seed"] < 1.0 for v in init_ctrl.values()),
          "median(arm-vs-B0)/median(B0-reinit) per arm: %s"
          % {a: round(v["median_ratio_arm_vs_seed"], 4) for a, v in init_ctrl.items()})

    # ---- per parameter-group breakdown ------------------------------------------
    groups = sorted({group_of(k) for k in shared})
    breakdown = {}
    for arm in ARMS[1:]:
        rows = []
        for g in groups:
            gk = [k for k in shared if group_of(k) == g]
            if not gk:
                continue
            ds = [rel_l2(flat(cells["%s_s%d" % (arm, s)], gk),
                         flat(cells["B0_s%d" % s], gk)) for s in SEEDS]
            rows.append({"group": g, "n_keys": len(gk),
                         "rel_l2_median": float(np.median(ds))})
        rows.sort(key=lambda d: -d["rel_l2_median"])
        breakdown[arm] = rows[:6]
    FIND.append({"item": "top_parameter_groups_changed_vs_B0", "value": breakdown})

    # ---- new-head parameter health ---------------------------------------------
    newhead = {}
    for arm in ARMS[1:]:
        nk = new_keys[arm]
        if not nk:
            continue
        newhead[arm] = [{
            "seed": s, "key": k, "n": int(cells["%s_s%d" % (arm, s)][k].size),
            "norm": float(np.linalg.norm(cells["%s_s%d" % (arm, s)][k].reshape(-1))),
            "absmax": float(np.max(np.abs(cells["%s_s%d" % (arm, s)][k]))),
            "nonzero_frac": float(np.mean(cells["%s_s%d" % (arm, s)][k] != 0.0)),
        } for s in SEEDS for k in nk]
    FIND.append({"item": "new_head_parameter_health", "value": newhead})

    # ---- skill side: EXACT key extraction ---------------------------------------
    per_seed_csi, missing = {}, []
    for seed in SEEDS:
        p = os.path.join(RUNS, "scores_summary_s%d.json" % seed)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        for arm in ARMS:
            v = exact_csi(obj, arm, seed)
            if v is None:
                missing.append("%s_s%d" % (arm, seed))
            else:
                per_seed_csi["%s_s%d" % (arm, seed)] = v
    check("P08_skill_csi_exact_for_all_21_cells", len(per_seed_csi) == 21 and not missing,
          "got %d/21, missing=%s" % (len(per_seed_csi), missing))

    paired_gain, diff_gain = {}, {}
    for arm in ARMS[1:]:
        ds = []
        for seed in SEEDS:
            a = per_seed_csi.get("%s_s%d" % (arm, seed))
            b = per_seed_csi.get("B0_s%d" % seed)
            if a is not None and b is not None:
                ds.append(a - b)
        if ds:
            paired_gain[arm] = {"per_seed": ds, "median": float(np.median(ds))}
            diff_gain[arm] = float(np.median([per_seed_csi["%s_s%d" % (arm, s)] for s in SEEDS])
                                    - np.median([per_seed_csi["B0_s%d" % s] for s in SEEDS]))
    FIND.append({"item": "skill_median_table",
                 "arm_median_csi_at_2.0": {a: float(np.median(
                     [per_seed_csi["%s_s%d" % (a, s)] for s in SEEDS])) for a in ARMS},
                 "paired_gain_median": {a: v["median"] for a, v in paired_gain.items()},
                 "difference_caliber_gain": diff_gain})
    check("P09_paired_gain_computed_for_all_six_arms", len(paired_gain) == 6,
          "arms=%d" % len(paired_gain))

    # ---- dose-response in weight space ------------------------------------------
    ws_med = {a: float(np.median([x["rel_l2"] for x in effect if x["arm"] == a]))
              for a in ARMS[1:]}
    ratio_med = {a: per_arm_ratio[a]["ratio_vs_own_seed"] for a in ARMS[1:]}
    xs = [ws_med[a] for a in ARMS[1:]]
    ys = [paired_gain[a]["median"] for a in ARMS[1:]]
    zs = [ratio_med[a] for a in ARMS[1:]]

    def pearson(u, v):
        return float(np.corrcoef(u, v)[0, 1])

    def spearman(u, v):
        def rank(z):
            order = np.argsort(z)
            r = np.empty(len(z), dtype=float)
            r[order] = np.arange(len(z), dtype=float)
            return r
        return pearson(rank(np.asarray(u)), rank(np.asarray(v)))

    rng = np.random.default_rng(20261114)
    perm_p_ws = perm_p_ratio = None
    if len(xs) >= 4:
        obs_ws = pearson(xs, ys)
        obs_ratio = pearson(zs, ys)
        cnt_ws = cnt_ratio = 0
        n_perm = 20000
        for _ in range(n_perm):
            py = rng.permutation(ys)
            if abs(pearson(xs, py)) >= abs(obs_ws) - 1e-15:
                cnt_ws += 1
            if abs(pearson(zs, py)) >= abs(obs_ratio) - 1e-15:
                cnt_ratio += 1
        perm_p_ws = (cnt_ws + 1) / (n_perm + 1)
        perm_p_ratio = (cnt_ratio + 1) / (n_perm + 1)
    FIND.append({"item": "weight_space_dose_response",
                 "pearson_rel_l2_vs_paired_gain": pearson(xs, ys),
                 "spearman_rel_l2_vs_paired_gain": spearman(xs, ys),
                 "pearson_ratio_vs_paired_gain": pearson(zs, ys),
                 "permutation_p_rel_l2": perm_p_ws,
                 "permutation_p_ratio": perm_p_ratio,
                 "n_perm": 20000,
                 "arms": ARMS[1:],
                 "weight_rel_l2_median": ws_med,
                 "effect_over_own_seed_ratio": ratio_med,
                 "paired_gain_median": {a: paired_gain[a]["median"] for a in paired_gain}})

    out = {"round": 63, "axis": "weight_space_distance", "remote_root": ROOT,
           "cells": meta, "shared_keys": shared, "arm_specific_keys": new_keys,
           "pairwise": pair, "findings": FIND, "pos": POS}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print("=== E45 ROUND 63 WEIGHT-SPACE AXIS (v2) ===")
    print("cells loaded: %d/21 | shared keys %d (published 66) | arm-specific %s"
          % (n_cells, len(shared), {k: len(v) for k, v in new_keys.items() if v}))
    print("nonfinite params total: %d" % total_nonfinite)
    print("seed-noise rel_l2:  median %.6f  min %.6f  max %.6f (n=%d)"
          % (np.median(sn), sn.min(), sn.max(), sn.size))
    print("mechanism rel_l2:   median %.6f  min %.6f  max %.6f (n=%d)"
          % (np.median(ef), ef.min(), ef.max(), ef.size))
    print("effect/seed-noise median ratio: %.4f | frac(seed-noise > median effect) = %.3f"
          % (ratio, frac))
    for arm in ARMS[1:]:
        v = per_arm_ratio[arm]
        print("  %-4s effect %.6f  own-seed-noise %.6f  ratio %.4f  per-seed %s"
              % (arm, v["arm_effect_vs_B0_median"], v["arm_seed_noise_median"],
                 v["ratio_vs_own_seed"],
                 {k: round(x, 6) for k, x in v["per_seed_effect"].items()}))
    print("init-identity control (median per-tensor arm-vs-B0 / B0-reinit):")
    for arm in ARMS[1:]:
        v = init_ctrl[arm]
        print("  %-4s ratio %.4f  max %.4f  frac<1.0 %.3f  frac>0.5 %.3f  identical-tensors %d/%d"
              % (arm, v["median_ratio_arm_vs_seed"], v["max_ratio_arm_vs_seed"],
                 v["frac_tensors_arm_effect_below_seed_reinit"],
                 v["frac_tensors_arm_effect_above_half_seed_reinit"],
                 v["n_tensors_identical_to_B0"], v["n_tensors_compared"]))
    print("skill medians: %s" % {a: round(float(np.median(
        [per_seed_csi["%s_s%d" % (a, s)] for s in SEEDS])), 6) for a in ARMS})
    print("paired gain median: %s" % {a: round(v["median"], 6) for a, v in paired_gain.items()})
    print("difference-caliber gain: %s" % {a: round(v, 6) for a, v in diff_gain.items()})
    print("dose-response: pearson(rel_l2,gain)=%.4f (perm p=%.4f)  pearson(ratio,gain)=%.4f (perm p=%.4f)"
          % (pearson(xs, ys), perm_p_ws or -1, pearson(zs, ys), perm_p_ratio or -1))
    for arm in ARMS[1:]:
        print("top groups %-4s: %s" % (arm, [(r["group"], round(r["rel_l2_median"], 5))
                                              for r in breakdown[arm][:4]]))
    npass = sum(1 for p in POS if p["pass"])
    print("POS %d/%d" % (npass, len(POS)))
    for p in POS:
        print("  %-46s %s  %s" % (p["name"], "PASS" if p["pass"] else "FAIL", p["detail"]))
    print("VERDICT: %s" % ("PASS" if npass == len(POS) else "FAIL"))
    print("artifact: %s" % OUT)


if __name__ == "__main__":
    sys.exit(main())
