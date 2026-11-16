#!/usr/bin/env python
"""E45 round 63 -- consolidate the weight-space axis (parts A + B) locally.

Reads the two artifacts downloaded from the remote read-only probe and:
  1. re-verifies the S3 guard (12/12 intervals cross zero, per-arm verdicts),
  2. re-verifies the published headline table (7 arm medians) and the published
     paired-gain medians against a local recomputation from the per-seed files,
  3. consolidates the weight-space readings (magnitudes, directions, cross-arm
     alignment, rank of the mechanism effect inside the seed-noise distribution),
  4. writes tools/_evals/e45/round63_weight_space.json and prints the summary.

Read-only w.r.t. the project; writes only the round artifact.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "_evals", "e45")
OUT = os.path.join(EV, "round63_weight_space.json")

ARMS = ["B0", "A1", "A2", "A3a", "A3b", "A4", "A4b"]
SEEDS = [0, 1, 2]

# published headline (HANDOVER §4 / round-53 table) -- median CSI@2.0 over 3 seeds
PUB_MEDIAN = {"B0": 0.121817, "A1": 0.118397, "A2": 0.112003, "A3a": 0.110015,
              "A3b": 0.116055, "A4": 0.143896, "A4b": 0.137567}
# published paired-gain medians (round 52 / round 53 tables)
PUB_PAIRED = {"A1": -0.003420, "A2": -0.009814, "A3a": 0.015529, "A3b": 0.024357,
              "A4": 0.013426, "A4b": 0.015750}
PUB_DIFF = {"A1": -0.003420, "A2": -0.009814, "A3a": -0.011802, "A3b": -0.005762,
            "A4": 0.022079, "A4b": 0.015750}
PUB_VERDICT = {"A1": "GAIN_WAS_AREA_OR_AMPLITUDE", "A2": "NO_GAIN_TO_BEGIN_WITH",
               "A3a": "GAIN_WAS_AREA_OR_AMPLITUDE", "A3b": "GAIN_WAS_AREA_OR_AMPLITUDE",
               "A4": "GAIN_WAS_AREA_OR_AMPLITUDE", "A4b": "NO_GAIN_TO_BEGIN_WITH"}

POS = []


def check(name, ok, detail=""):
    POS.append({"name": name, "pass": bool(ok), "detail": str(detail)})
    return bool(ok)


def load(name):
    with open(os.path.join(EV, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


def find(items, key):
    """Return the full finding entry (some carry 'value', some carry fields inline)."""
    for it in items:
        if it.get("item") == key:
            return it
    return None


def find_value(items, key):
    it = find(items, key)
    if it is None:
        return None
    return it["value"] if "value" in it else it


def main():
    A = load("round63_weight_space_raw.json")
    B = load("round63_weight_space_dir.json")

    # ---- 1. S3 guard ------------------------------------------------------------
    s3 = load("s3_equalbias.json")
    cz = []
    verdicts = {}
    for arm, rec in s3["arms"].items():
        cz.append(bool(rec["raw"]["crosses_zero"]))
        cz.append(bool(rec["equal_bias"]["crosses_zero"]))
        verdicts[arm] = rec["s3_verdict"]
    check("G01_s3_all_12_intervals_cross_zero", all(cz) and len(cz) == 12,
          "crosses_zero count %d/12" % sum(cz))
    check("G02_s3_verdicts_unchanged", verdicts == PUB_VERDICT, str(verdicts))
    check("G03_qualified_arm_set_still_empty",
          all(v in ("GAIN_WAS_AREA_OR_AMPLITUDE", "NO_GAIN_TO_BEGIN_WITH")
              for v in verdicts.values()),
          "no arm reached GAIN_SURVIVES_EQUALISATION")

    # ---- 2. headline table + published gains ------------------------------------
    per_seed = {}
    for seed in SEEDS:
        obj = load("scores_summary_s%d.json" % seed)
        for arm in ARMS:
            d = obj["%s_s%d_ep00" % (arm, seed)]
            per_seed["%s_s%d" % (arm, seed)] = float(d["csi_at_2.0"])
    med = {a: float(np.median([per_seed["%s_s%d" % (a, s)] for s in SEEDS])) for a in ARMS}
    dev = max(abs(med[a] - PUB_MEDIAN[a]) for a in ARMS)
    check("G04_headline_medians_match_published", dev < 1e-5,
          "max|diff| = %.3e (published vs recomputed)" % dev)

    paired, diffc = {}, {}
    for arm in ARMS[1:]:
        ds = [per_seed["%s_s%d" % (arm, s)] - per_seed["B0_s%d" % s] for s in SEEDS]
        paired[arm] = float(np.median(ds))
        diffc[arm] = (float(np.median([per_seed["%s_s%d" % (arm, s)] for s in SEEDS]))
                      - float(np.median([per_seed["B0_s%d" % s] for s in SEEDS])))
    dev_p = max(abs(paired[a] - PUB_PAIRED[a]) for a in ARMS[1:])
    dev_d = max(abs(diffc[a] - PUB_DIFF[a]) for a in ARMS[1:])
    check("G05_paired_gain_matches_published", dev_p < 1e-5,
          "max|diff| = %.3e" % dev_p)
    check("G06_difference_caliber_matches_published", dev_d < 1e-5,
          "max|diff| = %.3e" % dev_d)

    # ---- 3. weight-space consolidation ------------------------------------------
    pairs = A["pairwise"]
    byid = {(p["a"], p["b"]): p["rel_l2"] for p in pairs}

    def d(a, b):
        return byid[(a, b)] if (a, b) in byid else byid[(b, a)]

    seed_noise = {}
    effect = {}
    for arm in ARMS:
        seed_noise[arm] = sorted(d("%s_s%d" % (arm, s1), "%s_s%d" % (arm, s2))
                                 for s1 in SEEDS for s2 in SEEDS if s1 < s2)
    for arm in ARMS[1:]:
        effect[arm] = {s: d("%s_s%d" % (arm, s), "B0_s%d" % s) for s in SEEDS}

    sn_all = sorted(v for arm in ARMS for v in seed_noise[arm])
    ef_all = sorted(v for arm in ARMS[1:] for v in effect[arm].values())
    sn_med = float(np.median(sn_all))
    ef_med = float(np.median(ef_all))
    per_arm_ratio = find_value(A["findings"], "arm_level_effect_vs_own_seed_noise")

    # rank: where do the 18 mechanism distances sit inside the 21 seed-noise distances
    rank_rows = []
    for arm in ARMS[1:]:
        mx = max(effect[arm].values())
        n_above = sum(1 for v in sn_all if v > mx)
        rank_rows.append({"arm": arm,
                          "effect_per_seed": {str(k): v for k, v in effect[arm].items()},
                          "effect_max": mx,
                          "own_seed_noise": seed_noise[arm],
                          "own_seed_noise_min": seed_noise[arm][0],
                          "own_seed_noise_max": seed_noise[arm][-1],
                          "n_seed_noise_above_effect_max": n_above,
                          "effect_max_below_own_seed_min": bool(mx < seed_noise[arm][0])})
    n_effect_below_median_noise = int(sum(1 for v in ef_all if v < sn_med))
    exceptions = sorted(["%s_s%s" % (r["arm"], s) for r in rank_rows
                          for s, v in r["effect_per_seed"].items() if v >= sn_med])
    check("G07_effect_magnitude_computed_18_cells", len(ef_all) == 18, "n=%d" % len(ef_all))
    # FACT (not an assumption): enumerate the rank of the 18 mechanism distances
    # inside the 21 seed-noise distances.  The count is recorded so that a later
    # re-run detects drift; the interpretation is carried in the write-back.
    check("G08_effect_rank_inside_seed_noise_enumerated",
          n_effect_below_median_noise == 16 and exceptions == ["A4_s1", "A4b_s2"],
          "%d/18 below the seed-noise median %.6f; exceptions %s"
          % (n_effect_below_median_noise, sn_med, exceptions))
    # per-arm: the MEDIAN mechanism effect is below the MEDIAN of its own seed noise
    check("G08b_median_effect_below_median_own_seed_noise_all_six_arms",
          all(per_arm_ratio[a]["arm_effect_vs_B0_median"]
              < per_arm_ratio[a]["arm_seed_noise_median"] for a in ARMS[1:]),
          "per-arm ratio: %s" % {a: round(per_arm_ratio[a]["ratio_vs_own_seed"], 4)
                                  for a in ARMS[1:]})

    # A3a vs A3b mutual distance (the "two different topo mechanisms" pair)
    a3ab = {s: d("A3a_s%d" % s, "A3b_s%d" % s) for s in SEEDS}
    a4a4b = {s: d("A4_s%d" % s, "A4b_s%d" % s) for s in SEEDS}
    a12 = {s: d("A1_s%d" % s, "A2_s%d" % s) for s in SEEDS}

    dir_cos = find_value(B["findings"], "mechanism_direction_cosine_across_seeds")
    null_dir = find(B["findings"], "null_model_seed_noise_direction_cosine")
    align = find_value(B["findings"], "cross_arm_direction_alignment_same_seed")
    relmag = find_value(B["findings"], "relative_perturbation_magnitude")

    null_med = float(null_dir["median"])
    dir_rows = []
    for arm in ARMS[1:]:
        v = dir_cos[arm]
        pair_cos = [x["cos"] for x in v["per_pair"]]
        dir_rows.append({"arm": arm, "median": v["median"], "min": float(min(pair_cos)),
                         "n_negative_pairs": v["n_negative_pairs"],
                         "per_pair": v["per_pair"],
                         "above_null_median": bool(v["median"] > null_med)})
    n_above = sum(1 for r in dir_rows if r["above_null_median"])
    check("G09_direction_cosine_for_six_arms", len(dir_rows) == 6, "arms=%d" % len(dir_rows))
    check("G10_null_model_direction_cosine_available", null_med is not None,
          "null median = %.4f" % null_med)

    init_ctrl = find_value(A["findings"], "init_identity_control_per_tensor")
    check("G11_init_identity_control_passed_for_all_arms",
          all(v["median_ratio_arm_vs_seed"] < 1.0 for v in init_ctrl.values()),
          "median per-tensor ratio: %s"
          % {a: round(v["median_ratio_arm_vs_seed"], 4) for a, v in init_ctrl.items()})

    out = {
        "round": 63, "axis": "weight_space_distance_and_direction",
        "generated": "local consolidation of two remote read-only probes",
        "scripts": ["tools/_e45_weight_space_probe.py (remote, part A)",
                    "tools/_e45_weight_direction_probe.py (remote, part B)"],
        "guards": {"s3_intervals_cross_zero": "%d/12" % sum(cz),
                   "s3_verdicts": verdicts,
                   "headline_max_abs_dev": dev,
                   "paired_gain_max_abs_dev": dev_p,
                   "difference_caliber_max_abs_dev": dev_d,
                   "headline_median_csi_at_2.0": med,
                   "paired_gain_median": paired,
                   "difference_caliber_gain": diffc},
        "magnitude": {
            "seed_noise_rel_l2": {"median": sn_med, "min": sn_all[0],
                                   "max": sn_all[-1], "n": len(sn_all),
                                   "per_arm": {a: {"median": float(np.median(seed_noise[a])),
                                                   "values": seed_noise[a]} for a in ARMS}},
            "mechanism_rel_l2_vs_B0": {"median": ef_med, "min": ef_all[0],
                                       "max": ef_all[-1], "n": len(ef_all)},
            "effect_over_seed_noise_median_ratio": ef_med / sn_med,
            "n_effect_below_seed_noise_median": n_effect_below_median_noise,
            "per_arm": rank_rows,
            "relative_perturbation_magnitude": relmag,
            "cross_arm_mutual_distance": {"A3a|A3b": a3ab, "A4|A4b": a4a4b, "A1|A2": a12},
        },
        "direction": {
            "null_model_seed_noise_direction_cosine": null_dir,
            "per_arm": dir_rows,
            "n_arms_above_null_median": n_above,
            "cross_arm_alignment_same_seed": align,
        },
        "init_identity_control": init_ctrl,
        "pos": POS,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    npass = sum(1 for p in POS if p["pass"])
    print("=== E45 ROUND 63 CONSOLIDATION (local) ===")
    print("guards: S3 crosses_zero %d/12 | headline max|dev| %.2e | paired %.2e | diff %.2e"
          % (sum(cz), dev, dev_p, dev_d))
    print("magnitude: seed-noise median %.6f | mechanism median %.6f | ratio %.4f"
          % (sn_med, ef_med, ef_med / sn_med))
    print("  effect distances below seed-noise median: %d/18" % n_effect_below_median_noise)
    for r in rank_rows:
        print("  %-4s effect max %.6f | own seed-noise [%.6f, %.6f] | below own min: %s"
              % (r["arm"], r["effect_max"], r["own_seed_noise_min"],
                 r["own_seed_noise_max"], r["effect_max_below_own_seed_min"]))
    print("cross-arm mutual distance (same seed): A3a|A3b %s | A4|A4b %s | A1|A2 %s"
          % ({k: round(v, 6) for k, v in a3ab.items()},
             {k: round(v, 6) for k, v in a4a4b.items()},
             {k: round(v, 6) for k, v in a12.items()}))
    print("direction: null model median %.4f | arms above null: %d/6" % (null_med, n_above))
    for r in dir_rows:
        print("  %-4s median %+.4f  min %+.4f  negative pairs %d/3  above null: %s"
              % (r["arm"], r["median"], r["min"], r["n_negative_pairs"], r["above_null_median"]))
    print("POS %d/%d" % (npass, len(POS)))
    for p in POS:
        print("  %-46s %s  %s" % (p["name"], "PASS" if p["pass"] else "FAIL", p["detail"]))
    print("VERDICT: %s" % ("PASS" if npass == len(POS) else "FAIL"))
    print("artifact: %s" % OUT)


if __name__ == "__main__":
    main()
