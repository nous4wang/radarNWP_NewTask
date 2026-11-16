import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else "runs/e33_freq_control/summary.json"
d = json.load(open(path))
print("n_seeds", d["n_seeds"])
for s, e in d["per_seed"].items():
    print("seed", s)
    print("  equal_bias_max_abs_bias_error:", e["equal_bias_max_abs_bias_error"])
    print("  equal_bias_scalars[:8]:", e["equal_bias_scalars"][:8])
    print("  negative_fraction_mean:", e["negative_fraction_mean"])
    print("  high_k_energy_ratio_mean:", e["high_k_energy_ratio_mean"])
    print("  high_k_energy_ratio_range:", e["high_k_energy_ratio_range"])
    for arm, pb in e["paired_vs_orig"].items():
        for k in ("0.5", "2.0", "4.0"):
            x = pb[k]
            print("  %-22s %s csi %.6f vs %.6f diff %+.6f CI [%+.6f,%+.6f] excl0=%s bias %.4f vs %.4f truthMism %.0f" % (
                arm, k, x["csi_a"], x["csi_b"], x["diff"], x["ci95_low"], x["ci95_high"],
                x["excludes_zero"], x["bias_a"], x["bias_b"], x["truth_pairing_max_abs_diff"]))
    break
print()
print("=== cross-seed consistency ===")
for arm, v in d.get("cross_seed_consistency", {}).items():
    print("%-22s n=%d pos=%d neg=%d same=%s mean=%+.6f" % (
        arm, v["n_seeds"], v["n_positive"], v["n_negative"], v["all_same_sign"], v["mean_csi2_diff"]))
    print("   per-seed:", v["csi2_diff_per_seed"])
    print("   bias2 pairs:", v["bias2_pairs"])
